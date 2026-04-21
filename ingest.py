"""
Ingest a film into the vector store.

Usage — full pipeline:
    uv run python ingest.py \
        --video filmes/little_shop_of_horrors.mp4 \
        --subtitles subtitles/little_shop_of_horrors.txt \
        --film-id little_shop_of_horrors_1960 \
        --title "Little Shop of Horrors" \
        --year 1960 \
        --director "Roger Corman"

Usage — two-step (recommended: isolates slow ffmpeg work from embedding):
    # Step 1: scene detection + thumbnail/frame extraction (~20 min)
    uv run python ingest.py --step extract \
        --video filmes/little_shop_of_horrors.mp4 \
        --subtitles subtitles/little_shop_of_horrors.txt \
        --film-id little_shop_of_horrors_1960

    # Step 2: embedding + Qdrant upsert (requires step 1 cache)
    uv run python ingest.py --step embed \
        --film-id little_shop_of_horrors_1960 \
        --title "Little Shop of Horrors" \
        --year 1960 \
        --director "Roger Corman"

Resume after interruption: just run the same command again — completed steps are skipped.

Cache files (under cache/):
  {film_id}_scenes.json          scene metadata (created after thumbnail extraction)
  {film_id}_emb_{N:06d}.npy      embeddings for batch starting at index N
Cache is deleted automatically after a successful Qdrant upsert.
"""
import argparse
import json
import re
import subprocess
import uuid
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams
from scenedetect import AdaptiveDetector, detect

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "film_scenes"
EMBEDDING_DIM = 4096
THUMBNAIL_DIR = Path("thumbnails")
FRAMES_DIR = Path("frames")
CACHE_DIR = Path("cache")
FRAMES_PER_SEC = 0.5  # 1 frame per 2 seconds
MAX_SCENE_FRAMES = 8
BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Subtitle helpers
# ---------------------------------------------------------------------------

def parse_srt(subtitle_path: str) -> list[dict]:
    """Parse SRT file → list of {start, end, text} dicts (times in seconds)."""
    with open(subtitle_path, encoding="utf-8") as f:
        content = f.read()

    def to_seconds(ts: str) -> float:
        h, m, s = ts.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    pattern = re.compile(
        r"\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)",
        re.DOTALL,
    )
    results = []
    for m in pattern.finditer(content):
        text = re.sub(r"<[^>]+>", "", m.group(3)).strip().replace("\n", " ")
        if text:
            results.append({
                "start": to_seconds(m.group(1)),
                "end": to_seconds(m.group(2)),
                "text": text,
            })
    return results


def align_subtitles(subtitles: list[dict], start: float, end: float) -> str:
    """Collect all subtitle text overlapping with the scene [start, end]."""
    return " ".join(
        s["text"] for s in subtitles if s["end"] > start and s["start"] < end
    )


# ---------------------------------------------------------------------------
# Thumbnail helper
# ---------------------------------------------------------------------------

def extract_thumbnail(video_path: str, time_sec: float, output_path: Path) -> str:
    """Extract a single frame at time_sec from the video."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(time_sec),
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
    )
    return str(output_path)


def extract_frames(video_path: str, start: float, end: float, out_dir: Path, n: int) -> list[str]:
    """Extract n evenly-spaced frames from [start, end] using ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = end - start
    # For very short scenes, extract at least 1 frame (the midpoint)
    timestamps = (
        [start + duration * i / (n - 1) for i in range(n)]
        if duration > 0 and n > 1
        else [start + duration / 2]
    )
    paths = []
    for j, t in enumerate(timestamps):
        out_path = out_dir / f"frame_{j:02d}.png"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(t),
                "-i", video_path,
                "-vframes", "1",
                "-q:v", "2",
                str(out_path),
            ],
            check=True,
            capture_output=True,
        )
        paths.append(str(out_path.resolve()))
    return paths


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def scenes_cache_path(film_id: str) -> Path:
    return CACHE_DIR / f"{film_id}_scenes.json"


def emb_cache_path(film_id: str, batch_start: int) -> Path:
    return CACHE_DIR / f"{film_id}_emb_{batch_start:06d}.npy"


def load_cached_scenes(film_id: str) -> list[dict] | None:
    p = scenes_cache_path(film_id)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def save_scenes_cache(film_id: str, scenes: list[dict]):
    CACHE_DIR.mkdir(exist_ok=True)
    with open(scenes_cache_path(film_id), "w") as f:
        json.dump(scenes, f)


def load_cached_embedding(film_id: str, batch_start: int) -> np.ndarray | None:
    p = emb_cache_path(film_id, batch_start)
    if p.exists():
        return np.load(p)
    return None


def save_embedding_cache(film_id: str, batch_start: int, vecs: np.ndarray):
    CACHE_DIR.mkdir(exist_ok=True)
    np.save(emb_cache_path(film_id, batch_start), vecs)


def clear_cache(film_id: str):
    for p in CACHE_DIR.glob(f"{film_id}_*.json"):
        p.unlink()
    for p in CACHE_DIR.glob(f"{film_id}_emb_*.npy"):
        p.unlink()


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def ingest(
    video_path: str | None,
    subtitle_path: str | None,
    film_id: str,
    title: str | None,
    year: int | None,
    director: str | None,
    genres: list[str] | None = None,
    step: str = "all",
    delete_existing: bool = False,
):
    run_extract = step in ("extract", "all")
    run_embed = step in ("embed", "all")

    # -- Step 1-3: scene detection + thumbnails (skipped if cache exists) ---
    if run_extract:
        cached_scenes = load_cached_scenes(film_id)
        if cached_scenes is not None:
            scenes = cached_scenes
            print(f"[1-3] Loaded {len(scenes)} scenes from cache — skipping detection & thumbnails.")
        else:
            if not video_path or not subtitle_path:
                raise ValueError("--video and --subtitles are required for the extract step.")
            print(f"[1] Detecting scenes in {video_path} ...")
            scene_list = detect(video_path, AdaptiveDetector())
            print(f"    Found {len(scene_list)} scenes.")

            print("[2] Parsing subtitles ...")
            subtitles = parse_srt(subtitle_path)
            print(f"    Parsed {len(subtitles)} subtitle entries.")

            print("[3] Extracting thumbnails and frames ...")
            scenes = []
            for i, (start_tc, end_tc) in enumerate(scene_list):
                start = start_tc.get_seconds()
                end = end_tc.get_seconds()
                mid = (start + end) / 2
                thumb_path = THUMBNAIL_DIR / f"{film_id}_scene_{i + 1:04d}.png"
                frames_dir = FRAMES_DIR / f"{film_id}_scene_{i + 1:04d}"
                thumbnail_path = extract_thumbnail(video_path, mid, thumb_path)
                n_frames = max(1, min(MAX_SCENE_FRAMES, round((end - start) * FRAMES_PER_SEC)))
                frame_paths = extract_frames(video_path, start, end, frames_dir, n_frames)
                subtitle_text = align_subtitles(subtitles, start, end)
                scenes.append({
                    "scene_number": i + 1,
                    "start_time": start,
                    "end_time": end,
                    "duration": end - start,
                    "thumbnail_path": thumbnail_path,
                    "frame_paths": frame_paths,
                    "subtitle_text": subtitle_text,
                })
                if (i + 1) % 10 == 0:
                    print(f"    {i + 1}/{len(scene_list)} scenes extracted.")
            print(f"    Done: {len(scenes)} scenes.")
            save_scenes_cache(film_id, scenes)

        if step == "extract":
            print(f"\nExtract step complete. Run with --step embed to continue.")
            return

    if step == "embed":
        scenes = load_cached_scenes(film_id)
        if scenes is None:
            raise RuntimeError(
                f"No scene cache found for '{film_id}'. Run --step extract first."
            )

    if run_embed:
        if not title or year is None or not director:
            raise ValueError("--title, --year, and --director are required for the embed step.")

    # -- Step 4: embedding (batch-level resume) -----------------------------
    print("[4/5] Embedding scenes ...")
    embedder = None  # load lazily — avoid loading if all batches are cached

    all_embeddings: list[list[float]] = []
    for batch_start in range(0, len(scenes), BATCH_SIZE):
        cached = load_cached_embedding(film_id, batch_start)
        if cached is not None:
            all_embeddings.extend(cached.tolist())
            done = min(batch_start + BATCH_SIZE, len(scenes))
            print(f"      {done}/{len(scenes)} scenes — loaded from cache.")
            continue

        if embedder is None:
            print("      Loading embedding model ...")
            embedder = Qwen3VLEmbedder("nkamiy/Qwen3-VL-Embedding-8B-8bit-mlx")

        batch = scenes[batch_start : batch_start + BATCH_SIZE]
        inputs = [
            {"video": s["frame_paths"], "text": s["subtitle_text"]}
            if s["subtitle_text"]
            else {"video": s["frame_paths"]}
            for s in batch
        ]
        vecs = embedder.process(inputs)
        save_embedding_cache(film_id, batch_start, vecs)
        all_embeddings.extend(vecs.tolist())
        done = min(batch_start + BATCH_SIZE, len(scenes))
        print(f"      {done}/{len(scenes)} scenes embedded.")

    # -- Step 5: upsert to Qdrant ------------------------------------------
    print("[5/5] Upserting to Qdrant ...")
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

    if delete_existing and client.collection_exists(COLLECTION_NAME):
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="film_id", match=MatchValue(value=film_id))])
            ),
        )
        print(f"  Deleted existing records for '{film_id}'.")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "film_id": film_id,
                "film_title": title,
                "film_year": year,
                "film_director": director,
                "film_genre": genres or [],
                "scene_id": f"{film_id}_scene_{scene['scene_number']:04d}",
                "scene_number": scene["scene_number"],
                "start_time": scene["start_time"],
                "end_time": scene["end_time"],
                "duration": scene["duration"],
                "subtitle_text": scene["subtitle_text"],
                "thumbnail_path": scene["thumbnail_path"],
                "video_path": video_path,
            },
        )
        for scene, vec in zip(scenes, all_embeddings)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"\nDone. {len(points)} scenes ingested for '{title}'.")

    clear_cache(film_id)
    print("Cache cleared.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a film into the vector store.")
    parser.add_argument(
        "--step",
        choices=["extract", "embed", "all"],
        default="all",
        help="extract: scene detection + thumbnails only; embed: embedding + upsert only; all: full pipeline (default)",
    )
    parser.add_argument("--video", default=None, help="Path to the MP4 file (required for extract/all)")
    parser.add_argument("--subtitles", default=None, help="Path to the SRT subtitle file (required for extract/all)")
    parser.add_argument("--film-id", required=True, help="Unique film identifier")
    parser.add_argument("--title", default=None, help="Film title (required for embed/all)")
    parser.add_argument("--year", type=int, default=None, help="Release year (required for embed/all)")
    parser.add_argument("--director", default=None, help="Director name (required for embed/all)")
    parser.add_argument("--genres", nargs="*", default=[], help="Genre list")
    parser.add_argument("--delete-existing", action="store_true", help="Delete existing records for this film before upserting")
    args = parser.parse_args()

    ingest(
        video_path=args.video,
        subtitle_path=args.subtitles,
        film_id=args.film_id,
        title=args.title,
        year=args.year,
        director=args.director,
        genres=args.genres,
        step=args.step,
        delete_existing=args.delete_existing,
    )
