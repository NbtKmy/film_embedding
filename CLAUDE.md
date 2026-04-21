# film_embedding

## Project Goal

Build a vector store for multiple films that enables natural language search over films and individual scenes.

- **Prototype**: ~10 films
- **Production target**: ~10,000 films
- **Hardware**: macOS M4, 24GB RAM

---

## Architecture Overview

```
MP4 + SRT subtitle
  ↓  scenedetect (AdaptiveDetector)  → scene boundary list
  ↓  ffmpeg                          → per-scene thumbnail (PNG) + clip (MP4)
  ↓  subtitle alignment              → subtitle text per scene
  ↓  Qwen3-VL-Embedding-8B-mlx      → embedding vector (image + text)
  ↓  Qdrant (local file mode)        → vector + metadata stored
                                       ↑
                         natural language query
                         → embed query (text only)
                         → Qdrant similarity search
                         → display thumbnails + metadata
                         → user selects scene → play clip
```

---

## Package Management

This project is managed with [uv](https://github.com/astral-sh/uv). Use `uv run` to execute scripts and `uv add` to install dependencies. Do **not** use `pip` or `python` directly.

```bash
uv run python ingest.py ...
uv add <package>
```

---

## Tech Stack

| Component | Choice | Notes |
|---|---|---|
| Runtime | Python 3.13 + uv | |
| Scene detection | `scenedetect[opencv]` | `AdaptiveDetector` |
| Frame/clip extraction | `ffmpeg` via subprocess | |
| Embedding model | `Qwen/Qwen3-VL-Embedding-8B` | ~16GB fp32 / ~8GB fp16, Apache 2.0 |
| Embedding runtime | `transformers` + PyTorch MPS | Apple Silicon MPS backend |
| Vector store | `qdrant-client` | Local file mode, no server needed |
| Image handling | `Pillow` | Thumbnail processing |

### Embedding model note

`Qwen3-VL-Embedding-8B` is a multimodal embedding model — it accepts image + text together and returns a single cosine-normalized vector (dim=4096). The embedding logic lives in `scripts/qwen3_vl_embedding.py` (adapted from the official Qwen model scripts), which defines `Qwen3VLForEmbedding` and `Qwen3VLEmbedder`. MPS device support for Apple Silicon has been added.

```python
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

embedder = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-8B")
vecs = embedder.process([
    {"image": "thumbnails/scene_0001.png", "text": "dialogue here"},
    {"text": "text-only query"},
])
# vecs.shape == (2, 4096), L2-normalized
```

**No reranker** is used in the prototype — Qdrant cosine search results are returned directly. A reranker (e.g. Qwen3-VL-Reranker-2B) can be added later if search quality needs improvement.

### Alternative: MLX backend (Plan B if PyTorch MPS runs OOM)

Current implementation uses `transformers` + PyTorch MPS (`torch_dtype=torch.float16`). If OOM persists, switch to:

- **Library**: [`mlx-embeddings`](https://github.com/Blaizzy/mlx-embeddings) — supports Qwen3-VL multimodal (image+text) embeddings natively on Apple Silicon
- **Model**: `jedisct1/Qwen3-VL-Embedding-8B-mlx` (8.7GB, 8-bit MLX) — MLX-converted 8B model
  - **Caveat**: may be based on Qwen3-VL-8B-Instruct rather than the dedicated Embedding variant; verify before committing
- **Usage**:
  ```python
  from mlx_embeddings import load
  model, processor = load("jedisct1/Qwen3-VL-Embedding-8B-mlx")
  ```
- This would require rewriting `scripts/qwen3_vl_embedding.py` to use the mlx-embeddings API

**Other options investigated and ruled out (as of 2026-04):**
- GGUF via Ollama: llama.cpp has no multimodal embedding support yet (PR #18665 draft); Ollama image+text embedding is an open request (#5304)
- `mlx-community/Qwen3-VL-Embedding-2B-8bit`: 2B only, smaller embedding dim, lower quality

---

## Metadata Schema (Qdrant payload)

Each vector point in Qdrant stores the following payload:

```json
{
  "film_id":        "little_shop_of_horrors_1960",
  "film_title":     "Little Shop of Horrors",
  "film_year":      1960,
  "film_director":  "Roger Corman",
  "film_genre":     ["comedy", "horror"],
  "scene_id":       "lsoh_scene_003",
  "scene_number":   3,
  "start_time":     142.5,
  "end_time":       198.0,
  "duration":       55.5,
  "subtitle_text":  "raw subtitle text for this scene",
  "thumbnail_path": "./thumbnails/lsoh_scene_003.png",
  "video_path":     "./filmes/little_shop_of_horrors.mp4"
}
```

---

## Data Pipeline

### Step 1 — Scene detection
```bash
python scene_separation.py filmes/little_shop_of_horrors.mp4 scenes/
```
Outputs: split scene clips into `scenes/`

The ingest pipeline can also be run in two separate steps to avoid re-running slow ffmpeg extraction on embedding errors:
```bash
# Step 1: scene detection + thumbnails (~20 min, saves cache)
python ingest.py --step extract --video filmes/foo.mp4 --subtitles subtitles/foo.srt --film-id foo_1960

# Step 2: embedding + Qdrant upsert (reads from cache)
python ingest.py --step embed --film-id foo_1960 --title "Foo" --year 1960 --director "Bar"
```

### Step 2 — Thumbnail extraction
Extract one representative frame per scene (e.g. middle frame) using ffmpeg.
Output: `thumbnails/{scene_id}.png`

### Step 3 — Subtitle alignment
Parse SRT file → map each subtitle line to a scene using timestamp overlap.
Concatenate all subtitle lines whose `start_time` falls within `[scene.start, scene.end]`.

### Step 4 — Embedding
For each scene: pass `(thumbnail_image, subtitle_text)` to Qwen3-VL-Embedding → extract vector.

### Step 5 — Upsert to Qdrant
```python
from qdrant_client import QdrantClient
client = QdrantClient(path="./qdrant_db")
```
Create collection with cosine distance. Upsert vectors + payloads.

---

## Project File Structure

```
film_embedding/
├── CLAUDE.md
├── pyproject.toml          # add: qdrant-client, Pillow
├── main.py                 # CLI entry point (wraps search.py)
├── scene_separation.py     # optional: split video into clip files
├── ingest.py               # full pipeline: detect → thumbnail → embed → upsert
├── search.py               # query embedding → Qdrant search → display results
├── scripts/
│   └── qwen3_vl_embedding.py  # Qwen3VLEmbedder (MPS-patched official script)
├── filmes/                 # input video files
├── subtitles/              # SRT subtitle files
├── thumbnails/             # extracted scene thumbnails (gitignored)
└── qdrant_db/              # Qdrant local storage (gitignored)
```

---

## Additional Dependencies to Install

```bash
uv add qdrant-client Pillow
```

---

## Known Issues / TODOs

- Playback is handled by seeking to `start_time` in the original video — no scene splitting is needed for the prototype.
- Embedding dimension is fixed at 4096 (hardcoded in `ingest.py` as `EMBEDDING_DIM`).
- For production (10k films), consider Qdrant server mode instead of local file mode for concurrent access and better performance.
- A reranker (Qwen3-VL-Reranker-2B) can be added post-search when RAM budget allows both models.
- `ingest.py --step extract/embed/all` allows running scene extraction and embedding as separate steps, so embedding errors don't force a re-run of the slow ffmpeg step.
