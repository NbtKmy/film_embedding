"""
Search for film scenes using natural language.

Usage:
    uv run python search.py "a man arguing with a plant"
    uv run python search.py "a man arguing with a plant" --top-k 10
    uv run python search.py "comedic scene in a flower shop" --film-id little_shop_of_horrors_1960
"""
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Query

from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "film_scenes"

_embedder: Qwen3VLEmbedder | None = None


def get_embedder() -> Qwen3VLEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = Qwen3VLEmbedder("nkamiy/Qwen3-VL-Embedding-8B-8bit-mlx")
    return _embedder


def search(
    query: str,
    top_k: int = 5,
    film_id: str | None = None,
) -> list[dict]:
    embedder = get_embedder()
    client = QdrantClient(path=QDRANT_PATH)

    query_vec = embedder.process([{"text": query}])[0].tolist()

    query_filter = None
    if film_id:
        query_filter = Filter(
            must=[FieldCondition(key="film_id", match=MatchValue(value=film_id))]
        )

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points
    return [{"score": h.score, **h.payload} for h in hits]


def display_results(results: list[dict]):
    if not results:
        print("No results found.")
        return
    for i, r in enumerate(results, 1):
        print(f"\n{'=' * 60}")
        print(f"#{i}  Score: {r['score']:.4f}")
        print(f"    Film:  {r['film_title']} ({r['film_year']}) — dir. {r['film_director']}")
        print(f"    Scene: {r['scene_number']}  [{r['start_time']:.1f}s – {r['end_time']:.1f}s]  ({r['duration']:.1f}s)")
        subtitle = r["subtitle_text"]
        if subtitle:
            print(f"    Text:  {subtitle[:120]}{'...' if len(subtitle) > 120 else ''}")
        print(f"    Thumb: {r['thumbnail_path']}")
        print(f"    Play:  {r['video_path']}  (seek to {r['start_time']:.1f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for film scenes.")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--film-id", default=None, help="Filter by film ID")
    args = parser.parse_args()

    results = search(args.query, top_k=args.top_k, film_id=args.film_id)
    display_results(results)
