"""
Streamlit UI for film scene search.

Run with:
    uv run streamlit run app.py
"""
import streamlit as st
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "film_scenes"
GRID_COLS = 4


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource
def get_qdrant() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedder():
    from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
    return Qwen3VLEmbedder("nkamiy/Qwen3-VL-Embedding-8B-8bit-mlx")


@st.cache_data(ttl=60)
def load_all_scenes() -> list[dict]:
    client = get_qdrant()
    records, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10_000,
        with_payload=True,
        with_vectors=False,
    )
    scenes: list[dict] = [r.payload for r in records if r.payload is not None]
    return sorted(scenes, key=lambda s: (s.get("film_title", ""), s.get("scene_number", 0)))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def semantic_search(query: str, filters: dict, top_k: int = 20) -> list[dict]:
    embedder = get_embedder()
    client = get_qdrant()

    query_vec = embedder.process([{"text": query}])[0].tolist()

    must = []
    if filters.get("films"):
        must.append(FieldCondition(key="film_title", match=MatchAny(any=filters["films"])))
    if filters.get("directors"):
        must.append(FieldCondition(key="film_director", match=MatchAny(any=filters["directors"])))
    if filters.get("years"):
        must.append(FieldCondition(key="film_year", match=MatchAny(any=filters["years"])))

    query_filter = Filter(must=must) if must else None

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    ).points

    return [{"score": h.score, **(h.payload or {})} for h in hits]


def apply_filters(scenes: list[dict], filters: dict) -> list[dict]:
    results = scenes
    if filters.get("films"):
        results = [s for s in results if s.get("film_title") in filters["films"]]
    if filters.get("directors"):
        results = [s for s in results if s.get("film_director") in filters["directors"]]
    if filters.get("years"):
        results = [s for s in results if s.get("film_year") in filters["years"]]
    return results


# ---------------------------------------------------------------------------
# Modal dialog
# ---------------------------------------------------------------------------

@st.dialog("Scene Player", width="large")
def play_scene(scene: dict):
    st.subheader(f"{scene.get('film_title', '?')} ({scene.get('film_year', '?')})")
    st.caption(
        f"Scene {scene.get('scene_number')} · "
        f"dir. {scene.get('film_director', '?')} · "
        f"{scene.get('start_time', 0):.0f}s – {scene.get('end_time', 0):.0f}s "
        f"({scene.get('duration', 0):.0f}s)"
    )

    video_path = scene.get("video_path")
    start = int(scene.get("start_time", 0))

    if video_path:
        st.video(video_path, start_time=start)
    else:
        st.warning("video_path が見つかりません。ingest 時のペイロードを確認してください。")

    subtitle = scene.get("subtitle_text", "").strip()
    if subtitle:
        st.markdown(f"> {subtitle}")


# ---------------------------------------------------------------------------
# Scene card
# ---------------------------------------------------------------------------

def render_card(scene: dict, idx: int):
    thumb = scene.get("thumbnail_path", "")
    if thumb and Path(thumb).exists():
        st.image(thumb, use_container_width=True)
    else:
        st.markdown(
            "<div style='height:120px;background:#1e1e1e;border-radius:6px;"
            "display:flex;align-items:center;justify-content:center;"
            "font-size:2rem'>🎞️</div>",
            unsafe_allow_html=True,
        )

    score = scene.get("score")
    score_str = f"  `{score:.2f}`" if score is not None else ""
    st.markdown(
        f"**{scene.get('film_title', '?')}** ({scene.get('film_year', '?')}){score_str}"
    )
    st.caption(
        f"Scene {scene.get('scene_number')} · "
        f"{scene.get('start_time', 0):.0f}s–{scene.get('end_time', 0):.0f}s"
    )

    subtitle = scene.get("subtitle_text", "").strip()
    if subtitle:
        snippet = subtitle[:80] + ("…" if len(subtitle) > 80 else "")
        st.caption(f'"{snippet}"')

    if st.button("▶ Play", key=f"play_{idx}", use_container_width=True):
        play_scene(scene)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Film Scene Search",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Ensure static symlinks exist
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    for name in ("filmes", "thumbnails"):
        link = static_dir / name
        if not link.exists():
            link.symlink_to(f"../{name}")

    all_scenes = load_all_scenes()

    # --- Sidebar: facets ---
    with st.sidebar:
        st.title("🎬 Filters")

        film_options = sorted({s.get("film_title", "") for s in all_scenes} - {""})
        director_options = sorted({s.get("film_director", "") for s in all_scenes} - {""})
        year_options = sorted({s.get("film_year") for s in all_scenes if s.get("film_year") is not None})

        selected_films = st.multiselect("Film", film_options)
        selected_directors = st.multiselect("Director", director_options)
        selected_years = st.multiselect("Year", year_options)

        st.divider()
        top_k = st.slider("Max results", min_value=4, max_value=50, value=20, step=4)

    filters = {
        "films": selected_films,
        "directors": selected_directors,
        "years": selected_years,
    }

    # --- Main: search bar ---
    st.title("Film Scene Search")
    query = st.text_input(
        "query",
        placeholder="Describe a scene — e.g. 'a man arguing with a plant'",
        label_visibility="collapsed",
    )

    # --- Compute results ---
    if query:
        with st.spinner("Embedding query and searching…"):
            results = semantic_search(query, filters, top_k=top_k)
    else:
        results = apply_filters(all_scenes, filters)[:top_k]

    # --- Status line ---
    if query:
        st.caption(f"{len(results)} scenes · semantic search")
    else:
        st.caption(f"{len(results)} scenes · showing all")

    if not results:
        st.info("No scenes found. Try a different query or adjust the filters.")
        return

    # --- Results grid ---
    cols = st.columns(GRID_COLS)
    for i, scene in enumerate(results):
        with cols[i % GRID_COLS]:
            render_card(scene, i)


if __name__ == "__main__":
    main()
