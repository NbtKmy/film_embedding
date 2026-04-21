# Installation

## Requirements

- macOS with Apple Silicon (M1 or later)
- [uv](https://github.com/astral-sh/uv) — Python package manager
- ffmpeg

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install ffmpeg
brew install ffmpeg
```

## Setup

```bash
git clone https://github.com/NbtKmy/film_embedding.git
cd film_embedding
uv sync
```

## Running the demo (with pre-built data)

If you received the data archive (video files, thumbnails, and vector database), extract it into the project root. You should end up with:

```
film_embedding/
├── filmes/          # video files (.mp4)
├── thumbnails/      # scene thumbnails (.png)
└── qdrant_db/       # vector database
```

Then launch the Streamlit app:

```bash
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Ingesting new films

See [README.md](README.md) for the full ingest workflow (scene detection, transcription, embedding).

> **Note:** Embedding requires the MLX model `jedisct1/Qwen3-VL-Embedding-8B-mlx` (~8.7 GB), which is downloaded automatically on first run. The ingest pipeline uses Apple Silicon GPU via MLX and requires ~10 GB of free RAM.
