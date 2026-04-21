# film_embedding

Vector store for natural language search over films and individual scenes.

## Requirements

- macOS (Apple Silicon)
- Python 3.13 + [uv](https://github.com/astral-sh/uv)
- ffmpeg (`brew install ffmpeg`)

## Setup

```bash
uv sync
```

## Workflow

### 1. Prepare subtitles

For films without SRT files, transcribe the audio using Whisper:

```bash
uv run python scripts/transcribe.py filmes/my_film.mp4
# Output: subtitles/my_film.srt
```

Options:

```bash
# Specify output path
uv run python scripts/transcribe.py filmes/my_film.mp4 --output subtitles/custom.srt

# Change language (default: ja)
uv run python scripts/transcribe.py filmes/my_film.mp4 --language en
```

Model: `mlx-community/whisper-large-v3-mlx` (~3GB, downloaded automatically on first run).

### 1b. Manually editing SRT files

Whisper transcription may contain hallucinations or mishear dialect/accents — some manual correction is expected. Edit the SRT file directly in any text editor.

SRT format:

```
1
00:00:05,000 --> 00:00:08,500
Subtitle text here.

2
00:00:09,000 --> 00:00:12,000
Next subtitle.
```

The sequence numbers (1, 2, 3...) are not used for playback — only the timestamps matter. If your edits leave gaps or duplicates in the numbering, renumber the file:

```bash
# Renumber in place
uv run python scripts/renumber_srt.py subtitles/my_film.srt

# Or write to a new file
uv run python scripts/renumber_srt.py subtitles/my_film.srt subtitles/my_film_fixed.srt
```

### 2. Ingest a film

```bash
# Full pipeline
uv run python ingest.py \
    --video filmes/my_film.mp4 \
    --subtitles subtitles/my_film.srt \
    --film-id my_film_1960 \
    --title "My Film" \
    --year 1960 \
    --director "Director Name"

# Two-step (recommended: avoids re-running slow ffmpeg on embedding errors)
uv run python ingest.py --step extract \
    --video filmes/my_film.mp4 \
    --subtitles subtitles/my_film.srt \
    --film-id my_film_1960

uv run python ingest.py --step embed \
    --video filmes/my_film.mp4 \
    --subtitles subtitles/my_film.srt \
    --film-id my_film_1960 \
    --title "My Film" \
    --year 1960 \
    --director "Director Name"
```

### 3. Search

```bash
uv run python main.py "a scene where someone eats a plant"
```

## Project Structure

```
film_embedding/
├── ingest.py               # Full pipeline: detect → thumbnail → embed → upsert
├── search.py               # Query embedding → Qdrant search → display results
├── main.py                 # CLI entry point
├── scene_separation.py     # Optional: split video into scene clips
├── scripts/
│   ├── transcribe.py       # Audio transcription → SRT (mlx-whisper)
│   ├── renumber_srt.py     # Renumber SRT sequence numbers after manual edits
│   └── qwen3_vl_embedding.py  # Qwen3-VL embedder (MLX)
├── filmes/                 # Input video files
├── subtitles/              # SRT subtitle files
├── thumbnails/             # Extracted scene thumbnails (gitignored)
└── qdrant_db/              # Qdrant local storage (gitignored)
```
