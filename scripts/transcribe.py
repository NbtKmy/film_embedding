"""
Transcribe a video file to SRT using mlx-whisper (Apple Silicon).

Usage:
    uv run python scripts/transcribe.py filmes/my_japanese_film.mp4
    uv run python scripts/transcribe.py filmes/my_japanese_film.mp4 --output subtitles/my_japanese_film.srt
    uv run python scripts/transcribe.py filmes/my_japanese_film.mp4 --language en
    uv run python scripts/transcribe.py filmes/my_japanese_film.mp4 --no-separate

Model: mlx-community/whisper-large-v3-mlx (~3GB, downloaded on first run)
Output defaults to subtitles/<video_stem>.srt

Vocal separation (on by default):
  Uses Facebook's demucs (htdemucs model, ~80MB) to separate dialogue from BGM/noise
  before passing audio to Whisper. Greatly reduces hallucinations on old films with
  heavy background music. First run downloads the model (~80MB).
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import mlx_whisper
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    ms = int(round(seconds * 1000))
    h = ms // 3_600_000
    ms -= h * 3_600_000
    m = ms // 60_000
    ms -= m * 60_000
    s = ms // 1_000
    ms -= s * 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def is_hallucination(text: str, prev_texts: list[str], window: int = 2) -> bool:
    """Detect likely hallucination: empty, repeated chars, or same text repeating in context."""
    if not text:
        return True
    # Repeated single character (e.g. "オオオオオ...")
    unique_chars = set(text.replace(" ", "").replace("\u3000", ""))
    if len(unique_chars) <= 2 and len(text) > 6:
        return True
    # Same phrase repeated consecutively in recent context
    if prev_texts.count(text) >= window:
        return True
    return False


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    prev_texts: list[str] = []
    idx = 1
    filtered = 0
    for seg in segments:
        text = seg["text"].strip()
        if is_hallucination(text, prev_texts):
            filtered += 1
            continue
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
        prev_texts.append(text)
        if len(prev_texts) > 10:
            prev_texts.pop(0)
        idx += 1
    if filtered:
        print(f"Filtered {filtered} likely-hallucinated segments.")
    return "\n".join(lines)


def extract_raw_audio(video_path: str, wav_path: str) -> None:
    """Extract audio as mono 44.1kHz WAV (demucs requires 44.1kHz)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "44100",
        "-f", "wav",
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")


def separate_vocals(wav_path: str, out_wav_path: str) -> None:
    """
    Use demucs htdemucs to isolate vocals from BGM.
    Runs on CPU to avoid MPS memory pressure (model is small, ~80MB).
    """
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    print("Loading demucs htdemucs model (downloads ~80MB on first run)...")
    model = get_model("htdemucs")
    model.eval()
    # CPU to avoid competing with Whisper for MPS memory
    device = torch.device("cpu")
    model.to(device)

    print("Separating vocals from BGM (this takes a few minutes)...")
    # Use scipy to avoid torchaudio 2.11+ TorchCodec dependency
    sr, data = wavfile.read(wav_path)
    # Normalize to float32 [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    wav = torch.from_numpy(data).unsqueeze(0)  # (1, samples) mono

    # demucs expects stereo — duplicate mono channel
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # Resample to model's expected sample rate if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)

    wav = wav.unsqueeze(0).to(device)  # (1, 2, samples)

    with torch.no_grad():
        sources = apply_model(model, wav, device=device, progress=True)
    # sources shape: (1, num_sources, 2, samples)
    # htdemucs sources: drums, bass, other, vocals
    source_names = model.sources
    vocal_idx = source_names.index("vocals")
    vocals = sources[0, vocal_idx]  # (2, samples)

    # Back to mono
    vocals_mono = vocals.mean(dim=0, keepdim=True)  # (1, samples)
    vocals_np = (vocals_mono.cpu().squeeze().numpy() * 32768.0).astype(np.int16)
    wavfile.write(out_wav_path, model.samplerate, vocals_np)
    print("Vocal separation done.\n")


def transcribe(video_path: str, output_path: str, language: str = "ja", separate: bool = True) -> None:
    print(f"Transcribing: {video_path}")
    print(f"Language: {language}")
    print(f"Model: mlx-community/whisper-large-v3-mlx (downloading if needed...)\n")

    audio_input = video_path
    tmp_files: list[Path] = []

    if separate:
        raw_wav = Path(tempfile.mktemp(suffix="_raw.wav"))
        vocals_wav = Path(tempfile.mktemp(suffix="_vocals.wav"))
        tmp_files = [raw_wav, vocals_wav]
        try:
            print("Extracting audio...")
            extract_raw_audio(video_path, str(raw_wav))
            separate_vocals(str(raw_wav), str(vocals_wav))
            audio_input = str(vocals_wav)
        except Exception as e:
            print(f"Warning: vocal separation failed ({e}), falling back to raw audio.")
            audio_input = video_path

    try:
        result = mlx_whisper.transcribe(
            audio_input,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language=language,
            word_timestamps=False,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
            compression_ratio_threshold=2.0,
            no_speech_threshold=0.5,
        )
    finally:
        for f in tmp_files:
            f.unlink(missing_ok=True)

    segments = result.get("segments", [])
    if not segments:
        print("Warning: no segments returned. The audio may be silent or unsupported.")
        return

    srt_content = segments_to_srt(segments)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(srt_content, encoding="utf-8")
    print(f"Saved {len(segments)} segments → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe video to SRT via mlx-whisper.")
    parser.add_argument("video", help="Path to the video file (MP4, MKV, etc.)")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output SRT path (default: subtitles/<video_stem>.srt)",
    )
    parser.add_argument(
        "--language", "-l",
        default="ja",
        help="Language code (default: ja for Japanese)",
    )
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="Skip demucs vocal separation (use raw video audio)",
    )
    args = parser.parse_args()

    video = Path(args.video)
    output = args.output or str(Path("subtitles") / f"{video.stem}.srt")

    transcribe(str(video), output, language=args.language, separate=not args.no_separate)
