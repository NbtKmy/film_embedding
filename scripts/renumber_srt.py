"""Renumber SRT subtitle sequence numbers sequentially starting from 1."""

import sys
from pathlib import Path


def renumber_srt(input_path: Path, output_path: Path | None = None) -> None:
    text = input_path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]

    out = []
    for i, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if lines[0].strip().isdigit():
            lines[0] = str(i)
        out.append("\n".join(lines))

    result = "\n\n".join(out) + "\n"

    dest = output_path or input_path
    dest.write_text(result, encoding="utf-8")
    print(f"Renumbered {len(out)} entries -> {dest}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/renumber_srt.py <input.srt> [output.srt]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    renumber_srt(input_path, output_path)
