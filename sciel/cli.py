"""
cli.py — Command-line interface for Sciel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import (
    DEFAULT_DIFF_THRESHOLD, DEFAULT_FORCE_INTERVAL,
    DEFAULT_FRAME_INTERVAL, DEFAULT_FRAMES_DIR, DEFAULT_MAX_FRAMES,
)
from .report import print_report, write_json
from .scanner import scan

BANNER = r"""
        ___
     ☀ /   \ 🌙
      \_◑_◐_/    S C I E L
   clair-obscur secret scanner
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sciel",
        description="Sciel — Scan images, videos and PDFs for cleartext secrets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  sciel ./evidence/
  sciel ./evidence/ --vision --out report.json
  sciel recording.mp4 --frame-interval 0.5 --vision
  sciel screenshot.png --lang eng
""",
    )
    parser.add_argument(
        "target",
        type=Path,
        help="File or directory to scan (recursive).",
    )
    parser.add_argument(
        "--lang",
        default="fra+eng",
        metavar="LANG",
        help="Tesseract language(s) (default: fra+eng).",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=DEFAULT_FRAME_INTERVAL,
        metavar="SEC",
        help=f"Video sampling interval in seconds (default: {DEFAULT_FRAME_INTERVAL}).",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=DEFAULT_DIFF_THRESHOLD,
        metavar="RATIO",
        help=(
            f"Changed-pixel ratio to declare a new scene (default: {DEFAULT_DIFF_THRESHOLD}). "
            "Calibrated to detect even a single line of text changing on a static background."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=DEFAULT_MAX_FRAMES,
        metavar="N",
        help=f"Max frames sampled per video (default: {DEFAULT_MAX_FRAMES}).",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path(DEFAULT_FRAMES_DIR),
        metavar="DIR",
        help=f"Directory for saving suspect video frames (default: ./{DEFAULT_FRAMES_DIR}).",
    )
    parser.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Don't save video frames; report timecodes only.",
    )
    parser.add_argument(
        "--force-interval",
        type=float,
        default=DEFAULT_FORCE_INTERVAL,
        metavar="SEC",
        help=(
            f"Safety-net: force-process a video frame every N seconds even when "
            f"the scene appears identical (default: {DEFAULT_FORCE_INTERVAL}). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--vision",
        action="store_true",
        help=(
            "Enable Vision LLM as fallback for hard cases: handwritten text, "
            "post-its, real-world camera footage, heavily compressed frames. "
            "Called automatically when the origin classifier detects a real image."
        ),
    )
    parser.add_argument(
        "--vision-all",
        action="store_true",
        help=(
            "Like --vision but forces Vision on every file, even digital ones. "
            "Slower but maximises coverage."
        ),
    )
    parser.add_argument(
        "--vision-backend",
        default="anthropic",
        choices=["anthropic", "openai", "ollama"],
        metavar="BACKEND",
        help=(
            "Vision LLM backend (default: anthropic). "
            "anthropic: Claude Vision — needs ANTHROPIC_API_KEY. "
            "openai: GPT-4o — needs OPENAI_API_KEY. "
            "ollama: local model, no key needed — install https://ollama.ai "
            "then: ollama pull qwen2-vl"
        ),
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen2-vl",
        metavar="MODEL",
        help=(
            "Ollama model to use (default: qwen2-vl). "
            "Good choices: qwen2-vl (best OCR), llava, llama3.2-vision, moondream."
        ),
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        metavar="URL",
        help="Ollama server URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write results to a JSON file.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.target.exists():
        print(f"Error: {args.target} does not exist.", file=sys.stderr)
        sys.exit(1)

    use_vision  = args.vision or args.vision_all
    vision_all  = args.vision_all
    backend     = args.vision_backend
    ollama_model = args.ollama_model
    ollama_host  = args.ollama_host

    if use_vision:
        from .vision import check_backend
        ok, msg = check_backend(backend, ollama_host=ollama_host)
        status = "✅" if ok else "⚠️ "
        print(f"  [~] Vision backend: {backend} — {status} {msg}")
        if not ok:
            print(f"  [~] Continuing without Vision (Tesseract only).")
            use_vision = False
        elif vision_all:
            print("  [~] --vision-all: Vision will run on every file")

    frames_dir = None if args.no_save_frames else args.frames_dir

    reports = scan(
        target=args.target,
        lang=args.lang,
        frame_interval=args.frame_interval,
        diff_threshold=args.diff_threshold,
        max_frames=args.max_frames,
        frames_out_dir=frames_dir,
        force_interval=args.force_interval,
        use_vision=use_vision,
        vision_all=vision_all,
        vision_backend=backend,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
    )

    print_report(reports)

    if args.out:
        write_json(reports, args.out)
