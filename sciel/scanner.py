"""
scanner.py — Top-level scan orchestration.

Walks a file or directory, dispatches each file to the appropriate extraction
pipeline, applies the right OCR engine (Tesseract or Claude Vision) based on
the automatic origin classifier, and aggregates FileReport objects.

Engine selection logic:
  - Images:       classify_origin() → digital → Tesseract only
                                    → real    → Vision (if --vision) + Tesseract
  - PDF pages:    native text       → run_detectors() directly
                  scanned page      → classify_origin() → same as image
  - Video frames: Tesseract always; if keyword hit and frame is 'real'
                  → Vision (if --vision) for value extraction
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .classifier import needs_vision
from .config import (
    IMAGE_EXTS, PDF_EXTS, VIDEO_EXTS,
    DEFAULT_DIFF_THRESHOLD, DEFAULT_FORCE_INTERVAL,
    DEFAULT_FRAME_INTERVAL, DEFAULT_MAX_FRAMES,
)
from .detectors import run_detectors
from .extractors import extract_video_hits, iter_pdf_pages
from . import ocr, vision


@dataclass
class FileReport:
    """Scan result for a single file or page/frame within a file."""
    file: str
    file_type: str            # "image" | "pdf" | "video"
    findings: list[dict]


def iter_files(target: Path) -> Iterable[Path]:
    """Yield all regular files under *target* (recursively if it is a directory)."""
    if target.is_file():
        yield target
        return
    for p in sorted(target.rglob("*")):
        if p.is_file():
            yield p


def scan(
    target: Path,
    lang: str = "fra+eng",
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frames_out_dir: Path | None = None,
    force_interval: float = DEFAULT_FORCE_INTERVAL,
    use_vision: bool = False,
    vision_all: bool = False,
    vision_backend: str = "anthropic",
    ollama_model: str = "qwen2-vl",
    ollama_host: str = "http://localhost:11434",
) -> list[FileReport]:
    """
    Scan a file or directory and return a list of FileReport objects.

    Args:
        target:         File or directory to scan.
        lang:           Tesseract language string (e.g. "fra+eng").
        frame_interval: Video sampling interval in seconds.
        diff_threshold: Scene-change pixel ratio threshold.
        max_frames:     Max sampled frames per video.
        frames_out_dir: Directory for saving suspect video frame PNGs.
        force_interval: Safety-net forced interval in seconds (0/inf = off).
        use_vision:     Enable Vision LLM for real-world / hard images.
        vision_all:     Force Vision on every file regardless of origin.
        vision_backend: "anthropic" | "openai" | "ollama"
        ollama_model:   Ollama model name (when backend="ollama").
        ollama_host:    Ollama server URL (when backend="ollama").
    """
    reports: list[FileReport] = []
    force_vision = vision_all

    def _vision_analyze(img: Image.Image) -> dict:
        """Wrapper that injects the selected backend."""
        return vision.analyze(
            img,
            backend=vision_backend,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
        )

    for path in iter_files(target):
        ext = path.suffix.lower()
        try:

            # ── Images ────────────────────────────────────────────────────────
            if ext in IMAGE_EXTS:
                img = Image.open(path)
                use_v, v_reason = needs_vision(img, force=force_vision)
                label = "📷 real" if use_v else "🖥 digital"
                print(f"[*] Image : {path}  [{label}]")

                findings: list[dict] = []

                if use_vision and use_v:
                    print(f"  [~] Vision LLM ({v_reason})")
                    vr = _vision_analyze(img)
                    vf = vision.to_findings(vr)
                    for f in vf:
                        f["kind"] = "vision"
                    findings.extend(vf)
                    if vr.get("notes"):
                        print(f"  [~] {vr['notes']}", file=sys.stderr)
                    # Also run Tesseract as a safety net for structured patterns
                    for f in run_detectors(ocr.from_pil(img, lang)):
                        f["kind"] = "structured"
                        findings.append(f)
                else:
                    for f in run_detectors(ocr.from_file(path, lang)):
                        f["kind"] = "structured"
                        findings.append(f)

                if findings:
                    reports.append(FileReport(str(path), "image", findings))

            # ── PDFs ──────────────────────────────────────────────────────────
            elif ext in PDF_EXTS:
                print(f"[*] PDF   : {path}")

                for page_num, native, page_img in iter_pdf_pages(path):
                    findings = []

                    if page_img is None:  # native text page
                        page_label = f"page {page_num} (native text)"
                        for f in run_detectors(native):
                            f["location"] = page_label
                            f["kind"] = "structured"
                            findings.append(f)
                    else:
                        # Scanned page: classify then choose engine
                        use_v, v_reason = needs_vision(page_img, force=force_vision)
                        engine = "Vision" if (use_vision and use_v) else "OCR"
                        page_label = f"page {page_num} (scanned → {engine})"
                        print(f"  [~] {page_label} [{v_reason}]")

                        if use_vision and use_v:
                            vr = _vision_analyze(page_img)
                            for f in vision.to_findings(vr):
                                f["location"] = page_label
                                f["kind"] = "vision"
                                findings.append(f)
                        else:
                            for f in run_detectors(ocr.from_pil(page_img, lang)):
                                f["location"] = page_label
                                f["kind"] = "structured"
                                findings.append(f)

                    if findings:
                        reports.append(FileReport(f"{path} [{page_label}]", "pdf", findings))

            # ── Videos ────────────────────────────────────────────────────────
            elif ext in VIDEO_EXTS:
                fi = force_interval if force_interval > 0 else float("inf")
                print(
                    f"[*] Video : {path} "
                    f"(sampling {frame_interval}s, safety net every {fi}s)"
                )
                vid_frames_dir = (frames_out_dir / path.stem) if frames_out_dir else None

                # In vision_mode, save every scene-change frame regardless of OCR
                # result. Real-world videos (phone footage, camera pans) are
                # invisible to Tesseract; we need Vision to read them.
                hits = extract_video_hits(
                    path, lang, frame_interval, diff_threshold,
                    max_frames, vid_frames_dir, fi,
                    vision_mode=use_vision,
                )

                vid_findings: list[dict] = []
                for h in hits:
                    vision_fds: list[dict] = []
                    vision_notes = ""

                    if use_vision and h.get("frame_path"):
                        fp = Path(h["frame_path"])
                        if fp.exists():
                            frame_img = Image.open(fp)
                            use_v, v_reason = needs_vision(frame_img, force=force_vision)
                            # Call Vision when: frame classified as real, OR OCR
                            # found nothing (needs_vision flag from extractor), OR --vision-all
                            if use_v or h.get("needs_vision") or vision_all:
                                print(f"  [~] Vision LLM on {h['label']} ({v_reason})")
                                vr = _vision_analyze(frame_img)
                                vision_fds = vision.to_findings(vr)
                                vision_notes = vr.get("notes", "")

                    vid_findings.append({
                        "kind": "video_candidate",
                        "location": h["label"],
                        "trigger": h.get("trigger", "scene"),
                        "keywords": h["keywords"],
                        "ocr_lines": h["ocr_lines"],
                        "frame_path": h["frame_path"],
                        "best_effort_findings": h["best_effort_findings"],
                        "needs_vision": h.get("needs_vision", False),
                        "vision_findings": vision_fds,
                        "vision_notes": vision_notes,
                    })

                if vid_findings:
                    reports.append(FileReport(str(path), "video", vid_findings))

        except Exception as exc:
            print(f"  [!] Error on {path}: {exc}", file=sys.stderr)

    return reports
