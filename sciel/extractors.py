"""
extractors.py — Media content extraction pipelines.

PDF:   Native text extraction (PyMuPDF) with per-page OCR fallback for
       scanned/image-only pages.

Video: Hybrid two-pass sampling strategy:
       1. Scene-change deduplication (diff_threshold): skips near-identical
          frames (static slides, lock screens) to avoid redundant OCR.
       2. Force interval (safety net): processes a frame every N seconds
          regardless of scene diff, catching subtle single-line changes
          (terminal output, form fields) that fall below the pixel threshold.

       When a keyword hit is found in a frame, the frame is saved as a PNG
       (twilight_frames/) so the analyst can visually verify the exact value
       — much more reliable than trusting noisy OCR output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .config import PDF_NATIVE_TEXT_MIN_CHARS
from .detectors import keyword_hits, run_detectors
from . import ocr


def iter_pdf_pages(
    path: Path,
    dpi: int = 200,
    native_min_chars: int = PDF_NATIVE_TEXT_MIN_CHARS,
) -> Iterator[tuple[int, str, Image.Image | None]]:
    """
    Yield one entry per PDF page, opening the document only once.

    A page is treated as *native text* when it carries at least
    *native_min_chars* characters; otherwise it is considered scanned
    (image-only) and rasterised at *dpi* for OCR / Vision analysis.

    Args:
        path:             Path to the PDF.
        dpi:              Rendering resolution for scanned pages.
        native_min_chars: Character threshold separating native from scanned.

    Yields:
        (page_number, native_text, page_image) tuples where *page_number* is
        1-based and *page_image* is a PIL Image for scanned pages, or None
        when the page has usable native text.
    """
    doc = fitz.open(path)
    try:
        for idx in range(doc.page_count):
            page = doc[idx]
            native = page.get_text()
            if len(native.strip()) >= native_min_chars:
                yield idx + 1, native, None
            else:
                pix = page.get_pixmap(dpi=dpi)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                yield idx + 1, native, img
    finally:
        doc.close()


def extract_video_hits(
    path: Path,
    lang: str,
    sample_interval: float = 1.0,
    diff_threshold: float = 0.003,
    max_frames: int = 600,
    frames_out_dir: Path | None = None,
    force_interval: float = 5.0,
    vision_mode: bool = False,
) -> list[dict]:
    """
    Sample a video for frames that contain credential-related keywords.

    Strategy:
      - Sample one frame every *sample_interval* seconds.
      - Skip frames whose pixel diff vs the last kept frame is below
        *diff_threshold* (scene-change deduplication).
      - Always process a frame every *force_interval* seconds as a safety net
        for subtle text changes (a single terminal line appearing on an
        otherwise static background — typically < 0.4 % pixel change).

    Two modes:

    • Standard mode (vision_mode=False):
        Run Tesseract on each candidate frame. Save the frame and report the
        timecode only when a keyword is found in the OCR output.
        Works well for digital content (screenshots, screen recordings, slides).

    • Vision mode (vision_mode=True):
        Save every unique scene-change frame regardless of OCR result.
        Tesseract is still run as a best-effort first pass, but the frame is
        saved even when OCR returns nothing. This is necessary for real-world
        videos (camera footage) where Tesseract is completely blind —
        the saved frames are handed to Claude Vision by the caller.

    Args:
        path:           Path to the video file.
        lang:           Tesseract language string.
        sample_interval: Seconds between sampled frames.
        diff_threshold: Changed-pixel ratio above which a new scene is declared.
        max_frames:     Hard cap on total sampled frames.
        frames_out_dir: Directory to save suspect frame PNGs (None = don't save).
        force_interval: Force-process a frame every N seconds (0 / inf = off).
        vision_mode:    If True, save all scene-change frames for Vision LLM
                        analysis, bypassing the keyword-hit requirement.

    Returns:
        List of hit dicts:
        {
          "label":                str,        # "t=12.3s"
          "timestamp":            float,
          "trigger":              str,        # "scene" | "forced interval"
          "keywords":             list[str],  # empty list in vision_mode when OCR fails
          "ocr_lines":            list[str],  # raw OCR output (may be empty/noisy)
          "frame_path":           str | None, # path to saved PNG
          "best_effort_findings": list[dict], # Tesseract + detector findings
          "needs_vision":         bool,       # True when OCR found nothing useful
        }
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  [!] Cannot open video {path.name}", file=sys.stderr)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(fps * sample_interval), 1)
    force_step = max(int(fps * force_interval), 1) if force_interval < 1e9 else None

    hits: list[dict] = []
    last_small: np.ndarray | None = None
    last_forced_idx = -force_step if force_step is not None else 0
    frame_idx = 0
    sampled = 0

    while True:
        ok = cap.grab()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            ok, frame = cap.retrieve()
            if ok:
                sampled += 1
                if sampled > max_frames:
                    print(
                        f"  [!] Frame cap ({max_frames}) reached for {path.name}, stopping early.",
                        file=sys.stderr,
                    )
                    break

                small = cv2.cvtColor(
                    cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY
                )

                # Scene-change detection
                is_new_scene = True
                if last_small is not None:
                    diff = cv2.absdiff(small, last_small)
                    changed_ratio = float(np.mean(diff > 25))
                    is_new_scene = changed_ratio > diff_threshold

                # Safety-net forced interval
                forced = (
                    force_step is not None
                    and (frame_idx - last_forced_idx) >= force_step
                )

                if is_new_scene or forced:
                    last_small = small
                    if forced:
                        last_forced_idx = frame_idx

                    timestamp = frame_idx / fps
                    trigger = "scene" if is_new_scene else "forced interval"
                    text = ocr.from_cv2_frame(frame, lang)
                    kw = keyword_hits(text)
                    best_effort = run_detectors(text)
                    ocr_found_nothing = not kw and not best_effort

                    # Decide whether to keep this frame
                    # In vision_mode: keep every scene-change frame (OCR is likely blind)
                    # Otherwise: keep only when OCR found a keyword
                    should_keep = vision_mode or bool(kw)

                    if should_keep:
                        frame_path: Path | None = None
                        if frames_out_dir is not None:
                            frames_out_dir.mkdir(parents=True, exist_ok=True)
                            fname = f"{path.stem}_t{timestamp:.1f}s.png"
                            frame_path = frames_out_dir / fname
                            cv2.imwrite(str(frame_path), frame)

                        hits.append({
                            "label": f"t={timestamp:.1f}s",
                            "timestamp": timestamp,
                            "trigger": trigger,
                            "keywords": sorted({k for k, _ in kw}),
                            "ocr_lines": [line for _, line in kw],
                            "frame_path": str(frame_path) if frame_path else None,
                            "best_effort_findings": best_effort,
                            "needs_vision": ocr_found_nothing,
                        })

        frame_idx += 1

    cap.release()
    return hits

