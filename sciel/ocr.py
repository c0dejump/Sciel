"""
ocr.py — Tesseract OCR wrappers for images and video frames.

Tesseract is the primary OCR engine. It handles screenshots, clean PDFs,
terminals and other digital content efficiently and locally.
For real-world / compressed / handwritten content, see vision.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


def from_file(path: Path, lang: str) -> str:
    """
    Run Tesseract on an image file and return extracted text.

    Args:
        path: Path to the image.
        lang: Tesseract language string (e.g. "fra+eng").

    Returns:
        Extracted text, or empty string on failure.
    """
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang=lang)
    except Exception as exc:
        print(f"  [!] OCR failed on {path.name}: {exc}", file=sys.stderr)
        return ""


def from_pil(img: Image.Image, lang: str) -> str:
    """Run Tesseract on a PIL Image and return extracted text."""
    return pytesseract.image_to_string(img, lang=lang)


def from_cv2_frame(frame: np.ndarray, lang: str) -> str:
    """
    Run Tesseract on an OpenCV BGR frame (numpy array) and return extracted text.
    Converts BGR → RGB before handing off to Tesseract.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pytesseract.image_to_string(Image.fromarray(rgb), lang=lang)
