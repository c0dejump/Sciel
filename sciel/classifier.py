"""
classifier.py — Automatic origin classifier: digital vs real.

Determines whether Tesseract OCR is sufficient (screenshots, synthetic images,
clean PDFs) or whether the Claude Vision LLM fallback is needed (camera photos,
physical post-its, compressed video frames, scanned documents).

Four complementary signals — all computed locally, no network required:

  A. Histogram occupancy
     Fraction of quantised colour bins actually used.
     Digital → tiny palette (UI/CSS colours)   → low occupancy
     Real    → continuous sensor distribution   → high occupancy

  B. Dominant-colour pixel-to-pixel std
     Standard deviation of pixels sharing the most common quantised colour,
     measured per-channel across pixels (NOT across channels within one pixel).
     Digital → perfectly uniform background    → std ≈ 0
     Real    → sensor noise / codec artefacts  → std > 0

  C. Sobel gradient kurtosis
     Excess kurtosis of the gradient-magnitude distribution.
     Digital → bimodal (0 or very high) → high kurtosis
     Real    → continuous gradients     → moderate kurtosis

  D. Exact-mode pixel ratio
     Fraction of pixels whose RGB value is byte-identical to the dominant colour.
     Digital → large flat background region → ratio high
     Real    → lossy compression breaks uniformity → ratio low
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def classify_origin(img: Image.Image) -> dict:
    """
    Classify a PIL image as 'digital' (Tesseract sufficient) or 'real'
    (Vision LLM recommended).

    Returns:
        {
          "origin":        "digital" | "real",
          "confidence":    float in [0, 1],
          "digital_score": int,
          "real_score":    int,
          "reasons":       list[str],   # human-readable signal hits
        }
    """
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    pixels = arr.reshape(-1, 3).astype(np.int32)

    # A: histogram occupancy (bins of 8 per channel → 32 bins/channel, 32^3 total)
    binned = pixels // 8  # quantise all three channels, not just blue
    keys = binned[:, 0] * 1024 + binned[:, 1] * 32 + binned[:, 2]
    unique_keys, counts = np.unique(keys, return_counts=True)
    occupancy = len(unique_keys) / (32 ** 3)  # 32768 possible bins

    # B: per-pixel std on dominant colour (axis=0 → across pixels per channel)
    dom_key = unique_keys[np.argmax(counts)]
    dom_pix = arr.reshape(-1, 3)[(keys == dom_key)].astype(np.float32)
    mode_std = float(dom_pix.std(axis=0).mean()) if len(dom_pix) > 100 else 99.0

    # C: Sobel gradient kurtosis
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2).flatten()
    m, s = mag.mean(), mag.std()
    kurtosis = float(np.mean(((mag - m) / s) ** 4)) if s > 0 else 0.0

    # D: exact-mode pixel ratio
    mode_color = dom_pix.mean(axis=0).astype(np.uint8)
    exact_ratio = float((arr.reshape(-1, 3) == mode_color).all(axis=1).mean())

    # --- Scoring ----------------------------------------------------------------
    d, r = 0, 0
    reasons: list[str] = []

    # A: occupancy — most globally reliable signal, weighted strongly
    if occupancy < 0.010:
        d += 3; reasons.append(f"tiny palette ({occupancy:.5f})")
    elif occupancy < 0.030:
        d += 2
    elif occupancy < 0.060:
        d += 1
    elif occupancy < 0.200:
        r += 1
    elif occupancy < 0.500:
        r += 2
    else:
        r += 4; reasons.append(f"very rich palette ({occupancy:.3f})")  # dead giveaway

    # B: mode_std — only weighted when occupancy is low (avoids false digital on photos
    #    whose dominant colour happens to have low per-pixel variance)
    if mode_std < 0.5 and occupancy < 0.15:
        d += 3; reasons.append(f"perfect monochrome background (std={mode_std:.3f})")
    elif mode_std < 2.0 and occupancy < 0.15:
        d += 2
    elif mode_std > 8.0:
        r += 2; reasons.append(f"sensor noise / codec artefacts (std={mode_std:.1f})")

    # C: kurtosis
    if kurtosis > 60:
        d += 2; reasons.append(f"hard edges (kurt={kurtosis:.0f})")
    elif kurtosis > 25:
        d += 1
    else:
        r += 1

    # D: exact-mode ratio
    if exact_ratio > 0.25 and occupancy < 0.15:
        d += 2; reasons.append(f"uniform background ({exact_ratio:.0%} identical px)")
    elif exact_ratio > 0.10 and occupancy < 0.15:
        d += 1
    elif exact_ratio < 0.05:
        r += 1

    total = d + r
    origin = "digital" if d >= r else "real"
    confidence = max(d, r) / total if total > 0 else 0.5

    return {
        "origin": origin,
        "confidence": round(confidence, 2),
        "digital_score": d,
        "real_score": r,
        "reasons": reasons,
    }


def needs_vision(img: Image.Image, force: bool = False) -> tuple[bool, str]:
    """
    Decide whether Claude Vision should be used for this image.

    Args:
        img:   PIL image to evaluate.
        force: If True, always return (True, reason) regardless of signals
               (used for --vision-all mode).

    Returns:
        (use_vision: bool, human_readable_reason: str)
    """
    if force:
        return True, "forced by --vision-all"
    result = classify_origin(img)
    if result["origin"] == "real":
        detail = ", ".join(result["reasons"]) or "multiple signals"
        reason = f"real image detected ({result['confidence']:.0%}): {detail}"
        return True, reason
    detail = f"digital content ({result['confidence']:.0%}): Tesseract sufficient"
    return False, detail
