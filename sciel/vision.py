"""
vision.py — Vision LLM backends for hard OCR cases.

Handles images that Tesseract cannot reliably read:
  - Handwritten text (post-its, whiteboards)
  - Camera-captured scenes with natural lighting / perspective
  - Heavily compressed video frames
  - Small text in complex real-world backgrounds

Supported backends:
  - anthropic  : Claude claude-sonnet-4-6 via Anthropic API (ANTHROPIC_API_KEY required)
  - openai     : GPT-4o via OpenAI API    (OPENAI_API_KEY required)
  - ollama     : Any local vision model via Ollama (no key, runs on your machine)
                 Recommended models: qwen2-vl, llava, moondream, llama3.2-vision
                 Install: https://ollama.ai — then: ollama pull qwen2-vl
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys
import urllib.request
from typing import Any

from PIL import Image

from .config import VISION_MAX_IMAGE_SIDE, VISION_MAX_TOKENS, VISION_MODEL


# --------------------------------------------------------------------------- #
# Shared prompt
# --------------------------------------------------------------------------- #

VISION_PROMPT = """\
You are an offensive security tool. Analyse this image and respond ONLY with \
valid JSON, no markdown fences.

Look for any visible text that looks like credentials: passwords, API keys, \
tokens, secrets, connection strings, SSH keys, WiFi passphrases, etc.
Include post-its, handwritten notes, form fields, config files, terminals, \
and any other source.

STRICT response format:
{
  "found": true|false,
  "items": [
    {
      "type": "password|api_key|token|wifi_key|...",
      "context": "where/how it appears in the image",
      "value": "exact value or best-guess transcription"
    }
  ],
  "notes": "useful observations (handwriting quality, image blur, partial visibility, etc.)"
}

If nothing sensitive: {"found": false, "items": [], "notes": "..."}"""


# --------------------------------------------------------------------------- #
# Image encoding
# --------------------------------------------------------------------------- #

def _encode_image(img: Image.Image, max_side: int = VISION_MAX_IMAGE_SIDE) -> tuple[str, str]:
    """
    Resize if needed and base64-encode as JPEG.
    Returns (base64_data, media_type).
    """
    if max(img.width, img.height) > max_side:
        ratio = max_side / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)),
            Image.LANCZOS,
        )
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


def _parse_json_response(text: str) -> dict:
    """Strip accidental markdown fences and parse JSON."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    # Some models wrap in extra text — try to extract the JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


# --------------------------------------------------------------------------- #
# Backend: Anthropic (Claude Vision)
# --------------------------------------------------------------------------- #

def _analyze_anthropic(img: Image.Image) -> dict:
    """Send image to Claude claude-sonnet-4-6 via Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    b64, media_type = _encode_image(img)

    payload = json.dumps({
        "model": VISION_MODEL,
        "max_tokens": VISION_MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return _parse_json_response(data["content"][0]["text"])


# --------------------------------------------------------------------------- #
# Backend: OpenAI (GPT-4o Vision)
# --------------------------------------------------------------------------- #

def _analyze_openai(img: Image.Image) -> dict:
    """Send image to GPT-4o via OpenAI API."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    b64, media_type = _encode_image(img)

    payload = json.dumps({
        "model": "gpt-4o",
        "max_tokens": VISION_MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": VISION_PROMPT},
            ],
        }],
    }).encode()

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return _parse_json_response(data["choices"][0]["message"]["content"])


# --------------------------------------------------------------------------- #
# Backend: Ollama (local)
# --------------------------------------------------------------------------- #

def _analyze_ollama(
    img: Image.Image,
    model: str = "qwen2-vl",
    host: str = "http://localhost:11434",
) -> dict:
    """
    Send image to a local Ollama vision model.

    Recommended models (run 'ollama pull <model>' first):
      qwen2-vl        — best for OCR / text in scenes (~8GB)
      llava           — good generalist vision model (~4GB)
      llama3.2-vision — Meta's vision model (~8GB)
      moondream       — very lightweight (~2GB), fast

    Args:
        img:   PIL image.
        model: Ollama model name (default: qwen2-vl).
        host:  Ollama server URL (default: http://localhost:11434).
    """
    b64, _ = _encode_image(img)

    payload = json.dumps({
        "model": model,
        "prompt": VISION_PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0},
    }).encode()

    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # local can be slower
        data = json.loads(resp.read())
    return _parse_json_response(data["response"])


# --------------------------------------------------------------------------- #
# Public interface
# --------------------------------------------------------------------------- #

def analyze(
    img: Image.Image,
    backend: str = "anthropic",
    ollama_model: str = "qwen2-vl",
    ollama_host: str = "http://localhost:11434",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Analyse an image with the selected Vision LLM backend.

    Args:
        img:          PIL image to analyse.
        backend:      "anthropic" | "openai" | "ollama"
        ollama_model: Ollama model name (only used when backend="ollama").
        ollama_host:  Ollama server URL (only used when backend="ollama").
        verbose:      Print raw response to stderr for debugging.

    Returns:
        {
          "found": bool,
          "items": [{"type": str, "context": str, "value": str}],
          "notes": str,
        }
        On error: {"found": False, "items": [], "notes": "<error message>"}
    """
    try:
        if backend == "anthropic":
            result = _analyze_anthropic(img)
        elif backend == "openai":
            result = _analyze_openai(img)
        elif backend == "ollama":
            result = _analyze_ollama(img, model=ollama_model, host=ollama_host)
        else:
            raise ValueError(f"Unknown vision backend: {backend!r}. "
                             f"Choose: anthropic, openai, ollama")
        if verbose:
            print(f"  [vision/{backend}] {result}", file=sys.stderr)
        return result
    except Exception as exc:
        if verbose:
            print(f"  [vision/{backend}] error: {exc}", file=sys.stderr)
        return {"found": False, "items": [], "notes": f"{backend} error: {exc}"}


def to_findings(vision_result: dict[str, Any]) -> list[dict]:
    """Convert a Vision API result dict into the standard findings list format."""
    if not vision_result.get("found"):
        return []
    out = []
    for item in vision_result.get("items", []):
        out.append({
            "detector": "VisionLLM",
            "secret_type": item.get("type", "unknown"),
            "line_number": 0,
            "value": item.get("value", ""),
            "context": item.get("context", "")[:200],
        })
    return out


def check_backend(backend: str, ollama_host: str = "http://localhost:11434") -> tuple[bool, str]:
    """
    Quick connectivity check for the chosen backend.

    Returns (available: bool, message: str).
    """
    if backend == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return False, "ANTHROPIC_API_KEY not set"
        return True, "ANTHROPIC_API_KEY found"

    elif backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return False, "OPENAI_API_KEY not set"
        return True, "OPENAI_API_KEY found"

    elif backend == "ollama":
        try:
            req = urllib.request.Request(f"{ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                return True, f"Ollama running — available models: {', '.join(models)}"
            return False, "Ollama running but no models installed (run: ollama pull qwen2-vl)"
        except Exception as exc:
            return False, f"Ollama not reachable at {ollama_host}: {exc}"

    return False, f"Unknown backend: {backend!r}"
