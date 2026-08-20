"""
config.py — Global constants and configuration for Sciel.
"""

# Supported file extensions by media type
IMAGE_EXTS: set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
PDF_EXTS:   set[str] = {".pdf"}
VIDEO_EXTS: set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# Pages with fewer native characters than this threshold are treated as scanned
# (image-only) and fall back to OCR.
PDF_NATIVE_TEXT_MIN_CHARS: int = 20

# detect-secrets plugins to skip (not relevant for media scanning).
DETECT_SECRETS_EXCLUDE: set[str] = {"IPPublicDetector"}

# Default video sampling interval (seconds between candidate frames).
DEFAULT_FRAME_INTERVAL: float = 1.0

# Pixel-change ratio below which two frames are considered the same scene.
# Calibrated to catch even a single line of terminal text changing on a fixed
# background (~0.33 % of pixels in practice).
DEFAULT_DIFF_THRESHOLD: float = 0.003

# Safety net: force-process a frame every N seconds regardless of scene diff,
# to catch subtle text changes that stay below the diff threshold.
DEFAULT_FORCE_INTERVAL: float = 5.0

# Hard cap on sampled frames per video to prevent runaway processing.
DEFAULT_MAX_FRAMES: int = 600

# Default output folder for saved video frames that triggered a keyword hit.
DEFAULT_FRAMES_DIR: str = "twilight_frames"

# Anthropic model used for Vision LLM fallback.
VISION_MODEL: str = "claude-sonnet-4-6"
VISION_MAX_TOKENS: int = 512
VISION_MAX_IMAGE_SIDE: int = 1280
