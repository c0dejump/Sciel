# Sciel

<p align="center">
  <img src="./static/Logo_sciel.png" alt="Logo" width="420">
</p>

> A scanner for passwords and plaintext secrets left in photos, videos, and PDFs.

**Sciel** pulls cleartext credentials out of the *obscure* — a password on a
sticky note in a photo, a key flashing by in a screen recording, a secret buried
in a scanned PDF — and into the *clear*. It combines local OCR (Tesseract), an
automatic image-origin classifier, structured secret detectors, and an optional
Vision-LLM fallback for the hard, real-world cases.

---

## Features

- **Images, videos and PDFs** — one command, recursive over a directory.
- **Automatic origin classifier** — decides *locally* (no network) whether an
  image is *digital* (screenshot/UI → Tesseract is enough) or *real* (camera
  photo, post-it, compressed frame → Vision LLM recommended).
- **Two detection engines**
  - [`detect-secrets`](https://github.com/Yelp/detect-secrets) plugins: AWS,
    Azure, Stripe, Slack, GitHub tokens, JWTs, private keys, high-entropy
    strings, …
  - A FR/EN keyword heuristic for natural-language patterns
    (`mot de passe : …`, `clé wifi : …`, `api_key = …`) that structured
    detectors miss on OCR'd text.
- **Smart video sampling** — scene-change de-duplication plus a forced-interval
  safety net, so a single terminal line changing on a static background is not
  missed. Suspect frames are saved to `twilight_frames/` for visual review.
- **Optional Vision LLM fallback** — Claude, GPT-4o, or a local Ollama model for
  handwriting, real-world scenes, and heavily compressed frames.

---

## Installation

Sciel needs the **Tesseract OCR** engine installed on the system (the Python
package `pytesseract` is only a wrapper), with the French and English language
data.

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng

# macOS
brew install tesseract tesseract-lang
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# Scan a directory recursively
python sciel.py ./evidence/

# Scan a single image
python sciel.py screenshot.png --lang eng

# Scan a video, sampling every 0.5s
python sciel.py recording.mp4 --frame-interval 0.5

# Enable the Vision LLM fallback and write a JSON report
python sciel.py ./evidence/ --vision --out report.json
```

Sciel is also runnable as a module: `python -m sciel <target>`.

### Vision backends

The Vision fallback is off by default. Enable it with `--vision` (called
automatically only when the classifier flags an image as *real*) or
`--vision-all` (forced on every file).

| Backend            | Flag                          | Requirement                                         |
| ------------------ | ----------------------------- | --------------------------------------------------- |
| Claude (default)   | `--vision-backend anthropic`  | `ANTHROPIC_API_KEY`                                 |
| GPT-4o             | `--vision-backend openai`     | `OPENAI_API_KEY`                                    |
| Ollama (local)     | `--vision-backend ollama`     | [Ollama](https://ollama.ai) + `ollama pull qwen2-vl` |

```bash
# Local, no API key, no data leaving the machine
python sciel.py ./evidence/ --vision --vision-backend ollama --ollama-model qwen2-vl
```

### Options

| Option              | Default            | Description                                                        |
| ------------------- | ------------------ | ----------------------------------------------------------------- |
| `--lang`            | `fra+eng`          | Tesseract language(s).                                             |
| `--frame-interval`  | `1.0`              | Video sampling interval (seconds).                                |
| `--diff-threshold`  | `0.003`            | Changed-pixel ratio to declare a new scene.                       |
| `--max-frames`      | `600`              | Max frames sampled per video.                                     |
| `--force-interval`  | `5.0`              | Force-process a frame every N seconds (`0` to disable).           |
| `--frames-dir`      | `twilight_frames`  | Where suspect video frames are saved.                             |
| `--no-save-frames`  | off                | Report timecodes only; don't save frames.                         |
| `--vision`          | off                | Enable Vision LLM for images the classifier flags as *real*.      |
| `--vision-all`      | off                | Force Vision on every file (slower, maximal coverage).            |
| `--vision-backend`  | `anthropic`        | `anthropic` \| `openai` \| `ollama`.                              |
| `--ollama-model`    | `qwen2-vl`         | Ollama model (with `--vision-backend ollama`).                    |
| `--ollama-host`     | `http://localhost:11434` | Ollama server URL.                                          |
| `--out`             | —                  | Write results to a JSON file.                                     |

---

## How it works

```
                 ┌───────────┐
   file  ──────▶ │ dispatch  │ by extension
                 └─────┬─────┘
        ┌──────────────┼─────────────────┐
      image           pdf               video
        │              │                   │
   classify        native text?      sample frames
  digital/real     ├─ yes → detectors  (scene diff +
        │          └─ no  → classify    forced interval)
        │                   (as image)      │
        ▼                                    ▼
   Tesseract / Vision  ───▶  detect-secrets + FR/EN keyword heuristic
                                     │
                                     ▼
                        console report  (+ optional JSON)
```

Results are printed as a human-readable report and, with `--out`, written as
structured JSON for downstream tooling.

---

## License

[MIT](./static/LICENSE) © c0dejump
