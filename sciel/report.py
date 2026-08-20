"""
report.py — Output formatting for Sciel scan results.

Two formats:
  - Human-readable console report (print_report)
  - Machine-readable JSON file   (write_json)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .scanner import FileReport


def print_report(reports: list[FileReport]) -> None:
    """Print a human-readable summary to stdout."""
    if not reports:
        print("\n✅  No secrets detected.")
        return

    total = sum(len(r.findings) for r in reports)
    print(f"\n⚠️   {total} potential finding(s) across {len(reports)} location(s):\n")

    for r in reports:
        print(f"── {r.file}  [{r.file_type}]")
        for f in r.findings:
            kind = f.get("kind")

            if kind == "video_candidate":
                print(f"   🔎 {f['location']} — keyword(s): {', '.join(f['keywords']) or '(none — OCR blind)'}")
                for line in f.get("ocr_lines", []):
                    print(f"      OCR (raw, unreliable): {line!r}")
                if f.get("frame_path"):
                    print(f"      → saved frame : {f['frame_path']}")
                if f.get("needs_vision") and not f.get("vision_findings"):
                    print(f"      ⚠️  Tesseract returned nothing — run with --vision for LLM analysis")
                for bf in f.get("best_effort_findings", []):
                    print(f"      Tesseract     : {bf['secret_type']} = {bf['value']!r}")
                for vf in f.get("vision_findings", []):
                    print(f"      👁  Vision    : {vf['secret_type']} = {vf['value']!r}  ({vf['context']})")
                if f.get("vision_notes"):
                    print(f"      👁  Notes     : {f['vision_notes']}")

            elif kind == "vision":
                print(f"   👁  [Vision] {f['secret_type']}")
                print(f"      value   : {f['value']!r}")
                print(f"      context : {f['context']!r}")

            else:  # "structured" findings from Tesseract + detect-secrets / keyword heuristic
                loc = f.get("location", f"line {f.get('line_number', '?')}")
                print(f"   • [{f['detector']}] {f['secret_type']}")
                print(f"     {loc} — line {f.get('line_number', '?')}")
                print(f"     value   : {f['value']!r}")
                print(f"     context : {f['context']!r}")
        print()


def write_json(reports: list[FileReport], out_path: Path) -> None:
    """Serialise all reports to a JSON file."""
    data = [asdict(r) for r in reports]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f"[*] JSON report written: {out_path}")
