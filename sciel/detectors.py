"""
detectors.py — Secret detection layer.

Two complementary engines:
  1. detect-secrets (official plugins): AWS keys, Azure, Stripe, Slack, GitHub
     tokens, JWTs, private keys, high-entropy strings, etc.
  2. FR/EN keyword heuristic: catches natural-language patterns that
     detect-secrets misses ("mot de passe : xxx", "clé wifi : xxx", etc.)
     — essential for OCR'd text from real images and videos.
"""

from __future__ import annotations

import re

from detect_secrets.core.plugins.util import get_mapping_from_secret_type_to_class
from detect_secrets.settings import default_settings

from .config import DETECT_SECRETS_EXCLUDE


# --------------------------------------------------------------------------- #
# FR/EN keyword regex — structured value extraction
# Matches "keyword [filler words] : value" patterns on a single line.
# Used for images and PDFs where the OCR output is reasonably clean.
# --------------------------------------------------------------------------- #

KEYWORD_RE = re.compile(
    r"""(?P<kw>
        mot\s*de\s*passe | mdp | passe | passwords? | pass | pwd | passwd | passcode |
        cl[ée]\s*(?:wifi|wi-?fi|api|secr[èe]te|priv[ée]e)? |
        secret(?:\s*key)? | api[\s_-]?key | access[\s_-]?key | token |
        identifiant | login | username | user
    )
    (?:\s+[\wÀ-ÿ'-]+){0,2}?   # up to 2 filler words ("wifi home", "d'accès")
    \s*[:=\-]\s*
    (?P<val>\S{4,60})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Stopwords: values that look like UI copy, not actual secrets.
KEYWORD_STOPWORDS: set[str] = {
    "doit", "est", "sera", "votre", "your", "the", "requis", "obligatoire",
    "oublié", "oublie", "perdu", "forgot", "below", "ci-dessous", "saisir",
    "entrez", "enter", "required", "contenir", "caractères", "caracteres",
    "incorrect", "invalide", "invalid", "masqué", "masque", "hidden",
    "different", "différent", "confirm", "confirmez", "field", "champ",
    "ici", "here", "vide", "empty", "null", "none", "n/a",
}


def _is_placeholder(val: str) -> bool:
    """Return True if the extracted value looks like a UI placeholder, not a secret."""
    stripped = val.strip("*.•_-●○\"'")
    if not stripped:
        return True
    if val.lower().strip(":\"'") in KEYWORD_STOPWORDS:
        return True
    # Majority masking characters (e.g. "••••••••")
    mask_chars = sum(1 for c in val if c in "*•●○_")
    if mask_chars / max(len(val), 1) > 0.5:
        return True
    return False


def keyword_findings(text: str) -> list[dict]:
    """
    Line-by-line FR/EN keyword heuristic for natural OCR'd text.
    Requires keyword + separator + value on the same line.
    """
    out = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for m in KEYWORD_RE.finditer(line):
            val = m.group("val")
            if _is_placeholder(val):
                continue
            out.append({
                "detector": "KeywordHeuristic",
                "secret_type": f"keyword: {m.group('kw').strip()}",
                "line_number": line_no,
                "value": val,
                "context": line.strip()[:200],
            })
    return out


# --------------------------------------------------------------------------- #
# Presence-only keyword regex — permissive, for video frame detection.
# No value extraction: noisy OCR from compressed video frames makes extracted
# values unreliable. We just check whether a sensitive keyword is present,
# then save the frame for human visual review.
# --------------------------------------------------------------------------- #

KEYWORD_PRESENCE_RE = re.compile(
    r"""(?:
        # Compound forms: wifi_password, db_pass, env_secret, auth_token, etc.
        # Leading \b intentionally omitted so we also match after _ or -
        [\w-]*passwords?[\w-]* |
        [\w-]*passwd[\w-]*     |
        [\w-]*secret[\w-]*     |
        [\w-]*token[\w-]*      |
        [\w-]*api[\s_-]?key[\w-]* |
        # Standalone forms (strict word boundary)
        \bmot\s*de\s*passe\b | \bmdp\b | \bpwd\b | \bpasscode\b |
        \bcl[ée]\s*(?:wifi|wi-?fi|api)\b |
        \baccess[\s_-]?key\b | \bidentifiant\b | \blogin\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def keyword_hits(text: str) -> list[tuple[str, str]]:
    """
    Return [(matched_keyword, raw_line)] — permissive presence check, no filtering.
    Intended for video frame screening: we just need to know whether a frame
    is worth saving for manual review.
    """
    hits = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for m in KEYWORD_PRESENCE_RE.finditer(line):
            hits.append((m.group(0), line[:200]))
    return hits


# --------------------------------------------------------------------------- #
# detect-secrets plugin layer
# --------------------------------------------------------------------------- #

_DS_PLUGINS: list | None = None


def _load_detect_secrets_plugins() -> list:
    with default_settings():
        mapping = get_mapping_from_secret_type_to_class()
    plugins = []
    for _secret_type, cls in mapping.items():
        if cls.__name__ in DETECT_SECRETS_EXCLUDE:
            continue
        if cls.__name__ == "Base64HighEntropyString":
            plugins.append(cls(limit=4.5))
        elif cls.__name__ == "HexHighEntropyString":
            plugins.append(cls(limit=3.0))
        else:
            plugins.append(cls())
    return plugins


def detect_secrets_findings(text: str) -> list[dict]:
    """Run all detect-secrets plugins line by line and return findings."""
    global _DS_PLUGINS
    if _DS_PLUGINS is None:
        _DS_PLUGINS = _load_detect_secrets_plugins()

    out = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        for plugin in _DS_PLUGINS:
            try:
                matches = list(plugin.analyze_string(line))
            except Exception:
                continue
            for val in matches:
                out.append({
                    "detector": plugin.__class__.__name__,
                    "secret_type": plugin.secret_type,
                    "line_number": line_no,
                    "value": val,
                    "context": line.strip()[:200],
                })
    return out


def run_detectors(text: str) -> list[dict]:
    """
    Run both detection engines and return deduplicated findings.
    Deduplication key: (line_number, value).
    """
    if not text or not text.strip():
        return []
    findings = detect_secrets_findings(text) + keyword_findings(text)
    seen: set[tuple] = set()
    deduped = []
    for f in findings:
        key = (f["line_number"], f["value"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped
