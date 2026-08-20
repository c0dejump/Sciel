"""
Sciel — clair-obscur secret scanner.

Scans images, videos and PDFs for cleartext passwords and secrets.

Named after the character Sciel from *Clair Obscur: Expedition 33* (Sandfall
Interactive), whose Sun + Moon → Twilight mechanic mirrors the tool's concept:
pulling secrets from the obscure (hidden in a photo, video frame or scanned PDF)
into the clear.

Public API:
    from sciel.scanner import scan, FileReport
    from sciel.report  import print_report, write_json
"""

__version__ = "1.0.0"
__author__  = "Plop"
