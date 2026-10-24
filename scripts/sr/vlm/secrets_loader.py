"""Shared credential loader, dropped into each project that needs one.

Reads the machine-wide store instead of a per-project .env, so a key is rotated
in one place and no repository ever contains a secret.

Search order, first hit wins:
    1. a variable already exported
    2. SECRETS_DIR, if set
    3. .secrets/ walking up from this file to the filesystem root
    4. ~/.secrets/

Inside the directory, *.env files are parsed as KEY=value lines and any other
plain file becomes the variable named by its uppercased filename.
"""

from __future__ import annotations

import os
from pathlib import Path

SECRETS_DIR_NAME = ".secrets"

# Gemini slots in preference order. The _NEW slot is the account credits were
# bought for, so it wins; the plain slot remains as the fallback.
GEMINI_KEY_VARS = ("GOOGLE_API_KEY_NEW", "GOOGLE_API_KEY", "GOOGLE_API_KEY_2")


def find_secrets_dir(start: Path | None = None) -> Path | None:
    """Locate the .secrets directory without hardcoding a machine path."""
    override = os.environ.get("SECRETS_DIR")
    if override:
        p = Path(override).expanduser()
        return p if p.is_dir() else None

    base = (start or Path(__file__)).resolve()
    for parent in (base, *base.parents):
        candidate = parent / SECRETS_DIR_NAME
        if candidate.is_dir():
            return candidate

    home = Path.home() / SECRETS_DIR_NAME
    return home if home.is_dir() else None


def _apply(name: str, value: str, applied: dict[str, str]) -> None:
    if name and value and name not in os.environ:
        os.environ[name] = value
        applied[name] = value


def load_secrets(start: Path | None = None) -> dict[str, str]:
    """Load the store into os.environ. Never overwrites an exported variable."""
    d = find_secrets_dir(start)
    if d is None:
        return {}
    applied: dict[str, str] = {}
    for f in sorted(d.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if f.suffix == ".env":
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                _apply(k.strip(), v.strip().strip("\"'"), applied)
        else:
            _apply(f.stem.upper().replace("-", "_"), text.strip(), applied)
    return applied


def gemini_api_key(start: Path | None = None) -> str:
    """Return the preferred Gemini key, loading the store if needed."""
    load_secrets(start)
    for name in GEMINI_KEY_VARS:
        value = os.environ.get(name)
        if value:
            return value
    d = find_secrets_dir(start)
    raise RuntimeError(
        f"No Gemini key. Tried {', '.join(GEMINI_KEY_VARS)}. "
        f"Secrets directory: {d or 'none found'}. "
        "Add GOOGLE_API_KEY_NEW=... to <store>/secrets.env, or export it."
    )
