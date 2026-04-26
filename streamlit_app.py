"""Streamlit Cloud entry point.

Streamlit Cloud auto-detects this file at the repo root. We just put `src/`
on `sys.path` and call the actual app's `main()`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oath_score.app import main  # noqa: E402

main()
