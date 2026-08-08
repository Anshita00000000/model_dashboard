"""Thin launcher — the real entrypoint is app/ui/main.py.

Kept so `streamlit run app/main.py` and `streamlit run app/ui/main.py` both
work; `make run` uses the latter directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable regardless of the caller's cwd/PYTHONPATH —
# `streamlit run` does not reliably put it there on its own (observed to work
# in some environments and fail in others, e.g. a fresh Colab shell).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.ui.main import main  # noqa: E402 (must follow the sys.path fix above)

if __name__ == "__main__":
    main()
