"""Thin launcher — the real entrypoint is app/ui/main.py.

Kept so `streamlit run app/main.py` and `streamlit run app/ui/main.py` both
work; `make run` uses the latter directly.
"""

from __future__ import annotations

from app.ui.main import main

if __name__ == "__main__":
    main()
