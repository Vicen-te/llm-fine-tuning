"""Shared pytest setup.

The package isn't pip-installed (scripts and tests reach it via `sys.path`),
so we prepend `src/` to sys.path here once and let every test import
`sql_ft.*` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
