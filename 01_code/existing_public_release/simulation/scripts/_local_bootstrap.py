"""Keep this repository's simulation packages ahead of other local copies."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_project_imports(anchor_file: str | os.PathLike[str]) -> Path:
    root = Path(anchor_file).resolve().parents[1]
    for path in (root, root / "relax_env"):
        value = str(path)
        while value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return root
