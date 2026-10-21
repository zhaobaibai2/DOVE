"""Local import bootstrap for the isolated test0508 workspace."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def workspace_root(anchor_file: str | os.PathLike[str]) -> Path:
    """Return the test0508 root for a file under test0508/scripts."""
    return Path(anchor_file).resolve().parents[1]


def ensure_test0508_imports(anchor_file: str | os.PathLike[str]) -> Path:
    """Put the test0508 root first so imports do not fall back to old copies."""
    root = workspace_root(anchor_file)
    preferred = [root, root / "relax_env"]
    for path in reversed(preferred):
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    return root
