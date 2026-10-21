"""Parse active YAML configuration files without importing experiment code."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in ROOT.rglob(pattern)
        if ".codex_backup" not in path.as_posix()
        and "/old/" not in path.as_posix()
    )
    for path in files:
        with path.open("r", encoding="utf-8") as stream:
            yaml.safe_load(stream)
    print(f"YAML_PARSE=PASS files={len(files)}")


if __name__ == "__main__":
    main()
