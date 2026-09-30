from __future__ import annotations

import csv
import os
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
CONDA_SH = os.environ.get("CONDA_SH")
DACER_ENV = "dacer_pvp"
HACO_ENV = "haco"
DEFAULT_LOG_DIR = Path(os.environ.get("DOVE_LOG_DIR", ROOT / "outputs" / "checkpoints"))
DEFAULT_POLICY = os.environ.get("DOVE_POLICY", "final_policy_20324.pkl")
DEFAULT_BC_POLICY = os.environ.get("DOVE_BC_POLICY", "stage1b_offline_bc_policy.pkl")
DEFAULT_MPLCONFIGDIR = ROOT / "outputs" / "matplotlib"


def parse_csv(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_int_csv(raw: str) -> List[int]:
    return [int(item) for item in parse_csv(raw)]


def qjoin(parts: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def bash_conda_command(
    env_name: str,
    argv: Sequence[Any],
    *,
    cwd: Path = ROOT,
    exports: Dict[str, str] | None = None,
) -> str:
    shell_parts = []
    if CONDA_SH and Path(CONDA_SH).is_file():
        shell_parts.extend((
            f"source {shlex.quote(CONDA_SH)}",
            f"conda activate {shlex.quote(env_name)}",
        ))
    shell_parts.extend((
        f"cd {shlex.quote(str(cwd))}",
        f"export MPLCONFIGDIR={shlex.quote(str(DEFAULT_MPLCONFIGDIR))}",
        "export PYTHONUNBUFFERED=1",
    ))
    for key, value in (exports or {}).items():
        shell_parts.append(f"export {shlex.quote(str(key))}={shlex.quote(str(value))}")
    shell_parts.append(qjoin(argv))
    return "/bin/bash -lc " + shlex.quote(" && ".join(shell_parts))


def python_cmd(script: Path, args: Sequence[Any]) -> List[str]:
    return ["python", str(script), *[str(arg) for arg in args]]


def write_command_files(rows: List[Dict[str, Any]], output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{stem}.csv"
    sh_path = output_dir / f"{stem}.sh"
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with sh_path.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for row in rows:
            label = row.get("label") or row.get("variant") or row.get("method") or "command"
            f.write(f"# {label}\n{row['command']}\n\n")
    return csv_path, sh_path


def print_written(rows: List[Dict[str, Any]], csv_path: Path, sh_path: Path) -> None:
    print(f"generated {len(rows)} commands")
    print(f"wrote {csv_path}")
    print(f"wrote {sh_path}")
    for row in rows[:10]:
        print(row["command"])
    if len(rows) > 10:
        print(f"... {len(rows) - 10} more commands in {sh_path}")
