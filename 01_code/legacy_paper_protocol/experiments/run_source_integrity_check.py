from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_metadrive_smoke import run_methods
from experiments.run_external_diffusion_baseline_checks import collect_rows as collect_external_rows


def _row(component: str, status: str, detail: str = "") -> Dict[str, Any]:
    return {"component": component, "status": status, "detail": detail}


def _add_path(path: Path) -> None:
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def check_imports() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checks = [
        ("local_pvpdacer", lambda: __import__("relax.algorithm.pvp_dacer")),
        ("local_metadrive_env", lambda: __import__("relax_env.human_in_the_loop_env")),
    ]
    _add_path(ROOT / "baselines" / "pvp")
    checks.append(("pvp_pvptd3", lambda: __import__("pvp.pvp_td3")))
    _add_path(ROOT / "baselines" / "haco")
    checks.append(("haco_env", lambda: __import__("haco.utils.human_in_the_loop_env")))
    _add_path(ROOT / "baselines" / "hdsac")
    _add_path(ROOT / "baselines" / "hdsac" / "utils")
    checks.append(("hdsac_dsac", lambda: __import__("dsac_v2_pvp")))

    for name, fn in checks:
        try:
            fn()
            rows.append(_row(name, "ok"))
        except Exception as exc:
            rows.append(_row(name, "failed", f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    rows.append(_row("haco_trainer_dependency_ray", "ok" if importlib.util.find_spec("ray") else "missing", "HACO full trainer imports ray/rllib"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check trimmed source imports and one-step MetaDrive usability.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "integrity")
    args = parser.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = check_imports()
    for item in collect_external_rows(run_cep_toy_smoke=False, seed=args.seed):
        rows.append(_row(f"{item['method']}_{item['component']}", item["status"], item.get("detail", "")))
    smoke_rows = run_methods(["dacer", "pvp", "hdsac", "haco"], args.seed, max(1, args.steps), args.output_dir)
    for item in smoke_rows:
        rows.append(_row(f"metadrive_smoke_{item['method']}", item["status"], item.get("note", "")))

    out_csv = args.output_dir / "source_integrity.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["component", "status", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
