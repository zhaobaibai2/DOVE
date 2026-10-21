from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.cep.adapter import dependency_status as cep_dependency_status
from baselines.cep.adapter import run_toy_smoke
from baselines.vgd.adapter import dependency_status as vgd_dependency_status
from baselines.vgd.adapter import runnable as vgd_runnable


def _row(method: str, component: str, status: str, detail: str = "") -> Dict[str, Any]:
    return {"method": method, "component": component, "status": status, "detail": detail}


def collect_rows(run_cep_toy_smoke: bool, seed: int, device: str = "cuda:0") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, status in cep_dependency_status().items():
        detail = "required for CEP D4RL training" if name == "d4rl" and status == "missing" else ""
        rows.append(_row("cep", name, status, detail))
    if run_cep_toy_smoke:
        try:
            run_toy_smoke(device=device, seed=seed)
            rows.append(_row("cep", f"toy_smoke_{device}", "ok"))
        except Exception as exc:
            traceback.print_exc()
            rows.append(_row("cep", f"toy_smoke_{device}", "failed", f"{type(exc).__name__}: {exc}"))

    for name, status in vgd_dependency_status().items():
        detail = ""
        if name in {"hydra", "omegaconf"} and status == "missing":
            detail = "required by train_vgd.py"
        if name in {"local_dppo_submodule", "local_stable_baselines3_submodule"} and status == "missing":
            detail = "source checkout has an empty submodule directory"
        rows.append(_row("vgd", name, status, detail))
    rows.append(_row("vgd", "runnable", "ok" if vgd_runnable() else "blocked", "full VGD training needs hydra/omegaconf and populated local submodules"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check imported CEP/VGD external diffusion baselines.")
    parser.add_argument("--run_cep_toy_smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0", help="device for CEP toy smoke, default cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "external_diffusion_baselines")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(args.run_cep_toy_smoke, args.seed, args.device)
    out_csv = args.output_dir / "external_diffusion_baseline_checks.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
