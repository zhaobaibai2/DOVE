from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import DEFAULT_LOG_DIR, DEFAULT_POLICY
from experiments.run_pdf_mechanism_action_shift import load_demo_samples
from scripts import eval_pvp_policies_fixed_v3 as ev


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.10.3 mechanism check: uncertainty gate bucket analysis.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--max_states", type=int, default=128)
    parser.add_argument("--output_csv", type=Path, default=ROOT / "runs" / "pdf_mechanism_uncertainty_gate" / "uncertainty_gate.csv")
    args = parser.parse_args()
    args.output_csv = args.output_csv.expanduser().resolve()

    samples = load_demo_samples(args.demo_root, max(1, args.max_states), only_interventions=False)
    if not samples:
        raise RuntimeError(f"no demo samples found under {args.demo_root}")

    ev.require_jax_gpu_ready()
    run_args = ev.load_run_args(args.log_dir)
    agent = ev.build_agent(int(samples[0][0].shape[0]), run_args)
    payload = ev.load_policy_payload(args.log_dir / args.policy)
    cfg = {
        "use_value_guidance": True,
        "lambda_0": 0.5,
        "beta_unc": 1.0,
        "p_decay": 1.0,
        "grad_clip": 1.0,
        "guidance_mode": "proxy_value",
        "guidance_mode_id": ev.GUIDANCE_MODE_IDS["proxy_value"],
        "guidance_target": "x0",
        "guidance_target_id": ev.GUIDANCE_TARGET_IDS["x0"],
        "guidance_step_interval": 1,
        "critic_params": "target",
        "guidance_q_agg": "lcb",
        "guidance_q_agg_id": ev.GUIDANCE_Q_AGG_IDS["lcb"],
        "critic_objective": "lower",
        "critic_objective_id": ev.CRITIC_OBJECTIVE_IDS["lower"],
        "guidance_kappa": 1.0,
        "guidance_injection": "clean_x0",
        "guidance_injection_id": ev.GUIDANCE_INJECTION_IDS["clean_x0"],
        "guidance_schedule": "noise_level",
        "guidance_schedule_id": ev.GUIDANCE_SCHEDULE_IDS["noise_level"],
        "rerank_topk": 0,
    }
    unguided = ev.resolve_action_adapter(agent, payload, samples[0][0], guidance_cfg={})
    guided = ev.resolve_action_adapter(agent, payload, samples[0][0], guidance_cfg=cfg)
    if unguided is None or guided is None:
        raise RuntimeError("failed to build action adapters")

    rows: List[Dict[str, Any]] = []
    for obs, _human, source, index in samples:
        a0 = ev.det_action(agent, unguided, obs)
        ag = ev.det_action(agent, guided, obs)
        info = ev.guidance_dry_run(agent, guided, obs) or {}
        rows.append({
            "source": source,
            "index": index,
            "sigma_q": float(info.get("guidance/sigma_q", np.nan)),
            "sigma_ref": float(info.get("guidance/sigma_ref", np.nan)),
            "lambda": float(info.get("guidance/lambda", np.nan)),
            "q_grad_norm": float(info.get("guidance/q_grad_norm", np.nan)),
            "q_mean": float(info.get("guidance/q_mean", np.nan)),
            "action_shift": float(np.linalg.norm(ag - a0)),
        })

    sigma_values = np.asarray([r["sigma_q"] for r in rows], dtype=np.float32)
    q1, q2 = np.nanquantile(sigma_values, [1.0 / 3.0, 2.0 / 3.0])
    for row in rows:
        if row["sigma_q"] <= q1:
            row["uncertainty_bucket"] = "low"
        elif row["sigma_q"] <= q2:
            row["uncertainty_bucket"] = "medium"
        else:
            row["uncertainty_bucket"] = "high"

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")
    for bucket in ("low", "medium", "high"):
        subset = [r for r in rows if r["uncertainty_bucket"] == bucket]
        print(
            f"{bucket}: n={len(subset)} "
            f"lambda={np.nanmean([r['lambda'] for r in subset]):.6f} "
            f"action_shift={np.nanmean([r['action_shift'] for r in subset]):.6f}"
        )


if __name__ == "__main__":
    main()
