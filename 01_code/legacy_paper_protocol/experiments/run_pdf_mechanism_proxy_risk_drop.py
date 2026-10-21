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
from experiments.run_pdf_mechanism_action_shift import load_demo_samples, q_score
from scripts import eval_pvp_policies_fixed_v3 as ev


VARIANTS = [
    ("none", False, "none"),
    ("random_grad", True, "random_grad"),
    ("opposite_signed_proxy", True, "reverse_grad"),
    ("proxy_value", True, "proxy_value"),
]


def cfg_for(use_guidance: bool, mode: str, q_agg: str) -> Dict[str, Any]:
    if not use_guidance:
        return {}
    return {
        "use_value_guidance": True,
        "lambda_0": 0.5,
        "beta_unc": 1.0,
        "p_decay": 1.0,
        "grad_clip": 1.0,
        "guidance_mode": mode,
        "guidance_mode_id": ev.GUIDANCE_MODE_IDS[mode],
        "guidance_target": "x0",
        "guidance_target_id": ev.GUIDANCE_TARGET_IDS["x0"],
        "guidance_step_interval": 1,
        "critic_params": "target",
        "guidance_q_agg": q_agg,
        "guidance_q_agg_id": ev.GUIDANCE_Q_AGG_IDS[q_agg],
        "guidance_kappa": 1.0,
        "guidance_injection": "clean_x0",
        "guidance_injection_id": ev.GUIDANCE_INJECTION_IDS["clean_x0"],
        "guidance_schedule": "noise_level",
        "guidance_schedule_id": ev.GUIDANCE_SCHEDULE_IDS["noise_level"],
        "rerank_topk": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF mechanism check: proxy risk decrease under corrected UPV-GDS.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--max_states", type=int, default=64)
    parser.add_argument("--guidance_q_agg", choices=["min", "mean", "single_q1", "lcb", "ucb", "conservative"], default="conservative")
    parser.add_argument("--output_csv", type=Path, default=ROOT / "runs" / "pdf_mechanism_proxy_risk_drop" / "proxy_risk_drop.csv")
    args = parser.parse_args()
    args.output_csv = args.output_csv.expanduser().resolve()

    samples = load_demo_samples(args.demo_root, max(1, args.max_states), only_interventions=False)
    if not samples:
        raise RuntimeError(f"no demo samples found under {args.demo_root}")

    ev.require_jax_gpu_ready()
    run_args = ev.load_run_args(args.log_dir)
    agent = ev.build_agent(int(samples[0][0].shape[0]), run_args)
    payload = ev.load_policy_payload(args.log_dir / args.policy)
    components = ev._extract_guidance_components_from_payload(payload)
    if payload is None or components is None:
        raise RuntimeError("policy payload must include q critics")
    q_params = components["target_q_params"]

    adapters: Dict[str, Any] = {}
    for name, use_guidance, mode in VARIANTS:
        adapters[name] = ev.resolve_action_adapter(agent, payload, samples[0][0], guidance_cfg=cfg_for(use_guidance, mode, args.guidance_q_agg))
        if adapters[name] is None:
            raise RuntimeError(f"failed to build adapter for {name}")

    rows: List[Dict[str, Any]] = []
    for obs, _human, source, index in samples:
        scores = {}
        for name, _use_guidance, _mode in VARIANTS:
            action = ev.det_action(agent, adapters[name], obs)
            scores[name] = q_score(agent, q_params, obs, action, args.guidance_q_agg)
        row = {"source": source, "index": index}
        row.update({f"proxy_risk_{k}": v for k, v in scores.items()})
        row["risk_drop_proxy_vs_none"] = scores["none"] - scores["proxy_value"]
        row["risk_drop_random_vs_none"] = scores["none"] - scores["random_grad"]
        row["risk_drop_opposite_vs_none"] = scores["none"] - scores["opposite_signed_proxy"]
        rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output_csv}")
    print(f"mean risk_drop_proxy_vs_none={np.mean([r['risk_drop_proxy_vs_none'] for r in rows]):.6f}")


if __name__ == "__main__":
    main()
