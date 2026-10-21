from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import ROOT


MAIN_COLUMNS = [
    # task / safety
    "avg_success",
    "avg_crash",
    "avg_out_of_route",
    "avg_timeout",
    "avg_route_completion",
    "avg_closed_loop_distance",
    # HITL or synthetic/proxy intervention fields when available
    "avg_takeover_rate",
    "avg_intervention_count",
    "avg_interventions_per_km",
    "avg_time_to_first_intervention",
    # comfort / deployment
    "avg_speed_mean",
    "avg_accel_mean",
    "avg_jerk_mean",
    "avg_steering_variance",
    "avg_accel_cmd_variance",
    "avg_action_saturation_ratio",
    "avg_inference_latency_mean",
    "avg_inference_latency_p95",
    # final gate diagnostics
    "avg_final_gate_accept_rate",
    "avg_final_gate_score_mean",
    "avg_final_gate_proxy_gain_mean",
    "avg_final_gate_delta_c_mean",
    "avg_final_gate_delta_e_mean",
    "avg_final_gate_uses_proxy_rate",
    "avg_final_gate_uses_prior_rate",
    "avg_final_gate_proxy_gain_lcb_mean",
    "avg_final_gate_proxy_gain_robust_mean",
    "avg_final_gate_prior_improvement_mean",
    "avg_final_gate_prior_delta_mean",
    "avg_final_gate_action_shift_mean",
    "avg_final_gate_energy_shift_mean",
    "avg_final_gate_hard_ok_rate",
    "avg_final_gate_prior_ok_rate",
    "avg_final_gate_proxy_ok_rate",
    "avg_final_gate_bound_ok_rate",
    "avg_final_gate_uncertainty_ok_rate",
    "avg_final_gate_tau_prior_mean",
    "avg_final_gate_tau_u_mean",
    "avg_final_gate_proxy_min_gain_mean",
    "avg_final_gate_reject_low_score_rate",
    "avg_final_gate_reject_proxy_rate",
    "avg_final_gate_reject_bound_rate",
    "avg_final_gate_reject_shift_rate",
    "avg_final_gate_reject_prior_rate",
    "avg_final_gate_reject_uncertainty_rate",
    "avg_final_gate_reject_nonfinite_rate",
    # UPV-GDS diagnostics
    "avg_guidance_q_grad_norm_mean",
    "avg_guidance_q_grad_norm_preclip_mean",
    "avg_guidance_grad_clip_frac_mean",
    "avg_guidance_grad_nan_frac_mean",
    "avg_guidance_lambda_mean",
    "avg_guidance_uncertainty_mean",
    "avg_guidance_x0_clip_frac_mean",
    "avg_guidance_q_mean",
    "avg_guidance_action_shift_mean",
    "avg_guidance_anchor_project_rate_mean",
    "avg_guidance_anchor_shift_mean",
    "avg_guidance_max_action_shift_mean",
    "avg_guidance_clean_injection_rate",
    # prior-aliasing mechanism diagnostics (from run_pdf_mechanism_prior_aliasing.py)
    "pairsat_e",
    "v_e",
    "iar",
    "ragr",
    "energy_gap_neg_minus_pos",
    "min_dist_to_rejected_mean",
]


def _float_or_nan(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def read_numeric_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: _float_or_nan(v) for k, v in raw.items()}
            rows.append(row)
    return rows


def summarize(values: Iterable[float]) -> Tuple[float, float, float, int]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    if len(vals) == 1:
        return vals[0], 0.0, 0.0, 1
    sd = stdev(vals)
    se = sd / math.sqrt(len(vals))
    return mean(vals), sd, se, len(vals)


def summarize_eval_csv(path: Path, variant: str) -> Dict[str, float | str | int]:
    numeric_rows = read_numeric_rows(path)
    out: Dict[str, float | str | int] = {
        "variant": variant,
        "source_csv": str(path),
        "n_rows": len(numeric_rows),
    }
    for col in MAIN_COLUMNS:
        m, sd, se, n = summarize(row.get(col, float("nan")) for row in numeric_rows)
        out[f"{col}_mean"] = m
        out[f"{col}_std"] = sd
        out[f"{col}_se"] = se
        out[f"{col}_n"] = n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PDF experiment CSV outputs into paper-style mean ± SE tables.")
    parser.add_argument("--input_dirs", nargs="+", type=Path, default=[
        ROOT / "runs" / "pdf_main_closed_loop",
        ROOT / "runs" / "pdf_guidance_source_ablation",
        ROOT / "runs" / "pdf_uncertainty_ablation",
        ROOT / "runs" / "pdf_guidance_target_ablation",
        ROOT / "runs" / "pdf_critic_choice_ablation",
        ROOT / "runs" / "pdf_final_gate_ablation",
    ])
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_tables")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    rows: List[Dict[str, float | str | int]] = []
    for input_dir in args.input_dirs:
        input_dir = input_dir.expanduser().resolve()
        if not input_dir.exists():
            continue
        for csv_path in sorted(input_dir.glob("*.csv")):
            if "commands" in csv_path.name:
                continue
            rows.append(summarize_eval_csv(csv_path, csv_path.stem))
    if not rows:
        raise RuntimeError("no result CSVs found; run evaluation commands first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "pdf_metric_summary.csv"
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
