#!/usr/bin/env python3
"""Aggregate checkpoint-based DOVE-ER mechanism diagnostics.

This script intentionally does not read TensorBoard scalar curves. The target
EnergyRank/CPCal tags were not recorded in the available logs, so mechanism
tables are rebuilt from saved checkpoints, intervention-pair diagnostics, and
episode CSVs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

try:
    from sklearn.metrics import roc_auc_score
except Exception:  # pragma: no cover
    roc_auc_score = None


PAIR_VARIANTS: Sequence[Tuple[str, str, str]] = (
    ("run11_no_er", "DBP + proxy critic, no EnergyRank", "critic_pair_diag_run11_no_er"),
    ("run13_no_cpcal", "EnergyRank + proxy critic, no CPCal", "critic_pair_diag_run13_no_cpcal"),
    ("run12_full", "EnergyRank + CPCal-calibrated proxy critic", "critic_pair_diag_run12_full"),
)

TRAINING_EPISODE_VARIANTS: Sequence[Tuple[str, str, str]] = (
    ("run11_no_er", "DBP + proxy critic, no EnergyRank", "run11_pvp_dbp_critic"),
    ("run13_no_cpcal", "EnergyRank + proxy critic, no CPCal", "run13_doveer_no_pv_constraint"),
    ("run12_full", "EnergyRank + CPCal-calibrated proxy critic", "run12_doveer_full"),
)

GATE_VARIANTS: Sequence[Tuple[str, str]] = (
    ("upvgds_no_gate", "UPV-GDS without Final Gate"),
    ("final_gate_proxy_only", "UPV-GDS + proxy-improvement gate"),
    ("final_gate_prior_only", "UPV-GDS + prior-compatibility gate"),
    ("final_gate_no_disagreement", "UPV-GDS + gate without ensemble check"),
    ("upvgds_final_gate_full", "UPV-GDS + full Final Gate"),
)

GATE_COLUMNS: Sequence[str] = (
    "final_gate_accept_rate",
    "final_gate_proxy_ok_rate",
    "final_gate_prior_ok_rate",
    "final_gate_bound_ok_rate",
    "final_gate_uncertainty_ok_rate",
    "final_gate_reject_proxy_rate",
    "final_gate_reject_prior_rate",
    "final_gate_reject_uncertainty_rate",
    "final_gate_reject_shift_rate",
    "final_gate_delta_c_mean",
    "final_gate_proxy_gain_mean",
    "inference_latency_mean",
)


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[skip] could not read {path}: {exc}")
        return None


def _mean(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return float(x.mean()) if len(x) else float("nan")


def _se(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if len(x) <= 1:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def _fmt(value: float, ndigits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "--"
    return f"{value:.{ndigits}f}"


def _fmt_pm(mean: float, se: float, ndigits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(se):
        return _fmt(mean, ndigits)
    return f"{mean:.{ndigits}f} $\\pm$ {se:.{ndigits}f}"


def _write_tex_table(path: Path, df: pd.DataFrame, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(df.to_latex(index=False, escape=False))
        f.write("\\end{table}\n")


def _aggregate_pair_summary(diagnostic_root: Path, marker: str) -> Optional[Dict[str, object]]:
    candidates = sorted(diagnostic_root.rglob(f"{marker}/intervention_pair_summary.csv"))
    if not candidates:
        candidates = sorted(diagnostic_root.rglob(f"*{marker}*/intervention_pair_summary.csv"))
    frames: List[pd.DataFrame] = []
    for path in candidates:
        df = _safe_read_csv(path)
        if df is not None and len(df):
            df["source_summary"] = str(path)
            frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    out: Dict[str, object] = {
        "policy_name": "pooled_checkpoints",
        "n_checkpoints": int(len(all_df)),
        "n_pairs": int(pd.to_numeric(all_df.get("n_pairs", pd.Series(dtype=float)), errors="coerce").max()),
        "source_summary": ";".join(sorted({str(x) for x in all_df.get("source_summary", [])})),
    }
    for col in all_df.columns:
        if col in {"policy_name", "policy_path", "source_summary"}:
            continue
        vals = pd.to_numeric(all_df[col], errors="coerce")
        if vals.notna().any():
            out[col] = float(vals.mean())
            out[f"{col}_se_over_checkpoints"] = _se(vals)
    return out


def build_pair_tables(diagnostic_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    actor_rows: List[Dict[str, object]] = []
    for variant_id, label, marker in PAIR_VARIANTS:
        row = _aggregate_pair_summary(diagnostic_root, marker)
        if row is None:
            rows.append({"variant_id": variant_id, "Variant": label, "n_pairs": 0})
            actor_rows.append({"variant_id": variant_id, "Variant": label, "n_pairs": 0})
            continue
        rows.append(
            {
                "variant_id": variant_id,
                "Variant": label,
                "policy": row.get("policy_name", ""),
                "n_checkpoints": int(row.get("n_checkpoints", 0)),
                "n_pairs": int(row.get("n_pairs", 0)),
                "PairAcc_C": row.get("pairacc_c_mean", np.nan),
                "PairSat_C": row.get("pairsat_c_margin_mean", np.nan),
                "V_C": row.get("v_c_mean", np.nan),
                "DeltaC_mean": row.get("delta_c_mean_mean", np.nan),
                "DeltaC_robust": row.get("delta_c_robust_mean", np.nan),
                "Unc_pos": row.get("unc_pos_mean", np.nan),
                "Unc_neg": row.get("unc_neg_mean", np.nan),
                "source_summary": row.get("source_summary", ""),
            }
        )
        actor_rows.append(
            {
                "variant_id": variant_id,
                "Variant": label,
                "policy": row.get("policy_name", ""),
                "n_checkpoints": int(row.get("n_checkpoints", 0)),
                "n_pairs": int(row.get("n_pairs", 0)),
                "PairSat_E": row.get("PairSat_E_mean", np.nan),
                "V_E": row.get("V_E_mean", np.nan),
                "IAR": row.get("IAR_mean", np.nan),
                "DeltaE": row.get("energy_gap_neg_minus_pos_mean", np.nan),
                "source_summary": row.get("source_summary", ""),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(actor_rows)


def _episode_paths(episode_root: Path, run_marker: str, mode: str) -> List[Path]:
    paths = sorted(
        p
        for p in episode_root.rglob(f"episodes_{mode}.csv")
        if run_marker in str(p)
    )
    # Prefer the current Table VI state if present, otherwise keep all matches.
    preferred = [p for p in paths if "current_per_csv_370_state" in str(p)]
    if preferred:
        return preferred
    preferred = [p for p in paths if "before_reward370_map" in str(p)]
    return preferred or paths


def _concat_episode_paths(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        df = _safe_read_csv(path)
        if df is not None and len(df):
            df["source_csv"] = str(path)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _gate_summary_row(panel: str, variant_id: str, label: str, df: pd.DataFrame) -> Dict[str, object]:
    row: Dict[str, object] = {
        "panel": panel,
        "variant_id": variant_id,
        "Variant": label,
        "n_episodes": int(len(df)),
        "n_csv": int(df["source_csv"].nunique()) if "source_csv" in df else 0,
    }
    for col in GATE_COLUMNS:
        if col in df:
            row[f"{col}_mean"] = _mean(df[col])
            row[f"{col}_se"] = _se(df[col])
    return row


def build_gate_table(episode_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for variant_id, label, run_marker in TRAINING_EPISODE_VARIANTS:
        df = _concat_episode_paths(_episode_paths(episode_root, run_marker, "upvgds_final_gate_full"))
        rows.append(_gate_summary_row("training_variant", variant_id, label, df))
    for mode, label in GATE_VARIANTS:
        df = _concat_episode_paths(_episode_paths(episode_root, "run12_doveer_full", mode))
        rows.append(_gate_summary_row("gate_variant", mode, label, df))
    return pd.DataFrame(rows)


def build_outcome_table(episode_root: Path) -> pd.DataFrame:
    paths: List[Path] = []
    for _, _, run_marker in TRAINING_EPISODE_VARIANTS:
        paths.extend(_episode_paths(episode_root, run_marker, "upvgds_final_gate_full"))
    df = _concat_episode_paths(paths)
    if df.empty:
        return pd.DataFrame()
    metrics = [
        "final_gate_delta_c_mean",
        "final_gate_proxy_gain_mean",
        "guidance_proxy_score_mean",
        "final_gate_action_shift_mean",
        "final_gate_uq_guided_mean",
        "guidance_uncertainty_mean",
    ]
    targets = ["success", "crash", "cost", "route_completion", "reward"]
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        if metric not in df:
            continue
        x = pd.to_numeric(df[metric], errors="coerce")
        for target in targets:
            if target not in df:
                continue
            y = pd.to_numeric(df[target], errors="coerce")
            mask = x.notna() & y.notna()
            if int(mask.sum()) < 5:
                continue
            out: Dict[str, object] = {
                "metric": metric,
                "target": target,
                "n": int(mask.sum()),
            }
            if spearmanr is not None:
                rho, p_value = spearmanr(x[mask], y[mask])
                out["spearman"] = float(rho)
                out["p_value"] = float(p_value)
            if roc_auc_score is not None and target in {"success", "crash"} and y[mask].nunique() == 2:
                try:
                    out["auc"] = float(roc_auc_score(y[mask], x[mask]))
                except Exception:
                    out["auc"] = np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def _display_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["PairAcc_C", "PairSat_C", "V_C", "DeltaC_mean", "DeltaC_robust", "Unc_pos", "Unc_neg"]:
        if col in display:
            display[col] = display[col].map(lambda x: _fmt(float(x), 3) if pd.notna(x) else "--")
    return display[["Variant", "n_checkpoints", "n_pairs", "PairAcc_C", "PairSat_C", "V_C", "DeltaC_mean", "DeltaC_robust", "Unc_pos", "Unc_neg"]]


def _display_actor_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for col in ["PairSat_E", "V_E", "IAR", "DeltaE"]:
        if col in display:
            display[col] = display[col].map(lambda x: _fmt(float(x), 3) if pd.notna(x) else "--")
    return display[["Variant", "n_checkpoints", "n_pairs", "PairSat_E", "V_E", "IAR", "DeltaE"]]


def _display_gate_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ("final_gate_accept_rate", "GateAcc"),
        ("final_gate_proxy_ok_rate", "ProxyOK"),
        ("final_gate_prior_ok_rate", "PriorOK"),
        ("final_gate_uncertainty_ok_rate", "UncOK"),
        ("final_gate_reject_proxy_rate", "RejectProxy"),
        ("final_gate_reject_prior_rate", "RejectPrior"),
        ("final_gate_reject_uncertainty_rate", "RejectUnc"),
        ("final_gate_reject_shift_rate", "RejectShift"),
    ]
    out = df[["panel", "Variant", "n_episodes"]].copy()
    for source, name in cols:
        mean_col = f"{source}_mean"
        se_col = f"{source}_se"
        if mean_col in df:
            out[name] = [
                _fmt_pm(float(m), float(s), 3)
                for m, s in zip(df[mean_col], df.get(se_col, pd.Series(np.nan, index=df.index)))
            ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--diagnostic_root", type=Path, default=Path("eval2/diagnostics"))
    ap.add_argument("--episode_root", type=Path, default=Path("eval2"))
    ap.add_argument("--output_dir", type=Path, default=Path("eval2/diagnostics/tables"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_df, actor_df = build_pair_tables(args.diagnostic_root)
    gate_df = build_gate_table(args.episode_root)
    outcome_df = build_outcome_table(args.episode_root)

    pair_df.to_csv(args.output_dir / "critic_pair_calibration_table.csv", index=False)
    actor_df.to_csv(args.output_dir / "actor_prior_audit_table.csv", index=False)
    gate_df.to_csv(args.output_dir / "gate_rejection_reason_table.csv", index=False)
    outcome_df.to_csv(args.output_dir / "critic_outcome_correlation_table.csv", index=False)

    _write_tex_table(
        args.output_dir / "critic_pair_calibration_table.tex",
        _display_pair_table(pair_df),
        "Proxy-risk critic pair-calibration diagnostics computed from saved checkpoints and retained intervention pairs.",
        "tab:critic_pair_calibration",
    )
    _write_tex_table(
        args.output_dir / "actor_prior_audit_table.tex",
        _display_actor_table(actor_df),
        "Actor-prior audit computed from saved checkpoints and retained intervention pairs.",
        "tab:actor_prior_audit_checkpoint",
    )
    _write_tex_table(
        args.output_dir / "gate_rejection_reason_table.tex",
        _display_gate_table(gate_df),
        "Final Gate predicate and rejection diagnostics aggregated from episode CSV files.",
        "tab:gate_rejection_reason",
    )
    if not outcome_df.empty:
        outcome_df.to_latex(args.output_dir / "critic_outcome_correlation_table.tex", index=False, escape=False)

    print(f"[wrote] {args.output_dir}")


if __name__ == "__main__":
    main()
