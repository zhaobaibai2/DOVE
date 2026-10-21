#!/usr/bin/env python3
"""Aggregate closed-loop critic reliability metrics from step-level eval logs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:  # pragma: no cover
    spearmanr = None

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
except Exception:  # pragma: no cover
    LogisticRegression = None
    brier_score_loss = None
    roc_auc_score = None


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    values = _num(series).dropna()
    return float(values.mean()) if len(values) else float("nan")


def _binary_auc(scores: pd.Series, labels: pd.Series) -> float:
    if roc_auc_score is None:
        return float("nan")
    x = _num(scores)
    y = _num(labels)
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 2 or y[mask].nunique() < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y[mask].astype(int), x[mask]))
    except Exception:
        return float("nan")


def _ece(prob: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out = 0.0
    for idx in range(int(n_bins)):
        lo, hi = bins[idx], bins[idx + 1]
        if idx == int(n_bins) - 1:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            continue
        out += float(mask.mean()) * abs(float(prob[mask].mean()) - float(labels[mask].mean()))
    return float(out)


def _platt_brier_ece(scores: pd.Series, labels: pd.Series, n_bins: int) -> Tuple[float, float, int]:
    if LogisticRegression is None or brier_score_loss is None:
        return float("nan"), float("nan"), 0
    x = _num(scores)
    y = _num(labels)
    mask = x.notna() & y.notna()
    x_np = x[mask].to_numpy(dtype=np.float64)
    y_np = y[mask].to_numpy(dtype=np.int64)
    if len(x_np) < 20 or len(np.unique(y_np)) < 2:
        return float("nan"), float("nan"), int(len(x_np))

    rng = np.random.default_rng(0)
    order = rng.permutation(len(x_np))
    split = max(1, len(order) // 2)
    val_idx = order[:split]
    test_idx = order[split:]
    if len(test_idx) == 0 or len(np.unique(y_np[val_idx])) < 2 or len(np.unique(y_np[test_idx])) < 2:
        return float("nan"), float("nan"), int(len(x_np))

    model = LogisticRegression(solver="lbfgs")
    model.fit(x_np[val_idx].reshape(-1, 1), y_np[val_idx])
    prob = model.predict_proba(x_np[test_idx].reshape(-1, 1))[:, 1]
    brier = float(brier_score_loss(y_np[test_idx], prob))
    ece = _ece(prob, y_np[test_idx], n_bins=n_bins)
    return brier, ece, int(len(x_np))


def _risk_bins(df: pd.DataFrame, score_col: str, label_col: str, n_bins: int, fail_col: Optional[str] = None) -> pd.DataFrame:
    cols = [score_col, label_col, "policy_name"]
    if fail_col and fail_col in df.columns:
        cols.append(fail_col)
    work = df[cols].copy()
    work[score_col] = _num(work[score_col])
    work[label_col] = _num(work[label_col])
    if fail_col and fail_col in work.columns:
        work[fail_col] = _num(work[fail_col])
    work = work.dropna(subset=[score_col, label_col])
    if work.empty:
        return pd.DataFrame()
    work = work.sort_values(score_col).reset_index(drop=True)
    chunks = np.array_split(work.index.to_numpy(), max(1, int(n_bins)))
    rows: List[Dict[str, object]] = []
    for idx, chunk in enumerate(chunks):
        if len(chunk) == 0:
            continue
        part = work.loc[chunk]
        row = {
            "policy_name": "pooled",
            "bin": idx,
            "n": int(len(part)),
            "score_mean": float(part[score_col].mean()),
            "score_min": float(part[score_col].min()),
            "score_max": float(part[score_col].max()),
            "collision_rate": float(part[label_col].mean()),
        }
        if fail_col and fail_col in part.columns:
            row["failure_rate"] = float(part[fail_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_mono(bin_df: pd.DataFrame, rate_col: str) -> float:
    if bin_df.empty or rate_col not in bin_df or len(bin_df) < 3 or spearmanr is None:
        return float("nan")
    try:
        rho, _ = spearmanr(bin_df["score_mean"], bin_df[rate_col])
        return float(rho)
    except Exception:
        return float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "NaN"
    return f"{value:.{digits}f}"


def summarize_one(df: pd.DataFrame, variant: str, n_bins: int) -> Tuple[Dict[str, object], pd.DataFrame]:
    score = "Cmax_ag" if "Cmax_ag" in df.columns else "Cmean_ag"
    if score not in df.columns:
        raise ValueError("step CSV must include Cmax_ag or Cmean_ag")

    collision_label = "future_collision_H"
    fail_label = "future_fail_H" if "future_fail_H" in df.columns else "episode_failure"
    episode_fail_label = "episode_failure" if "episode_failure" in df.columns else fail_label

    brier, ece, n_cal = _platt_brier_ece(df[score], df[collision_label], n_bins=n_bins)
    bin_df = _risk_bins(df, score, collision_label, n_bins=n_bins, fail_col=fail_label)
    bin_mono = _bin_mono(bin_df, "collision_rate")
    bin_mono_fail = _bin_mono(bin_df, "failure_rate")

    accepted = _num(df.get("gate_accepted", pd.Series(dtype=float))) > 0.5
    rejected = _num(df.get("gate_accepted", pd.Series(dtype=float))) <= 0.5
    future_collision = _num(df[collision_label])
    future_fail = _num(df[fail_label])
    delta_c = _num(df.get("delta_C", pd.Series(dtype=float)))
    improvement_label = 1.0 - future_fail

    median_score = _num(df[score]).median()
    critic_false_negative = ((_num(df[score]) <= median_score) & (future_collision > 0.5)).astype(float)
    disagr = _num(df.get("D_ag", pd.Series(dtype=float)))

    row: Dict[str, object] = {
        "variant": variant,
        "n_steps": int(len(df)),
        "n_episodes": int(df[["policy_name", "map_seed", "episode_idx"]].drop_duplicates().shape[0])
        if {"policy_name", "map_seed", "episode_idx"}.issubset(df.columns)
        else int(df.get("episode_idx", pd.Series(dtype=float)).nunique()),
        "score_col": score,
        "future_horizon": int(_num(df.get("reliability_horizon", pd.Series([math.nan]))).dropna().mode().iloc[0])
        if "reliability_horizon" in df and _num(df["reliability_horizon"]).notna().any()
        else math.nan,
        "AUROC_Coll": _binary_auc(df[score], df[collision_label]),
        "AUROC_Fail": _binary_auc(df[score], df[fail_label]),
        "AUROC_EpisodeFail": _binary_auc(df[score], df[episode_fail_label]),
        "ECE_Coll": ece,
        "Brier_Coll": brier,
        "calibration_n": n_cal,
        "BinMono_Coll": bin_mono,
        "BinMono_Fail": bin_mono_fail,
        "AUROC_DisagrErr": _binary_auc(disagr, critic_false_negative),
        "Corr_DeltaC_Improvement": float(delta_c.corr(improvement_label, method="spearman"))
        if delta_c.notna().sum() >= 3 and improvement_label.notna().sum() >= 3
        else float("nan"),
        "GateAcc": _safe_mean(df.get("gate_accepted", pd.Series(dtype=float))),
        "Accepted_FutureColl": float(future_collision[accepted].mean()) if accepted.any() else float("nan"),
        "Rejected_FutureColl": float(future_collision[rejected].mean()) if rejected.any() else float("nan"),
        "Accepted_FutureFail": float(future_fail[accepted].mean()) if accepted.any() else float("nan"),
        "Rejected_FutureFail": float(future_fail[rejected].mean()) if rejected.any() else float("nan"),
        "Mean_D_ag": _safe_mean(df.get("D_ag", pd.Series(dtype=float))),
        "Top10D_FutureColl": float(future_collision[disagr >= disagr.quantile(0.9)].mean())
        if disagr.notna().sum() >= 10
        else float("nan"),
    }
    bin_df.insert(0, "variant", variant)
    return row, bin_df


def write_markdown(summary: pd.DataFrame, bin_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Closed-loop critic reliability summary",
        "",
        "These metrics are computed from step-level closed-loop evaluation logs, not from retained intervention-pair ranking diagnostics.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Risk bins",
        "",
        bin_df.to_markdown(index=False) if not bin_df.empty else "No risk-bin rows were available.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step_csv", type=Path, nargs="+", required=True)
    parser.add_argument("--variant", type=str, nargs="*", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    bin_frames: List[pd.DataFrame] = []

    for idx, path in enumerate(args.step_csv):
        df = pd.read_csv(path)
        variant = args.variant[idx] if args.variant and idx < len(args.variant) else path.parent.name
        row, bins = summarize_one(df, variant=variant, n_bins=int(args.bins))
        row["source_csv"] = str(path)
        summary_rows.append(row)
        bin_frames.append(bins)

    summary = pd.DataFrame(summary_rows)
    risk_bins = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    summary.to_csv(args.output_dir / "critic_reliability_summary.csv", index=False)
    risk_bins.to_csv(args.output_dir / "critic_reliability_risk_bins.csv", index=False)
    write_markdown(summary, risk_bins, args.output_dir / "critic_reliability_summary.md")

    display_cols = [
        "variant", "n_steps", "n_episodes", "AUROC_Coll", "AUROC_Fail",
        "AUROC_EpisodeFail", "ECE_Coll", "Brier_Coll", "BinMono_Coll",
        "BinMono_Fail", "AUROC_DisagrErr", "GateAcc",
        "Accepted_FutureColl", "Rejected_FutureColl",
    ]
    print(summary[display_cols].to_string(index=False, formatters={
        col: (lambda x: _fmt(float(x))) for col in display_cols if col not in {"variant", "n_steps", "n_episodes"}
    }))
    print(f"[wrote] {args.output_dir}")


if __name__ == "__main__":
    main()
