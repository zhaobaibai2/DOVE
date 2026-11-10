from pathlib import Path
import math
import re

import pandas as pd


ROOT = Path(__file__).resolve().parent
DOVE_ROUTE1_RUNS = [
    Path("/home/aaa/data/real_car/暑假/0625/results/route1_action/control_logs/20260624_154816/control_commands.csv"),
    Path("/home/aaa/data/real_car/暑假/0625/results/route1_action/control_logs/20260624_164122/control_commands.csv"),
]


def method_from_dir(name: str) -> str:
    upper = name.upper()
    if "DACER" in upper:
        return "DACER"
    if upper.startswith("IWR"):
        return "IWR"
    if upper.startswith("PVP_TD3"):
        return "PVP-TD3"
    return name.split("_EVAL_")[0]


def scenario_from_dir(run_dir: Path) -> str:
    md = sorted(p.stem for p in run_dir.glob("*.md"))
    return md[0] if md else "-"


def metric_row(method: str, scenario: str, run: str, t, speed, lateral, yaw) -> dict:
    t = pd.to_numeric(t, errors="coerce")
    speed = pd.to_numeric(speed, errors="coerce")
    lateral = pd.to_numeric(lateral, errors="coerce")
    yaw = pd.to_numeric(yaw, errors="coerce")
    valid = pd.DataFrame({"t": t, "speed": speed}).dropna()
    valid = valid.sort_values("t").drop_duplicates("t")
    valid = valid[valid["t"].diff().fillna(1) > 0]
    dt = valid["t"].diff()
    acc = valid["speed"].diff() / dt
    jerk = acc.diff() / dt
    jerk = jerk.replace([float("inf"), float("-inf")], pd.NA).dropna()

    return {
        "method": method,
        "scenario": scenario,
        "run": run,
        "samples": int(pd.DataFrame({"t": t, "speed": speed, "lateral": lateral}).dropna().shape[0]),
        "duration_s": float(t.max() - t.min()),
        "avg_speed": float(speed.mean()),
        "max_speed": float(speed.max()),
        "lateral_mae": float(lateral.abs().mean()),
        "lateral_rmse": float((lateral.pow(2).mean()) ** 0.5),
        "yaw_mae": float(yaw.abs().mean()),
        "mean_abs_jerk": float(jerk.abs().mean()) if len(jerk) else float("nan"),
        "rms_jerk": float((jerk.pow(2).mean()) ** 0.5) if len(jerk) else float("nan"),
    }


def eval_log_metrics(csv_path: Path) -> dict:
    run_dir = csv_path.parent.parent
    df = pd.read_csv(csv_path).sort_values("ts").drop_duplicates("ts")
    return metric_row(
        method_from_dir(run_dir.name),
        scenario_from_dir(run_dir),
        run_dir.name,
        df["ts"],
        df["carspeed"],
        df["error_distance"],
        df["error_yaw"],
    )


def dove_route1_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).dropna(
        subset=["actual_timestamp", "actual_speed_mps", "path_lateral_error_m"]
    )
    yaw_rad = pd.to_numeric(df["path_heading_error_deg"], errors="coerce") * math.pi / 180.0
    return metric_row(
        "DOVE",
        "route1_action",
        csv_path.parent.name,
        df["actual_timestamp"],
        df["actual_speed_mps"],
        df["path_lateral_error_m"],
        yaw_rad,
    )


def fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def latex_escape(text: str) -> str:
    return re.sub(r"([_&%$#{}])", r"\\\1", str(text))


def main() -> None:
    rows = [eval_log_metrics(p) for p in sorted(ROOT.glob("*/eval_logs/*.csv"))]
    rows.extend(dove_route1_metrics(p) for p in DOVE_ROUTE1_RUNS if p.exists())
    detail = pd.DataFrame(rows).sort_values(["method", "scenario", "run"])
    detail.to_csv(ROOT / "metrics_by_run.csv", index=False)

    grouped = detail.groupby("method", sort=True)
    summary = grouped.agg(
        runs=("run", "count"),
        avg_speed_mean=("avg_speed", "mean"),
        avg_speed_std=("avg_speed", "std"),
        lateral_mae_mean=("lateral_mae", "mean"),
        lateral_mae_std=("lateral_mae", "std"),
        lateral_rmse_mean=("lateral_rmse", "mean"),
        yaw_mae_mean=("yaw_mae", "mean"),
        mean_abs_jerk_mean=("mean_abs_jerk", "mean"),
        mean_abs_jerk_std=("mean_abs_jerk", "std"),
    ).reset_index()
    summary.to_csv(ROOT / "metrics_by_method.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Real-car evaluation metrics averaged over runs. Lateral error is computed as the absolute lateral-offset signal in each log; jerk is computed from the second derivative of speed with respect to timestamp.}",
        r"\label{tab:real_car_metrics}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Runs & Avg. speed & Lat. MAE & Lat. RMSE & Yaw MAE & Mean $\lvert$jerk$\rvert$ \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{latex_escape(row['method'])} & "
            f"{int(row['runs'])} & "
            f"{fmt(row['avg_speed_mean'])} $\\pm$ {fmt(row['avg_speed_std'])} & "
            f"{fmt(row['lateral_mae_mean'])} $\\pm$ {fmt(row['lateral_mae_std'])} & "
            f"{fmt(row['lateral_rmse_mean'])} & "
            f"{fmt(row['yaw_mae_mean'])} & "
            f"{fmt(row['mean_abs_jerk_mean'])} $\\pm$ {fmt(row['mean_abs_jerk_std'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (ROOT / "metrics_three_line_table.tex").write_text("\n".join(lines), encoding="utf-8")

    detail_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Run-level real-car evaluation metrics.}",
        r"\label{tab:real_car_metrics_by_run}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"Method & Scenario & Run & Avg. speed & Lat. MAE & Yaw MAE & Mean $\lvert$jerk$\rvert$ \\",
        r"\midrule",
    ]
    for _, row in detail.iterrows():
        detail_lines.append(
            f"{latex_escape(row['method'])} & "
            f"{latex_escape(row['scenario'])} & "
            f"{latex_escape(row['run'])} & "
            f"{fmt(row['avg_speed'])} & "
            f"{fmt(row['lateral_mae'])} & "
            f"{fmt(row['yaw_mae'])} & "
            f"{fmt(row['mean_abs_jerk'])} \\\\"
        )
    detail_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (ROOT / "metrics_by_run_table.tex").write_text("\n".join(detail_lines), encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
