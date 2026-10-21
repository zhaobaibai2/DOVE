#!/usr/bin/env python3
from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path("/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508")
PDF_IN = ROOT / "paper/chac/arXiv-2506.03568v2/pic/takeoverrate_plot.pdf"
EVENT_FILE = ROOT / (
    "logs/12_train_doveer_er_seed104_50000_20260522_104112/"
    "pvp_dacer_20260522_104125_s0/tb/"
    "events.out.tfevents.1779417685.aaa-MS-7E44.31580.0"
)
OUT_DIR = ROOT / "latex4/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RENDERED_PNG = OUT_DIR / "takeoverrate_plot_rendered_300dpi.png"
BASELINE_CSV = OUT_DIR / "takeoverrate_chac_pvp_digitized.csv"
OURS_CSV = OUT_DIR / "takeover_effort_trace.csv"
OUT_PNG = OUT_DIR / "takeover_rate_comparison_chac_pvp_ours.png"


def render_pdf(pdf_path: Path, png_path: Path) -> None:
    tmp_prefix = png_path.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-singlefile", "-png", "-r", "300", str(pdf_path), str(tmp_prefix)],
        check=True,
    )
    rendered = tmp_prefix.with_suffix(".png")
    if rendered != png_path:
        rendered.replace(png_path)


def clusters(indices: np.ndarray) -> list[int]:
    if indices.size == 0:
        return []
    groups = []
    start = prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx > prev + 1:
            groups.append((start + prev) // 2)
            start = idx
        prev = idx
    groups.append((start + prev) // 2)
    return groups


def detect_axes(img: np.ndarray) -> tuple[int, int, int, int, int, int]:
    black = (img[:, :, 0] < 40) & (img[:, :, 1] < 40) & (img[:, :, 2] < 40)
    h = clusters(np.where(black.sum(axis=1) > 0.70 * img.shape[1])[0])
    v = clusters(np.where(black.sum(axis=0) > 0.50 * img.shape[0])[0])
    if len(h) < 2 or len(v) < 2:
        raise RuntimeError("could not detect plot frame in source PDF rendering")

    top, bottom = min(h), max(h)
    left, right = min(v), max(v)

    plot = img[top : bottom + 1, left : right + 1]
    r, g, b = plot[:, :, 0], plot[:, :, 1], plot[:, :, 2]
    gray_grid = (
        (np.abs(r.astype(int) - g.astype(int)) < 5)
        & (np.abs(g.astype(int) - b.astype(int)) < 5)
        & (r > 185)
        & (r < 235)
    )
    grid_y = clusters(np.where(gray_grid.sum(axis=1) > 0.20 * plot.shape[1])[0])
    if len(grid_y) >= 3:
        y_one = top + grid_y[0]
        y_zero = top + grid_y[-1]
    else:
        # Fallback for this fixed CHAC/PVP figure: frame spans a little beyond 0..1.
        y_one = top + int(round(0.10 * (bottom - top)))
        y_zero = bottom - int(round(0.045 * (bottom - top)))
    return left, right, top, bottom, y_zero, y_one


def extract_curve(
    img: np.ndarray,
    axes: tuple[int, int, int, int, int, int],
    mask_fn,
    x_grid: np.ndarray,
) -> np.ndarray:
    left, right, top, bottom, y_zero, y_one = axes
    plot = img[top : bottom + 1, left : right + 1]
    mask = mask_fn(plot)
    xs, ys = [], []
    for col in range(mask.shape[1]):
        rows = np.where(mask[:, col])[0]
        if rows.size:
            xs.append(left + col)
            ys.append(top + float(np.median(rows)))
    if len(xs) < 10:
        raise RuntimeError("too few curve pixels found while digitizing source PDF")
    x_vals = (np.asarray(xs) - left) / (right - left) * 50.0
    y_vals = (y_zero - np.asarray(ys)) / (y_zero - y_one)
    order = np.argsort(x_vals)
    return np.clip(np.interp(x_grid, x_vals[order], y_vals[order]), 0.0, 1.1)


def digitize_baselines() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    render_pdf(PDF_IN, RENDERED_PNG)
    img = np.asarray(Image.open(RENDERED_PNG).convert("RGB"))
    axes = detect_axes(img)
    x_grid = np.linspace(0.0, 50.0, 501)

    def blue_mask(plot: np.ndarray) -> np.ndarray:
        r, g, b = plot[:, :, 0], plot[:, :, 1], plot[:, :, 2]
        return (b > 100) & (r < 70) & (g < 70)

    def yellow_mask(plot: np.ndarray) -> np.ndarray:
        r, g, b = plot[:, :, 0], plot[:, :, 1], plot[:, :, 2]
        return (r > 220) & (g > 150) & (b < 80)

    chac = extract_curve(img, axes, blue_mask, x_grid)
    pvp = extract_curve(img, axes, yellow_mask, x_grid)
    with BASELINE_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_k", "chac_digitized", "pvp_digitized"])
        writer.writerows(zip(x_grid, chac, pvp))
    return x_grid, chac, pvp


def scalar_value(summary_value):
    if summary_value.HasField("tensor"):
        tensor = summary_value.tensor
        for field in ("float_val", "double_val", "int_val", "int64_val"):
            values = getattr(tensor, field)
            if values:
                return float(values[0])
    return float(summary_value.simple_value)


def build_ours_csv() -> None:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    tags = {"episode/takeover_rate", "episode/length"}
    by_step = defaultdict(dict)
    for i, event in enumerate(EventFileLoader(str(EVENT_FILE)).Load(), start=1):
        for value in event.summary.value:
            if value.tag in tags:
                by_step[int(event.step)][value.tag] = scalar_value(value)
        if i % 1_000_000 == 0:
            print(f"parsed_events={i}", flush=True)

    rows = []
    cum_takeover = 0.0
    cum_total = 0.0
    for step, item in sorted(by_step.items()):
        if "episode/takeover_rate" not in item or "episode/length" not in item:
            continue
        rate = item["episode/takeover_rate"]
        length = item["episode/length"]
        takeover_sec = rate * length
        cum_takeover += takeover_sec
        cum_total += length
        rows.append(
            [
                step,
                rate,
                length,
                takeover_sec,
                cum_takeover,
                cum_total,
                cum_takeover / cum_total if cum_total else 0.0,
            ]
        )
    if not rows:
        raise RuntimeError("no paired episode/takeover_rate and episode/length records found")
    with OURS_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "takeover_rate",
                "length",
                "takeover_sec",
                "cum_takeover_sec",
                "cum_total_sec",
                "cum_takeover_rate",
            ]
        )
        writer.writerows(rows)


def load_ours() -> tuple[np.ndarray, np.ndarray]:
    if not OURS_CSV.exists():
        build_ours_csv()
    steps, rates = [], []
    with OURS_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(float(row["step"]) / 1000.0)
            rates.append(float(row["takeover_rate"]))
    if not steps:
        raise RuntimeError(f"no rows found in {OURS_CSV}")
    return np.asarray(steps), np.clip(np.asarray(rates), 0.0, 1.1)


def smooth(y: np.ndarray, window: int = 25) -> np.ndarray:
    if y.size < 3:
        return y
    window = min(window, y.size if y.size % 2 else y.size - 1)
    window = max(window, 3)
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    x_base, chac, pvp = digitize_baselines()
    x_ours, ours = load_ours()
    ours_smoothed = smooth(ours)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 18,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.35), dpi=300)
    ax.plot(x_base, chac, color="#2626b8", linewidth=2.2, label="C-HAC (digitized)")
    ax.plot(x_base, pvp, color="#ffcc00", linewidth=2.2, label="PVP (digitized)")
    ax.plot(x_ours, ours, color="#2ca02c", linewidth=0.45, alpha=0.18, label="_nolegend_")
    ax.plot(x_ours, ours_smoothed, color="#16803a", linewidth=2.2, label="Ours")
    ax.set_title("Takeover Rate", pad=8)
    ax.set_xlabel("Training step (K)")
    ax.set_ylabel("Takeover rate")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.1)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xticklabels(["0", "10", "20", "30", "40", "50K"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG)

    print(f"wrote {BASELINE_CSV}")
    print(f"wrote {OURS_CSV}")
    print(f"wrote {OUT_PNG}")
    print(f"ours_final_episode_rate={ours[-1]:.6f}")
    print(f"ours_mean_episode_rate={ours.mean():.6f}")


if __name__ == "__main__":
    main()
