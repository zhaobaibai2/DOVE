#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dove_fig03a_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "raw_data"
METHODS = {"SAC": "SAC", "PPO": "PPO", "DSAC": "DSAC", "H-DSAC": "H-DSAC", "DOVE": "DOVE-ER"}
COLORS = {"SAC": "#c75a49", "PPO": "#5a2753", "DSAC": "#7fbd7b", "H-DSAC": "#303098", "DOVE": "#006400"}

for name in ("Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf", "Times_New_Roman_Italic.ttf", "Times_New_Roman_Bold_Italic.ttf"):
    font = Path("/usr/share/fonts/truetype/msttcorefonts") / name
    if font.exists():
        font_manager.fontManager.addfont(font)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.fonttype": 42, "path.simplify": False})


def calculate():
    curves = defaultdict(lambda: defaultdict(list))
    for folder, method in METHODS.items():
        files = sorted((DATA / folder).glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files in raw_data/{folder}")
        for run_index, path in enumerate(files, start=1):
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if folder == "DSAC":
                    if not {"Step", "Value"}.issubset(reader.fieldnames or []):
                        raise ValueError(f"Expected TensorBoard Step and Value columns in {path}")
                    points = sorted((float(row["Step"]), float(row["Value"])) for row in reader)
                    x, y = np.asarray(points, dtype=float).T
                    window = max(7, round(len(y) * 0.03))
                    if window % 2 == 0:
                        window -= 1
                    pad = window // 2
                    y = np.convolve(np.pad(y, pad, mode="edge"), np.ones(window) / window, mode="valid")
                    curves[folder][f"run{run_index}"].extend(zip(x, y))
                else:
                    for row in reader:
                        if row["Method"] != method:
                            raise ValueError(f"Unexpected method in {path}: {row['Method']}")
                        curves[folder][row["Run"]].append((float(row["Step"]), float(row["Value"])))

    output = []
    for folder, method in METHODS.items():
        runs = []
        for run, points in sorted(curves[folder].items()):
            points.sort()
            x, y = np.asarray(points, dtype=float).T
            if len(np.unique(x)) != len(x):
                raise ValueError(f"Duplicate steps in {folder}/{run}")
            runs.append((run, x, y))
        if len(runs) < 2:
            raise ValueError(f"Need at least two runs for {folder}")
        end = max(x[-1] for _, x, _ in runs)
        step = 1000 if end <= 50000 else 20000
        end = round(end / step) * step
        grid = np.arange(0, end + step, step, dtype=float)
        aligned = np.asarray([np.interp(grid, x, y) for _, x, y in runs])
        mean = aligned.mean(axis=0)
        sd = aligned.std(axis=0, ddof=1)
        output.extend((float(s), float(m), float(d), len(runs), method)
                      for s, m, d in zip(grid, mean, sd))

    output.sort(key=lambda row: (list(METHODS.values()).index(row[4]), row[0]))
    target = ROOT / "sim_reward.csv"
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Step", "Mean", "Std", "RunN", "Method"])
        writer.writerows(output)
    return target


def draw(data_file):
    with data_file.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    fig = plt.figure(figsize=(3.35, 2.65))
    ax = fig.add_axes([41.004/241.2, (190.8-158.364)/190.8, (236.376-41.004)/241.2, (158.364-5.724)/190.8])
    for method in METHODS.values():
        series = sorted((row for row in rows if row["Method"] == method), key=lambda row: float(row["Step"]))
        x = np.asarray([float(row["Step"]) for row in series])
        mean = np.asarray([float(row["Mean"]) for row in series])
        sd = np.asarray([float(row["Std"]) for row in series])
        label = "DOVE" if method == "DOVE-ER" else method
        ax.plot(x, mean, color=COLORS[label], linewidth=1.4 if label == "DOVE" else 1.1, label=label)
        ax.fill_between(x, mean-sd, mean+sd, color=COLORS[label], alpha=0.16, linewidth=0)
    ax.set(xlim=(0, 1000000), ylim=(0, 430), xlabel="Environment steps", ylabel="Test episodic reward")
    ax.set_xticks([0, 200000, 400000, 600000, 800000, 1000000], ["0", "0.2M", "0.4M", "0.6M", "0.8M", "1.0M"])
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.xaxis.label.set_size(7.2); ax.yaxis.label.set_size(7.2)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 3.0
    ax.tick_params(labelsize=6.6, length=2.5, width=0.55, pad=1)
    for spine in ax.spines.values(): spine.set_linewidth(0.55)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="--", color="#d9d9d9", linewidth=0.45, alpha=0.75)
    ax.legend(fontsize=6.4, frameon=True, loc="lower right", borderpad=0.25, labelspacing=0.25, handlelength=1.8)
    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(ROOT / f"fig_sim_reward_curve_hdsac_compare.{suffix}", **options)
    plt.close(fig)


if __name__ == "__main__":
    csv_file = calculate()
    draw(csv_file)
    print(f"Calculated data: {csv_file}")
    print(f"Figure: {ROOT / 'fig_sim_reward_curve_hdsac_compare.pdf'}")
