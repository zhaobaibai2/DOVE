#!/usr/bin/env python3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

ROOT = Path("/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508")
EVENT_FILE = ROOT / "logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0/tb/events.out.tfevents.1779417685.aaa-MS-7E44.31580.0"
OUT_DIR = ROOT / "latex4/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT = OUT_DIR / "takeover_effort_trace.csv"
PDF_OUT = OUT_DIR / "takeover_effort_trace.pdf"

TAGS = {"episode/takeover_rate", "episode/length"}


def scalar_value(summary_value):
    if summary_value.HasField("tensor"):
        tensor = summary_value.tensor
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.int_val:
            return float(tensor.int_val[0])
        if tensor.int64_val:
            return float(tensor.int64_val[0])
    return float(summary_value.simple_value)


by_step = defaultdict(dict)
for i, event in enumerate(EventFileLoader(str(EVENT_FILE)).Load(), start=1):
    for value in event.summary.value:
        if value.tag in TAGS:
            by_step[int(event.step)][value.tag] = scalar_value(value)
    if i % 1000000 == 0:
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
    rows.append((step, rate, length, takeover_sec, cum_takeover, cum_total, cum_takeover / cum_total if cum_total else 0.0))

if not rows:
    raise SystemExit("no paired episode/takeover_rate and episode/length records found")

with CSV_OUT.open("w") as f:
    f.write("step,takeover_rate,length,takeover_sec,cum_takeover_sec,cum_total_sec,cum_takeover_rate\n")
    for row in rows:
        f.write(",".join(str(x) for x in row) + "\n")

steps = [r[0] / 1000.0 for r in rows]
episode_rates = [r[1] * 100.0 for r in rows]
cum_rates = [r[6] * 100.0 for r in rows]
overall = cum_rates[-1]

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})
fig, ax = plt.subplots(figsize=(3.45, 2.15))
ax.plot(steps, episode_rates, color="#8c8c8c", linewidth=0.8, alpha=0.55, label="Episode takeover rate")
ax.plot(steps, cum_rates, color="#1f77b4", linewidth=1.6, label="Cumulative takeover effort")
ax.axhline(overall, color="#d62728", linewidth=0.9, linestyle="--", label=f"Final {overall:.2f}%")
ax.set_xlabel("Training step (K)")
ax.set_ylabel("Takeover effort (%)")
ax.set_ylim(bottom=0)
ax.grid(True, linewidth=0.3, alpha=0.35)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(pad=0.4)
fig.savefig(PDF_OUT)
print(f"wrote {CSV_OUT}")
print(f"wrote {PDF_OUT}")
print(f"final_takeover_rate={overall:.6f}%")
print(f"takeover_sec={rows[-1][4]:.6f}")
print(f"total_sec={rows[-1][5]:.6f}")
