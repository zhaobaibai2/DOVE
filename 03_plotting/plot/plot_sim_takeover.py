#!/usr/bin/env python3
"""Redraw this paper curve from final plot values; these are not raw measurements."""
from pathlib import Path
import csv
import os
import numpy as np

MODE = 'takeover'
STEM = 'fig_sim_takeover_hdsac_pvp_dove'
BUNDLE = Path(__file__).resolve().parents[2]
DATA = BUNDLE / "02_data/plot_data"
OUT = BUNDLE / "03_plotting/output"
CACHE = BUNDLE / "01_code/environment/mplconfig"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CACHE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ("Times_New_Roman.ttf", "Times_New_Roman_Bold.ttf", "Times_New_Roman_Italic.ttf", "Times_New_Roman_Bold_Italic.ttf"):
    path = Path("/usr/share/fonts/truetype/msttcorefonts") / name
    if path.exists():
        font_manager.fontManager.addfont(path)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.unicode_minus": False, "pdf.fonttype": 42, "path.simplify": False})


def main():
    filename = {"reward": "sim_reward.csv", "takeover": "sim_takeover.csv", "real_takeover": "real_takeover.csv"}[MODE]
    with (DATA / filename).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if MODE == "real_takeover":
        # Final PDF vertices, not original vehicle telemetry.
        width, height = 124.724411011, 82.771652222
        left, top, right, bottom = 29.310235977, 16.554328918, 117.240943909, 61.664878845
        fig = plt.figure(figsize=(width/72, height/72))
        ax = fig.add_axes([left/width, (height-bottom)/height, (right-left)/width, (bottom-top)/height])
        colors = {"Positive-only DBP": "#8f969e", "DOVE": "#2f6b4f"}
        lines = {}
        for method in ("Positive-only DBP", "DOVE"):
            series = [r for r in rows if r["method"] == method]
            x = [float(r["aligned_training_progress_pct"]) for r in series]
            y = [float(r["takeover_rate_pct"]) for r in series]
            lines[method], = ax.plot(x, y, linewidth=1.05, color=colors[method])
            ax.plot(x[-1], y[-1], marker="o", markersize=1.8, color=colors[method])
            label_x, label_y = (99.335021973,56.408905029) if method == "DOVE" else (96.522521973,48.045322418)
            fig.text(label_x/width,(height-label_y)/height,"0.3%" if method == "DOVE" else "20.1%",
                ha="left",va="baseline",fontsize=5.6,color=colors[method])
        ax.set(xlim=(0,105), ylim=(-2,105), xlabel="Aligned training progress (%)", ylabel="Takeover rate (%)")
        ax.set_xticks([0,20,40,60,80,100]); ax.set_yticks([0,20,40,60,80,100])
        ax.tick_params(labelsize=5.7, length=2.2, width=0.58, pad=1.3)
        ax.xaxis.label.set_size(6.3); ax.yaxis.label.set_size(6.3)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2.4
        for key in ("top","right"):
            ax.spines[key].set_visible(False)
        for key in ("left","bottom"):
            ax.spines[key].set_linewidth(0.58)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color="#eceff4", linewidth=0.3)
        ax.yaxis.grid(True, color="#d8deea", linewidth=0.38)
        ax.legend([lines["DOVE"],lines["Positive-only DBP"]],["DOVE","Positive-only"],
            loc="center", bbox_to_anchor=(0.49,1.15), ncol=2, fontsize=5.7, frameon=False,
            handlelength=1.25, columnspacing=1.1, handletextpad=0.5, borderaxespad=0)
    else:
        reward = MODE == "reward"
        methods = ["SAC","PPO","DSAC","H-DSAC","DOVE-ER"] if reward else ["H-DSAC","PVP","DOVE"]
        colors = {"SAC":"#c75a49","PPO":"#5a2753","DSAC":"#7fbd7b","H-DSAC":"#303098","DOVE-ER":"#006400","PVP":"#e3b600","DOVE":"#006400"}
        fig = plt.figure(figsize=(3.35,2.65))
        ax = fig.add_axes([41.004/241.2,(190.8-158.364)/190.8,(236.376-41.004)/241.2,(158.364-5.724)/190.8])
        for method in methods:
            series = sorted((r for r in rows if r["Method"] == method),key=lambda r:float(r["Step"]))
            if not series:
                raise ValueError("Missing series: "+method)
            x = np.array([float(r["Step"]) for r in series])
            y = np.array([float(r["Mean"]) for r in series])
            sd = np.array([float(r["Std"]) for r in series])
            lw = (1.4 if method=="DOVE-ER" else 1.1) if reward else {"H-DSAC":1.25,"PVP":1.15,"DOVE":1.45}[method]
            ax.plot(x,y,color=colors[method],linewidth=lw,label="DOVE" if method=="DOVE-ER" else method)
            low,high = (y-sd,y+sd) if reward else (np.clip(y-sd,0,1),np.clip(y+sd,0,1))
            ax.fill_between(x,low,high,color=colors[method],alpha=0.16,linewidth=0)
        ax.set(xlim=(0,1000000 if reward else 50000),ylim=(0,430 if reward else 1),
            xlabel="Environment steps",ylabel="Test episodic reward" if reward else "Takeover rate")
        ax.set_xticks([0,200000,400000,600000,800000,1000000] if reward else [0,10000,20000,30000,40000,50000],
            ["0","0.2M","0.4M","0.6M","0.8M","1.0M"] if reward else ["0","10K","20K","30K","40K","50K"])
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.set_yticks([0,100,200,300,400] if reward else [0.0,0.5,1.0])
        ax.xaxis.label.set_size(7.2); ax.yaxis.label.set_size(7.2)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 3.0
        ax.tick_params(labelsize=6.6,length=2.5,width=0.55,pad=1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
        ax.set_axisbelow(True)
        ax.grid(True,linestyle="--",color="#d9d9d9",linewidth=0.45,alpha=0.75)
        ax.legend(fontsize=6.4,frameon=True,loc="lower right" if reward else "upper right",borderpad=0.25,labelspacing=0.25,handlelength=1.8)
    OUT.mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/(STEM+".pdf"))
    fig.savefig(OUT/(STEM+".png"),dpi=300)
    plt.close(fig)
    print("Data plot:",OUT/(STEM+".pdf"))


if __name__ == "__main__":
    main()
