#!/usr/bin/env python3
from pathlib import Path
import csv
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"dove_fig05b_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
WIDTH,HEIGHT=124.724411011,82.771652222
LEFT,TOP,RIGHT,BOTTOM=29.310235977,16.554328918,117.240943909,61.664878845
COLORS={"Positive-only DBP":"#8f969e","DOVE":"#2f6b4f"}
def main():
    rows=[]
    for folder in sorted((ROOT/"raw_data").iterdir()):
        for p in folder.glob("*.csv"):
            with p.open(encoding="utf-8-sig",newline="") as f:rows.extend(csv.DictReader(f))
    with (ROOT/"real_takeover.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72))
    ax=fig.add_axes([LEFT/WIDTH,(HEIGHT-BOTTOM)/HEIGHT,(RIGHT-LEFT)/WIDTH,(BOTTOM-TOP)/HEIGHT])
    lines={}
    for method in ("Positive-only DBP","DOVE"):
        series=sorted((r for r in rows if r["method"]==method),key=lambda r:float(r["aligned_training_progress_pct"]))
        x=[float(r["aligned_training_progress_pct"]) for r in series];y=[float(r["takeover_rate_pct"]) for r in series]
        lines[method],=ax.plot(x,y,linewidth=1.05,color=COLORS[method]);ax.plot(x[-1],y[-1],marker="o",markersize=1.8,color=COLORS[method])
        lx,ly=(99.335021973,56.408905029) if method=="DOVE" else (96.522521973,48.045322418)
        fig.text(lx/WIDTH,(HEIGHT-ly)/HEIGHT,"0.3%" if method=="DOVE" else "20.1%",ha="left",va="baseline",fontsize=5.6,color=COLORS[method])
    ax.set(xlim=(0,105),ylim=(-2,105),xlabel="Aligned training progress (%)",ylabel="Takeover rate (%)")
    ax.set_xticks([0,20,40,60,80,100]);ax.set_yticks([0,20,40,60,80,100]);ax.tick_params(labelsize=5.7,length=2.2,width=.58,pad=1.3)
    ax.xaxis.label.set_size(6.3);ax.yaxis.label.set_size(6.3);ax.xaxis.labelpad=ax.yaxis.labelpad=2.4
    for k in ("top","right"):ax.spines[k].set_visible(False)
    for k in ("left","bottom"):ax.spines[k].set_linewidth(.58)
    ax.set_axisbelow(True);ax.xaxis.grid(True,color="#eceff4",linewidth=.3);ax.yaxis.grid(True,color="#d8deea",linewidth=.38)
    ax.legend([lines["DOVE"],lines["Positive-only DBP"]],["DOVE","Positive-only"],loc="center",bbox_to_anchor=(.49,1.15),ncol=2,fontsize=5.7,frameon=False,handlelength=1.25,columnspacing=1.1,handletextpad=.5,borderaxespad=0)
    fig.savefig(ROOT/"fig_real_takeover_dove_vs_positive_dbp.pdf");fig.savefig(ROOT/"fig_real_takeover_dove_vs_positive_dbp.png",dpi=300);plt.close(fig)
    print(ROOT/"fig_real_takeover_dove_vs_positive_dbp.pdf")
if __name__=="__main__":main()
