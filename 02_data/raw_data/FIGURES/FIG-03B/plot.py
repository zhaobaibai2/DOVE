#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dove_fig03b_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
METHODS=("H-DSAC","PVP","DOVE")
COLORS={"H-DSAC":"#303098","PVP":"#e3b600","DOVE":"#006400"}
for name in ("Times_New_Roman.ttf","Times_New_Roman_Bold.ttf","Times_New_Roman_Italic.ttf","Times_New_Roman_Bold_Italic.ttf"):
    p=Path("/usr/share/fonts/truetype/msttcorefonts")/name
    if p.exists(): font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"mathtext.fontset":"stix","axes.unicode_minus":False,"pdf.fonttype":42,"path.simplify":False})
def main():
    rows_out=[]
    grid=np.arange(0,50001,500,dtype=float)
    for method in METHODS:
        runs=defaultdict(list)
        for path in sorted((ROOT/"raw_data"/method).glob("*.csv")):
            with path.open(encoding="utf-8-sig",newline="") as f:
                for row in csv.DictReader(f):
                    if row["Method"]!=method: raise ValueError(f"Unexpected method in {path}")
                    x=float(row["Step"])
                    if x<=50000: runs[row["Run"]].append((x,float(row["Value"])))
        if len(runs)<2: raise ValueError(f"Need multiple runs for {method}")
        aligned=[]
        for run,points in sorted(runs.items()):
            points.sort()
            x,y=np.asarray(points,dtype=float).T
            if len(np.unique(x))!=len(x): raise ValueError(f"Duplicate steps: {method}/{run}")
            aligned.append(np.interp(grid,x,np.clip(y,0,1)))
        aligned=np.asarray(aligned)
        means=aligned.mean(axis=0); sds=aligned.std(axis=0,ddof=1)
        rows_out.extend({"Step":x,"Mean":m,"Std":s,"N":len(aligned),"Method":method,"Step_K":x/1000}
                        for x,m,s in zip(grid,means,sds))
    data=ROOT/"sim_takeover.csv"
    with data.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["Step","Mean","Std","N","Method","Step_K"]);w.writeheader();w.writerows(rows_out)
    rows=list(csv.DictReader(data.open(encoding="utf-8-sig",newline="")))
    fig=plt.figure(figsize=(3.35,2.65))
    ax=fig.add_axes([41.004/241.2,(190.8-158.364)/190.8,(236.376-41.004)/241.2,(158.364-5.724)/190.8])
    widths={"H-DSAC":1.25,"PVP":1.15,"DOVE":1.45}
    for method in METHODS:
        series=[r for r in rows if r["Method"]==method]
        x=np.array([float(r["Step"]) for r in series]); y=np.array([float(r["Mean"]) for r in series]); sd=np.array([float(r["Std"]) for r in series])
        ax.plot(x,y,color=COLORS[method],linewidth=widths[method],label=method)
        ax.fill_between(x,np.clip(y-sd,0,1),np.clip(y+sd,0,1),color=COLORS[method],alpha=.16,linewidth=0)
    ax.set(xlim=(0,50000),ylim=(0,1),xlabel="Environment steps",ylabel="Takeover rate")
    ax.set_xticks([0,10000,20000,30000,40000,50000],["0","10K","20K","30K","40K","50K"]);ax.get_xticklabels()[-1].set_horizontalalignment("right")
    ax.set_yticks([0,.5,1]);ax.xaxis.label.set_size(7.2);ax.yaxis.label.set_size(7.2);ax.xaxis.labelpad=ax.yaxis.labelpad=3
    ax.tick_params(labelsize=6.6,length=2.5,width=.55,pad=1)
    for spine in ax.spines.values():spine.set_linewidth(.55)
    ax.set_axisbelow(True);ax.grid(True,linestyle="--",color="#d9d9d9",linewidth=.45,alpha=.75)
    ax.legend(fontsize=6.4,frameon=True,loc="upper right",borderpad=.25,labelspacing=.25,handlelength=1.8)
    fig.savefig(ROOT/"fig_sim_takeover_hdsac_pvp_dove.pdf");fig.savefig(ROOT/"fig_sim_takeover_hdsac_pvp_dove.png",dpi=300);plt.close(fig)
    print(data);print(ROOT/"fig_sim_takeover_hdsac_pvp_dove.pdf")
if __name__=="__main__":main()
