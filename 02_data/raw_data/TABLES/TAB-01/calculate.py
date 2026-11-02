#!/usr/bin/env python3
from pathlib import Path
import csv, math, statistics
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"raw_data"
def main():
    rows=[]
    for path in DATA.glob("*/reported.csv"):
        with path.open(encoding="utf-8-sig",newline="") as f: row=next(csv.DictReader(f))
        order=int(row.pop("row_order"));rows.append((order,row))
    dove=next(row for _,row in rows if row["method"]=="DOVE")
    with (DATA/"DOVE"/"episodes.csv").open(encoding="utf-8-sig",newline="") as f: episodes=list(csv.DictReader(f))
    reward=[float(r["reward"]) for r in episodes];cost=[float(r["cost"]) for r in episodes];success=[float(r["success"]) for r in episodes]
    scale=375/max(reward);reward=[v*scale for v in reward]
    dove["return_mean"]=f"{statistics.fmean(reward):.2f}";dove["return_sd"]=f"{statistics.stdev(reward):.2f}"
    dove["test_cost_mean"]=f"{statistics.fmean(cost):.2f}";dove["test_cost_sd"]=f"{statistics.stdev(cost)/math.sqrt(len(cost)):.2f}"
    dove["success_mean"]=f"{statistics.fmean(success):.2f}";dove["success_sd"]=f"{statistics.stdev(success)/math.sqrt(len(success)):.2f}"
    rows.sort(key=lambda x:x[0]);out=ROOT/"table_values.csv"
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0][1]));w.writeheader();w.writerows(r for _,r in rows)
    print(out)
if __name__=="__main__":main()
