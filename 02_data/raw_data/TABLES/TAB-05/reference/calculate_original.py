#!/usr/bin/env python3
from pathlib import Path
import csv
ROOT=Path(__file__).resolve().parent
SCENARIOS=('curve_meeting','pedestrian_yielding','obstacle_avoidance','straight_tracking')
def main():
 rows=[]
 for path in sorted((ROOT/'raw_data').glob('*/reported_counts.csv')):
  with path.open(encoding='utf-8-sig',newline='') as f:r=next(csv.DictReader(f))
  total=sum(int(r[k]) for k in SCENARIOS)
  rows.append({'paper_method':r['paper_method'],**{k:int(r[k]) for k in SCENARIOS},'total_successes':int(r['total_successes']),'trials_per_scenario':int(r['trials_per_scenario']),'total_trials':int(r['total_trials']),'arithmetic_match':total==int(r['total_successes']),'observed_trial_rows':0,'trial_level_recomputation':'MISSING'})
 path=ROOT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(path)
if __name__=='__main__':main()
