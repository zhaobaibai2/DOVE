#!/usr/bin/env python3
from pathlib import Path
import csv, math, statistics
ROOT=Path(__file__).resolve().parent
CONFIGS=(('A','Positive-only (DBC + critic)','positive_only',('0.442','0.362','0.887','324.99')),('A','ER w/o CPCal','ER_wo_CPCal',('0.680','0.320','0.852','315.25')),('A','Full DOVE','Full_DOVE',('0.940','0.060','0.966','363.36')),('B','Repaired actor without test-time guidance','Vanilla',('0.400','0.400','0.768','288.53')),('B','Top-K rerank (K=8)','TopK_8',('0.388','0.500','0.703','261.45')),('B','Projected guidance, always execute','No_gate',('0.472','0.490','0.658','244.14')),('B','Projected guidance with acceptance/fallback gate','Full_gate',('0.842','0.158','0.883','330.76')))
METRICS=('success','crash','route_completion','reward')
def main():
 out=[]
 for panel,label,folder,reported in CONFIGS:
  with (ROOT/'raw_data'/folder/'episodes.csv').open(encoding='utf-8-sig',newline='') as f:episodes=list(csv.DictReader(f))
  checkpoints=sorted({r['checkpoint'] for r in episodes});row={'panel':panel,'paper_setting':label,'n_checkpoints':len(checkpoints),'n_episodes':len(episodes),'checkpoint_membership':';'.join(checkpoints)};rounded=[]
  for metric in METRICS:
   values=[float(r[metric]) for r in episodes];m=statistics.fmean(values);sd=statistics.stdev(values) if len(values)>1 else 0.;se=sd/math.sqrt(len(values)) if len(values)>1 else 0.
   name='collision' if metric=='crash' else metric;row[name]=f'{m:.12g}';row[name+'_sample_sd']=f'{sd:.12g}';row[name+'_se']=f'{se:.12g}';rounded.append(f'{m:.{2 if metric=="reward" else 3}f}')
  row['reported_values']='/'.join(reported);row['rounded_values']='/'.join(rounded);row['matches_final_tex']=str(tuple(rounded)==reported).lower();row['matches_caption_500_episode_claim']=str(len(episodes)==500).lower();out.append(row)
 path=ROOT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(path)
if __name__=='__main__':main()
