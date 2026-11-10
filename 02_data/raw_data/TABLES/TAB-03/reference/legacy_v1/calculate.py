#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime
import csv, math, statistics
ROOT=Path(__file__).resolve().parent

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,default=ROOT/'raw_data')
parser.add_argument('--output-dir',type=Path,default=ROOT)
parser.add_argument('--terminal-csv',type=Path)
parser.add_argument('--smoke',action='store_true')
OPTIONS=parser.parse_args();DATA_ROOT=OPTIONS.data_root.resolve();OUTPUT=OPTIONS.output_dir.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)

CONFIGS=(('A','Positive-only (DBC + critic)','positive_only',('0.442','0.362','0.887','324.99')),('A','ER w/o CPCal','ER_wo_CPCal',('0.680','0.320','0.852','315.25')),('A','Full DOVE','Full_DOVE',('0.940','0.060','0.966','363.36')),('B','Repaired actor without test-time guidance','Vanilla',('0.400','0.400','0.768','288.53')),('B','Top-K rerank (K=8)','TopK_8',('0.388','0.500','0.703','261.45')),('B','Projected guidance, always execute','No_gate',('0.472','0.490','0.658','244.14')),('B','Projected guidance with acceptance/fallback gate','Full_gate',('0.842','0.158','0.883','330.76')))
METRICS=('success','crash','route_completion','reward')
DETAIL_NAME='per_checkpoint.csv'
EVIDENCE_NOTE='Local episode records; run.sh test loads the packaged models and generates fresh episodes.'
PROCESSING_NOTE='Pool episodes within each setting; retain sample SD and SE. Every stored reward already includes its fixed historical conversion exactly once. Preserve original checkpoint membership and 500/600/400 panel A counts.'

# Self-contained: only the Python standard library and this table's local inputs.
def save_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def publish(values,details,lines):
 save_csv(OUTPUT/'final_metrics.csv',values)
 if details:save_csv(OUTPUT/DETAIL_NAME,details)
 events=[]
 if OPTIONS.terminal_csv:
  with Path(OPTIONS.terminal_csv).open(encoding='utf-8-sig',newline='') as f:events=list(csv.DictReader(f))
 for row,line in zip(values,lines):
  print(line,flush=True);events.append({'record_type':'summary',**row,'terminal_text':line})
 save_csv(OUTPUT/'test_results.csv',events)
 if OUTPUT==ROOT:
  import shutil
  shutil.copyfile(OUTPUT/'test_results.csv',OUTPUT/('test_results_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv'))
 inputs=[{'file':str(p.relative_to(DATA_ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DATA_ROOT.rglob('*.csv'))]
 (OUTPUT/'calculation_audit.json').write_text(json.dumps({'table':ROOT.name,'smoke':OPTIONS.smoke,'inputs':inputs,'metrics_source':EVIDENCE_NOTE,'processing':PROCESSING_NOTE},ensure_ascii=False,indent=2)+'\n')

def main():
 out=[];details=[]
 for panel,label,folder,reported in CONFIGS:
  with (DATA_ROOT/folder/'episodes.csv').open(encoding='utf-8-sig',newline='') as f:episodes=list(csv.DictReader(f))
  for checkpoint in sorted({r['checkpoint'] for r in episodes}):
   rr=[r for r in episodes if r['checkpoint']==checkpoint];details.append({'setting':label,'checkpoint':checkpoint,'episodes':len(rr),**{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS}})
  checkpoints=sorted({r['checkpoint'] for r in episodes});row={'panel':panel,'paper_setting':label,'n_checkpoints':len(checkpoints),'n_episodes':len(episodes),'checkpoint_membership':';'.join(checkpoints)};rounded=[]
  for metric in METRICS:
   values=[float(r[metric]) for r in episodes];m=statistics.fmean(values);sd=statistics.stdev(values) if len(values)>1 else 0.;se=sd/math.sqrt(len(values)) if len(values)>1 else 0.
   name='collision' if metric=='crash' else metric;row[name]=f'{m:.12g}';row[name+'_sample_sd']=f'{sd:.12g}';row[name+'_se']=f'{se:.12g}';rounded.append(f'{m:.{2 if metric=="reward" else 3}f}')
  row['reported_values']='/'.join(reported);row['rounded_values']='/'.join(rounded);row['matches_final_tex']=str(tuple(rounded)==reported).lower();row['matches_caption_500_episode_claim']=str(len(episodes)==500).lower();out.append(row)
 path=OUTPUT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 values=[{'Panel':r['panel'],'Setting':r['paper_setting'],'Episodes':r['n_episodes'],'Succ.↑':float(r['success']),'Crash↓':float(r['collision']),'Route Completion↑':float(r['route_completion']),'Return↑':float(r['reward'])} for r in out]
 lines=[f"{r['Setting']} arrive={r['Succ.↑']:.1%} crash={r['Crash↓']:.1%} route_completion={r['Route Completion↑']:.3f} Return↑={r['Return↑']:.2f}" for r in values]
 publish(values,details,lines)
if __name__=='__main__':main()
