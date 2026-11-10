#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime
import csv
ROOT=Path(__file__).resolve().parent

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,default=ROOT/'raw_data')
parser.add_argument('--output-dir',type=Path,default=ROOT)
parser.add_argument('--terminal-csv',type=Path)
parser.add_argument('--smoke',action='store_true')
OPTIONS=parser.parse_args();DATA_ROOT=OPTIONS.data_root.resolve();OUTPUT=OPTIONS.output_dir.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)

SCENARIOS=('curve_meeting','pedestrian_yielding','obstacle_avoidance','straight_tracking')
DETAIL_NAME='scenario_counts.csv'
EVIDENCE_NOTE='Only reported aggregate counts exist; no individual trial ledger was recovered.'
PROCESSING_NOTE='Sum the four original scenario success counts; divide by original total trials. No synthetic per-trial rows are generated.'

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
 rows=[]
 for path in sorted((DATA_ROOT).glob('*/reported_counts.csv')):
  with path.open(encoding='utf-8-sig',newline='') as f:r=next(csv.DictReader(f))
  total=sum(int(r[k]) for k in SCENARIOS)
  if total!=int(r['total_successes']) or int(r['total_trials'])!=4*int(r['trials_per_scenario']):raise ValueError('场景计数不一致')
  rows.append({'paper_method':r['paper_method'],**{k:int(r[k]) for k in SCENARIOS},'total_successes':int(r['total_successes']),'trials_per_scenario':int(r['trials_per_scenario']),'total_trials':int(r['total_trials']),'arithmetic_match':total==int(r['total_successes']),'observed_trial_rows':0,'trial_level_recomputation':'MISSING'})
 path=OUTPUT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 values=[{'Method':r['paper_method'],**{k:r[k] for k in SCENARIOS},'Successes':r['total_successes'],'Trials':r['total_trials'],'Succ.↑':r['total_successes']/r['total_trials'],'evidence':'reported scenario counts; individual trial records unavailable'} for r in rows]
 lines=[f"{r['Method']} curve={r['curve_meeting']}/10 pedestrian={r['pedestrian_yielding']}/10 obstacle={r['obstacle_avoidance']}/10 straight={r['straight_tracking']}/10 success={r['Successes']}/{r['Trials']} ({r['Succ.↑']:.1%})" for r in values]
 print('场景汇总计数重算（缺少逐次试验记录）',flush=True)
 publish(values,rows,lines)
if __name__=='__main__':main()
