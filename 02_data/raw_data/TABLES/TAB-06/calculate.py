#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
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

METHODS=('Positive-only','DOVE','IWR','PVP-style_TD3')
def number(v):
 try:x=float(v) if v not in ('',None) else math.nan;return x if math.isfinite(x) else None
 except (TypeError,ValueError):return None
def mean(v):return statistics.fmean(v)
def metric(meta,rows):
 ts=[];speed=[];lat=[];yaw=[];pairs=[];samples=0
 for r in rows:
  if meta['input_format']=='winter_eval_log':t=number(r['ts']);s=number(r['carspeed']);l=number(r['error_distance']);y=number(r['error_yaw'])
  else:
   t=number(r['actual_timestamp']);s=number(r['actual_speed_mps']);l=number(r['path_lateral_error_m']);h=number(r['path_heading_error_deg']);y=h*math.pi/180 if h is not None else None
   if t is None or s is None or l is None:continue
  if t is not None:ts.append(t)
  if s is not None:speed.append(s)
  if l is not None:lat.append(l)
  if y is not None:yaw.append(y)
  if t is not None and s is not None:pairs.append((t,s))
  if t is not None and s is not None and l is not None:samples+=1
 unique={}
 for t,s in sorted(pairs,key=lambda v:v[0]):unique.setdefault(t,s)
 ordered=sorted(unique.items());acc=[]
 for i in range(1,len(ordered)):
  dt=ordered[i][0]-ordered[i-1][0]
  if dt>0:acc.append((dt,(ordered[i][1]-ordered[i-1][1])/dt))
 jerks=[]
 for i in range(1,len(acc)):
  v=(acc[i][1]-acc[i-1][1])/acc[i][0]
  if math.isfinite(v):jerks.append(v)
 return {'samples':samples,'duration_s':max(ts)-min(ts),'avg_speed':mean(speed),'max_speed':max(speed),'lateral_mae':mean([abs(x) for x in lat]),'lateral_rmse':math.sqrt(mean([x*x for x in lat])),'yaw_mae':mean([abs(x) for x in yaw]),'mean_abs_jerk':mean([abs(x) for x in jerks]),'rms_jerk':math.sqrt(mean([x*x for x in jerks]))}
DETAIL_NAME='per_run.csv'
EVIDENCE_NOTE='Full local physical driving traces; reported success counts remain separately sourced.'
PROCESSING_NOTE='Compute speed, absolute lateral error and absolute jerk separately per recorded run, then unweighted run mean and sample SD. Sort timestamp-speed pairs and keep first duplicate timestamp. Success counts come from reported.csv, not inferred from traces. Positive-only source is labelled DACER; the alias is unverified.'

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
 result=[];details=[]
 for method in METHODS:
  folder=method.replace(' ','_');root=DATA_ROOT/folder
  with (root/'runs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  groups=defaultdict(list)
  for r in rows:groups[r['run']].append(r)
  runmetrics=[]
  for run,rr in sorted(groups.items()):
   meta=rr[0];m=metric(meta,rr);m.update({'paper_method':method,'source_method':meta['source_method'],'scenario':meta['scenario'],'run':run});runmetrics.append(m)
  details.extend(runmetrics)
  with (root/'reported.csv').open(encoding='utf-8-sig',newline='') as f:paper=next(csv.DictReader(f))
  fields=('avg_speed','lateral_mae','mean_abs_jerk')
  vals={}
  for metric_name,key in [('speed','avg_speed'),('lateral_mae','lateral_mae'),('jerk','mean_abs_jerk')]:
   vs=[float(r[key]) for r in runmetrics];vals[metric_name+'_mean']=mean(vs);vals[metric_name+'_sd']=statistics.stdev(vs)
  row={'paper_method':method,'source_runs':len(runmetrics),'speed_mean':vals['speed_mean'],'speed_sd':vals['speed_sd'],'lateral_mae_mean':vals['lateral_mae_mean'],'lateral_mae_sd':vals['lateral_mae_sd'],'jerk_mean':vals['jerk_mean'],'jerk_sd':vals['jerk_sd'],'successes':paper['successes'],'trials':paper['trials']}
  for metric_name in ('speed_mean','speed_sd','lateral_mae_mean','lateral_mae_sd','jerk_mean','jerk_sd'):
   paper_value=float(paper[metric_name]);row[metric_name+'_paper']=paper_value;row[metric_name+'_match_3dp']=f'{row[metric_name]:.3f}'==f'{paper_value:.3f}'
  result.append(row)
 path=OUTPUT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 values=[{'Method':r['paper_method'],'Runs':r['source_runs'],'Speed mean':r['speed_mean'],'Speed SD':r['speed_sd'],'Lateral MAE mean':r['lateral_mae_mean'],'Lateral MAE SD':r['lateral_mae_sd'],'Jerk mean':r['jerk_mean'],'Jerk SD':r['jerk_sd'],'Successes':int(r['successes']),'Trials':int(r['trials'])} for r in result]
 lines=[f"{r['Method']} speed={r['Speed mean']:.3f} ± {r['Speed SD']:.3f} lateral_MAE={r['Lateral MAE mean']:.3f} ± {r['Lateral MAE SD']:.3f} jerk={r['Jerk mean']:.3f} ± {r['Jerk SD']:.3f} success={r['Successes']}/{r['Trials']}" for r in values]
 publish(values,details,lines)
if __name__=='__main__':main()
