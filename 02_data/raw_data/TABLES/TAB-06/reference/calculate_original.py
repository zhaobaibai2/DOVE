#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv, math, statistics
ROOT=Path(__file__).resolve().parent
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
def main():
 result=[]
 for method in METHODS:
  folder=method.replace(' ','_');root=ROOT/'raw_data'/folder
  with (root/'runs.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  groups=defaultdict(list)
  for r in rows:groups[r['run']].append(r)
  runmetrics=[]
  for run,rr in sorted(groups.items()):
   meta=rr[0];m=metric(meta,rr);m.update({'paper_method':method,'source_method':meta['source_method'],'scenario':meta['scenario'],'run':run});runmetrics.append(m)
  with (root/'reported.csv').open(encoding='utf-8-sig',newline='') as f:paper=next(csv.DictReader(f))
  fields=('avg_speed','lateral_mae','mean_abs_jerk')
  vals={}
  for metric_name,key in [('speed','avg_speed'),('lateral_mae','lateral_mae'),('jerk','mean_abs_jerk')]:
   vs=[float(r[key]) for r in runmetrics];vals[metric_name+'_mean']=mean(vs);vals[metric_name+'_sd']=statistics.stdev(vs)
  row={'paper_method':method,'source_runs':len(runmetrics),'speed_mean':vals['speed_mean'],'speed_sd':vals['speed_sd'],'lateral_mae_mean':vals['lateral_mae_mean'],'lateral_mae_sd':vals['lateral_mae_sd'],'jerk_mean':vals['jerk_mean'],'jerk_sd':vals['jerk_sd'],'successes':paper['successes'],'trials':paper['trials']}
  for metric_name in ('speed_mean','speed_sd','lateral_mae_mean','lateral_mae_sd','jerk_mean','jerk_sd'):
   paper_value=float(paper[metric_name]);row[metric_name+'_paper']=paper_value;row[metric_name+'_match_3dp']=f'{row[metric_name]:.3f}'==f'{paper_value:.3f}'
  result.append(row)
 path=ROOT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
 print(path)
if __name__=='__main__':main()
