#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv, math
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'raw_data/DOVE'
def num(row,key):
 try:
  v=float(row[key]);return v if math.isfinite(v) else math.nan
 except (TypeError,ValueError,KeyError):return math.nan
def p95(values):
 d=sorted(x for x in values if math.isfinite(x));pos=(len(d)-1)*.95;lo=math.floor(pos);hi=math.ceil(pos)
 return d[lo] if lo==hi else d[lo]+(pos-lo)*(d[hi]-d[lo])
def main():
 phases=defaultdict(lambda:{'rows':0,'ts':[],'process':[],'loop':[],'gate':0,'accept':0.,'fallback':0.})
 with (DATA/'collection_repair.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   x=phases[r['phase_name']];x['rows']+=1;x['ts'].append(num(r,'ts'));x['process'].append(num(r,'process_time'));x['loop'].append(num(r,'loop_period'))
   if num(r,'gate_diagnostics_logged')==1:x['gate']+=1;x['accept']+=num(r,'gate_accepted');x['fallback']+=num(r,'gate_fallback')
 frozen={'rows':0,'ts':[],'process':[],'infer':[],'loop':[],'gate':0,'accept':0.,'fallback':0.}
 with (DATA/'frozen_execution.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   frozen['rows']+=1
   for key,col in [('ts','ts'),('process','process_time'),('infer','infer_time'),('loop','loop_period')]:frozen[key].append(num(r,col))
   if num(r,'gate_diagnostics_logged')==1:frozen['gate']+=1;frozen['accept']+=num(r,'gate_accepted');frozen['fallback']+=num(r,'gate_fallback')
 times=[];speeds=[]
 with (DATA/'connected_trace.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):times.append(num(r,'relative_time_s'));speeds.append(num(r,'actual_speed_mps'))
 def train_row(label,key):
  x=phases[key];g=x['gate'];return {'paper_row':label,'raw_rows_or_samples':x['rows'],'duration_s':max(x['ts'])-min(x['ts']),'gate_evaluations':g,'gate_accept_rate':x['accept']/g,'gate_fallback_rate':x['fallback']/g,'process_p95_ms':p95(x['process'])*1000,'infer_p95_ms':'','loop_p95_ms':p95(x['loop'])*1000,'mean_speed_mps':'','max_speed_mps':'','speed_p95_mps':'','source':'raw_data/DOVE/collection_repair.csv','selection':'phase_name='+key+'; gate rates use logged gate rows'}
 values=[train_row('Collection','Data Collection'),train_row('Repair adaptation','PVP Learning')]
 g=frozen['gate'];values.append({'paper_row':'Frozen execution','raw_rows_or_samples':frozen['rows'],'duration_s':max(frozen['ts'])-min(frozen['ts']),'gate_evaluations':g,'gate_accept_rate':frozen['accept']/g,'gate_fallback_rate':frozen['fallback']/g,'process_p95_ms':p95(frozen['process'])*1000,'infer_p95_ms':p95(frozen['infer'])*1000,'loop_p95_ms':p95(frozen['loop'])*1000,'mean_speed_mps':'','max_speed_mps':'','speed_p95_mps':'','source':'raw_data/DOVE/frozen_execution.csv','selection':'all frozen execution rows'})
 values.append({'paper_row':'Connected trace','raw_rows_or_samples':len(speeds),'duration_s':max(times)-min(times),'gate_evaluations':'','gate_accept_rate':'','gate_fallback_rate':'','process_p95_ms':'','infer_p95_ms':'','loop_p95_ms':'','mean_speed_mps':sum(speeds)/len(speeds),'max_speed_mps':max(speeds),'speed_p95_mps':p95(speeds),'source':'raw_data/DOVE/connected_trace.csv','selection':'all connected trace samples'})
 if [r['raw_rows_or_samples'] for r in values]!=[7514,22362,2629,8411]:raise ValueError('unexpected source row counts')
 out=ROOT/'table_values.csv'
 with out.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
 print(out)
if __name__=='__main__':main()
