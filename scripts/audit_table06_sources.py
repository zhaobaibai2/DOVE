"""Independent recomputation from original vehicle CSVs, without table calculator imports."""
import argparse,csv,hashlib,importlib.util,json,math,statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];TABLE=ROOT/'02_data/raw_data/TABLES/TAB-06'
METRICS=('avg_speed','lateral_mae','mean_abs_jerk')
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def number(x):
 try:v=float(x);return v if math.isfinite(v) else None
 except (ValueError,TypeError):return None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def independent(rows,kind):
 fields=('ts','carspeed','error_distance','error_yaw') if kind=='winter_eval_log' else ('actual_timestamp','actual_speed_mps','path_lateral_error_m','path_heading_error_deg')
 values=np.array([[number(r.get(k)) if number(r.get(k)) is not None else np.nan for k in fields] for r in rows],dtype=np.float64)
 if kind!='winter_eval_log':values=values[np.isfinite(values[:,:3]).all(axis=1)];values[:,3]*=math.pi/180.
 t,v,lat,yaw=values.T
 usable=np.isfinite(t)&np.isfinite(v);tv=values[usable,:2];tv=tv[np.argsort(tv[:,0],kind='stable')];_,first=np.unique(tv[:,0],return_index=True);tv=tv[first]
 dt=np.diff(tv[:,0]);assert np.all(dt>0);acceleration=np.diff(tv[:,1])/dt;jerk=np.diff(acceleration)/dt[1:];jerk=jerk[np.isfinite(jerk)]
 speed=v[np.isfinite(v)];error=np.abs(lat[np.isfinite(lat)])
 return {'samples':int(np.isfinite(values[:,:3]).all(axis=1).sum()),'avg_speed':float(speed.mean()),'lateral_mae':float(error.mean()),'mean_abs_jerk':float(np.abs(jerk).mean()),'duplicate_timestamps':int(usable.sum()-len(tv)),'jerk_samples':len(jerk)},speed,error,np.abs(jerk),fields

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=TABLE/'source_audit');args=parser.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 manifests=read(TABLE/'provenance/source_manifest.csv');local={m:read(TABLE/'raw_data'/m/'runs.csv') for m in {r['paper_method'] for r in manifests}}
 existing={(r['paper_method'],r['run']):r for r in read(TABLE/'per_run.csv')}
 runs=[];checks=[];samples=defaultdict(lambda:[[],[],[]])
 original_script=Path('/home/aaa/data/real_car/寒假/results_0212/results/summarize_metrics.py');spec=importlib.util.spec_from_file_location('original_vehicle_metrics',original_script);original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
 for item in manifests:
  method=item['paper_method'];run=item['run'];path=Path(item['source_path']);rows=read(path);current=[r for r in local[method] if r['run']==run]
  assert len(rows)==len(current)==int(item['raw_rows'])
  actual_hash=sha(path);assert actual_hash==item['source_sha256'],str(path)
  result,v,e,j,fields=independent(rows,item['input_format']);errors={key:0. for key in fields};missing=0
  for a,b in zip(rows,current):
   for key in fields:
    x=number(a.get(key));y=number(b.get(key))
    if (x is None)!=(y is None):missing+=1
    elif x is not None:errors[key]=max(errors[key],abs(x-y))
  assert missing==0 and max(errors.values())<1e-12,(run,errors,missing)
  previous=original.eval_log_metrics(path) if item['input_format']=='winter_eval_log' else original.dove_route1_metrics(path)
  original_errors={key:result[key]-previous[key] for key in METRICS}
  # CSV floating-point parsers differ slightly in timestamp rounding. The
  # derivative amplification remains well below the displayed 0.001 precision.
  assert all(abs(x)<5e-5 for x in original_errors.values()),(run,original_errors)
  calc_errors={key:result[key]-float(existing[(method,run)][key]) for key in METRICS};assert all(abs(x)<1e-11 for x in calc_errors.values())
  runs.append({'method':method,'source_method':item['source_method'],'run':run,'raw_rows':len(rows),**result,'source':str(path),'sha256':actual_hash})
  checks.append({'method':method,'run':run,'source_hash_matches':True,'projected_rows':len(current),'projection_max_abs_error':max(errors.values()),'current_calculator_max_abs_error':max(abs(x) for x in calc_errors.values()),'original_script_max_abs_error':max(abs(x) for x in original_errors.values())})
  for target,values in zip(samples[method],(v,e,j)):target.extend(values.tolist())
 summary=[];pooled=[]
 for method in ('Positive-only','DOVE','IWR','PVP-style_TD3'):
  group=[r for r in runs if r['method']==method];paper=read(TABLE/'raw_data'/method/'reported.csv')[0];row={'method':method,'runs':len(group),'raw_rows':sum(r['raw_rows'] for r in group),'valid_frames':sum(r['samples'] for r in group)}
  for prefix,key in (('speed','avg_speed'),('lateral_mae','lateral_mae'),('jerk','mean_abs_jerk')):
   values=[r[key] for r in group];row[prefix+'_mean']=statistics.fmean(values);row[prefix+'_sd']=statistics.stdev(values)
  counts=read(TABLE.parent/'TAB-05/raw_data'/method/'reported_counts.csv')[0];scenario_keys=('curve_meeting','pedestrian_yielding','obstacle_avoidance','straight_tracking');total=sum(int(counts[k]) for k in scenario_keys)
  assert total==int(paper['successes']) and int(paper['trials'])==int(counts['total_trials'])
  row.update(successes=total,trials=int(paper['trials']),success_source='reported aggregate counts; no trial ledger',scenario_counts='/'.join(counts[k] for k in scenario_keys));summary.append(row)
  for key,values in zip(METRICS,samples[method]):pooled.append({'method':method,'metric':key,'samples':len(values),'frame_mean':statistics.fmean(values),'frame_sd':statistics.stdev(values)})
  print(f"{method} speed={row['speed_mean']:.3f} ± {row['speed_sd']:.3f} lateral_MAE={row['lateral_mae_mean']:.3f} ± {row['lateral_mae_sd']:.3f} jerk={row['jerk_mean']:.3f} ± {row['jerk_sd']:.3f} success={total}/{row['trials']}",flush=True)
 write(out/'per_run_from_original.csv',runs);write(out/'source_checks.csv',checks);write(out/'final_from_original.csv',summary);write(out/'pooled_frame_alternative.csv',pooled)
 report={'created':datetime.now().isoformat(),'original_files':len(manifests),'original_rows':sum(r['raw_rows'] for r in runs),'source_hashes_all_match':True,'projected_values_all_match':True,'arithmetic':'Per-run metric then equal-weight run mean and sample SD (ddof=1); jerk via diff(diff(speed)/dt)/dt[1:] after timestamp deduplication.','original_processing_script':str(original_script),'original_processing_script_sha256':sha(original_script),'result':'All current printed continuous metrics independently reproduced to display precision from the original 12 raw CSVs.','limitations':['Success counts are reported aggregates, not recomputed from 40 individual trials.','Positive-only source runs identify themselves as DACER; a positive-only-only objective/matched training-stream mapping is not established.','DOVE source is two Route1 control runs, not a recovered 40-trial/four-scenario ledger.','Current paper says mean/SD over valid frames; historical script and current printed output aggregate run means instead. pooled_frame_alternative.csv shows the different frame-level results.','IWR speed is 1.744121259... from source; the current paper prints 1.24. No source processing reproducing 1.24 has been established.']}
 (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print('Source audit saved:',out)
if __name__=='__main__':main()
