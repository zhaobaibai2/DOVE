#!/usr/bin/env python3
"""Standalone aggregation for the four-component RQ2 table in DOVE_v3.tex."""
import argparse,csv,hashlib,json,math,shutil,statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CONFIGS=(('Positive-only','Positive-only',50,(.619,.615,.295,.789)),('ER_wo_CPCal','ER w/o CPCal',50,(.989,.981,.007,.345)),('CPCal_only','CPCal only',46,(.772,.722,.189,.987)),('Full_DOVE','Full DOVE',46,(.821,.772,.151,.891)))
KEYS=('rho_vlb','hsr','rmr_0p5','pairacc_c_mean')
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row));path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def summarize(rows):
 n=len(rows);s=sum(int(r['samples']) for r in rows)
 return {'rho_vlb':sum(int(r['rho_vlb_indicator']) for r in rows)/n,'hsr':sum(int(r['hsr_count_of_64']) for r in rows)/s,'rmr_0p5':sum(int(r['rmr_0p5_count_of_64']) for r in rows)/s,'pairacc_c_mean':sum(int(r['pairacc_c_indicator']) for r in rows)/n}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'raw_data');p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--terminal-csv',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 terminal_csv=a.terminal_csv
 if terminal_csv is None and out==ROOT and a.data_root.resolve()==(ROOT/'raw_data').resolve() and (ROOT/'active_run.json').exists():
  active=json.loads((ROOT/'active_run.json').read_text())
  if active.get('schema')=='rq2_vlb_hsr_rmr_v2':
   run=ROOT/active['run'];candidate=run/'terminal.csv'
   same_inputs=all((run/'raw_data'/folder/'pairs.csv').exists() and (a.data_root/folder/'pairs.csv').exists() and hashlib.sha256((run/'raw_data'/folder/'pairs.csv').read_bytes()).digest()==hashlib.sha256((a.data_root/folder/'pairs.csv').read_bytes()).digest() for folder,_,_,_ in CONFIGS)
   if candidate.exists() and same_inputs:terminal_csv=candidate
 final=[];values=[];details=[];comparisons=[];events=read(terminal_csv) if terminal_csv else [];sources=[]
 for folder,label,expected,reported in CONFIGS:
  path=a.data_root/folder/'pairs.csv'
  if not path.exists():raise FileNotFoundError('当前论文的新指标测量尚未生成，请先运行 bash scripts/run_tab02.sh test')
  rows=read(path);groups=defaultdict(list)
  for r in rows:
   groups[r['checkpoint']].append(r)
   if int(r['samples'])!=64 or not 0<=int(r['hsr_count_of_64'])<=64 or not 0<=int(r['rmr_0p5_count_of_64'])<=64:raise ValueError('采样计数非法')
   if int(r['rho_vlb_indicator']) not in (0,1) or int(r['pairacc_c_indicator']) not in (0,1):raise ValueError('配对指示量非法')
  if not a.smoke and (len(groups)!=expected or any(len(v)!=256 for v in groups.values())):raise ValueError(f'{label} checkpoint/样本数与论文配置不一致')
  for checkpoint,rr in sorted(groups.items()):
   if len({int(r['pair_id']) for r in rr})!=len(rr):raise ValueError('重复配对行')
   details.append({'setting':label,'checkpoint':checkpoint,'pairs':len(rr),'samples':sum(int(r['samples']) for r in rr),**summarize(rr)})
  stats=summarize(rows);row={'setting':label,'checkpoints':len(groups),'pair_checkpoint_rows':len(rows),'actor_samples':sum(int(r['samples']) for r in rows),**stats}
  for key in KEYS:
   vv=[r[key] for r in details if r['setting']==label];row[key+'_checkpoint_sd']=statistics.stdev(vv) if len(vv)>1 else 0.;row[key+'_checkpoint_se']=row[key+'_checkpoint_sd']/math.sqrt(len(vv))
   row[key+'_paper']=reported[KEYS.index(key)];row[key+'_matches_paper_3dp']=f'{stats[key]:.3f}'==f'{reported[KEYS.index(key)]:.3f}'
   note=''
   if not row[key+'_matches_paper_3dp']:
    note=('Paper value matches legacy 512-pair/52-or-48-checkpoint aggregation; current manuscript requires 256 pairs and 50-or-46 checkpoints.' if key=='pairacc_c_mean' else 'Exact frozen 64-draw/64-sample audit does not round to the manuscript cell; original 64-noise VLB measurements and regenerated actor counts are retained for verification.')
   comparisons.append({'setting':label,'metric':key,'recomputed':stats[key],'paper':reported[KEYS.index(key)],'difference':stats[key]-reported[KEYS.index(key)],'matches_paper_3dp':row[key+'_matches_paper_3dp'],'explanation':note})
  values.append(row);final.append({'Setting':label,'rho_VLB↑':stats['rho_vlb'],'HSR↑':stats['hsr'],'RMR_0.5↓':stats['rmr_0p5'],'PairAcc_C_mean↑':stats['pairacc_c_mean']})
  text=f"{label} rho_VLB={stats['rho_vlb']:.3f} HSR={stats['hsr']:.3f} RMR_0.5={stats['rmr_0p5']:.3f} PairAcc_C_mean={stats['pairacc_c_mean']:.3f}"
  print(text,flush=True);events.append({'record_type':'summary','setting':label,**stats,'terminal_text':text});sources.append({'file':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows)})
 write(out/'table_values.csv',values);write(out/'final_metrics.csv',final);write(out/'per_checkpoint.csv',details);write(out/'test_results.csv',events)
 write(out/'comparison.csv',comparisons)
 audit={'schema':'rq2_vlb_hsr_rmr_v2','paper_table':'DOVE_v3.tex tab:rq2_audit (Table III)','smoke':a.smoke,'inputs':sources,'definitions':{'rho_vlb':'pooled fraction delta_VLB > 0; 20 VP steps, 64 matched draws per step; Gaussian decoder beta1 then posterior variance','hsr':'pooled fraction of 64 actor samples with (a-midpoint).(accepted-rejected)/gap_squared > 0','rmr_0p5':'pooled fraction with ||a-rejected|| < 0.5*||accepted-rejected||','pairacc_c_mean':'pooled fraction (target_q1_neg+target_q2_neg)/2 > (target_q1_pos+target_q2_pos)/2 on the same 256 pairs'},'processing':'No rescaling or target adjustment; all rates derive from integer event counts. Checkpoint SD and SE are supplementary.','paper_comparison':[{k:v for k,v in r.items() if k=='setting' or k.endswith('_matches_paper_3dp')} for r in values]}
 (out/'calculation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 if out==ROOT:shutil.copyfile(out/'test_results.csv',out/('test_results_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv'))
if __name__=='__main__':main()
