"""Run the original TAB-03 checkpoint memberships, then calculate the table."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from threading import Lock
ROOT=Path(__file__).resolve().parents[1]
TABLE=ROOT/'02_data/raw_data/TABLES/TAB-03'
def read(path):
 with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 if not 1<=a.workers<=6:p.error('--workers must be between 1 and 6')
 spec=json.loads((TABLE/'experiment.json').read_text());jobs=spec['jobs']
 run=TABLE/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 env=os.environ.copy();env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',DOVE_EPISODE_PROGRESS='1',DOVE_NO_TENSORBOARD='1',DOVE_CORE_METRICS_ONLY='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 env.pop('DOVE_TRAFFIC_SEED_BASE',None);env.pop('DOVE_EPISODE_START',None)
 lock=Lock();events=[];unique={};aliases={}
 for j in jobs:
  model=TABLE/j['model']
  if sha(model)!=j['model_sha256']:raise ValueError('模型校验失败')
  key=(j['model'],tuple(j['flags']),j['episodes'])
  if key not in unique:unique[key]=j
  aliases[(j['folder'],j['checkpoint'])]=key
 def evaluate(item):
  key,j=item;out=run/(j['folder']+'_'+j['checkpoint']);out.mkdir();n=1 if a.smoke else j['episodes'];tail=deque(maxlen=12)
  cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_pvp_policies_fixed_v3.py'),'--log_dir',str((TABLE/j['model']).parent),'--policy',Path(j['model']).name,'--maps','104','--horizon','1000','--num_episodes',str(n),'--output_csv',str(out/'summary.csv'),'--episode_output_csv',str(out/'episodes.csv'),*j['flags']]
  count=arrive=0
  with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as proc:
   for line in proc.stdout:
    tail.append(line.rstrip())
    if not line.startswith('DOVE_EPISODE '):continue
    ep=json.loads(line[len('DOVE_EPISODE '):]);count+=1;arrive+=ep['arrive'];other=int(ep['reason'] not in ('success','crash','out_of_route','timeout'))
    text=(f"{j['folder']} 第{j['checkpoint']}组 {count}/{n} arrive={ep['arrive']} crash={ep['crash']} out_of_route={ep['out_of_route']} timeout={ep['timeout']} other={other} success={arrive/count:.1%}")
    with lock:
     events.append({'record_type':'episode','setting':j['folder'],'checkpoint':j['checkpoint'],'episode':count,'arrive':ep['arrive'],'crash':ep['crash'],'out_of_route':ep['out_of_route'],'timeout':ep['timeout'],'other':other,'success':arrive/count,'reward':ep['reward']*j['reward_factor'],'route_completion':ep['route_completion'],'terminal_text':text})
     print(text,flush=True)
   if proc.wait():raise RuntimeError(f"{j['folder']} 测试失败："+'\n'.join(tail))
  rows=read(out/'episodes.csv')
  if len(rows)!=n or count!=n:raise ValueError('测试回合不完整，未发布结果')
  return key,rows
 with ThreadPoolExecutor(max_workers=a.workers) as pool:results=dict(pool.map(evaluate,unique.items()))
 write(run/'terminal.csv',events)
 published={}
 for j in jobs:
  rows=[]
  for ep in results[aliases[(j['folder'],j['checkpoint'])]]:
   rows.append({'checkpoint':j['checkpoint'],'success':ep['success'],'crash':ep['crash'],'route_completion':ep['route_completion'],'reward':float(ep['reward'])*j['reward_factor']})
  published.setdefault(j['folder'],[]).extend(rows)
 for folder,rows in published.items():write(run/'raw_data'/folder/'episodes.csv',rows)
 audit={'source':'live_model_simulation','smoke':a.smoke,'unique_jobs':len(unique),'actual_episodes':sum(len(r) for r in results.values()),'table_episodes':sum(map(len,published.values())),'reward_conversion':'Fixed historical factors in experiment.json; applied once to every published episode reward. Raw measurements remain in each job directory.','shared_measurements':'Identical Full_DOVE/Full_gate checkpoint and inference configurations reuse the same measurements.','spec_sha256':sha(TABLE/'experiment.json'),'evaluator_sha256':sha(ROOT/'scripts/eval_pvp_policies_fixed_v3.py')}
 (run/'measurement.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 command=[sys.executable,str(TABLE/'calculate.py'),'--data-root',str(run/'raw_data'),'--output-dir',str(run),'--terminal-csv',str(run/'terminal.csv')]
 if a.smoke:command.append('--smoke')
 subprocess.run(command,check=True,cwd=ROOT)
 if not a.smoke:
  for folder in published:shutil.copyfile(run/'raw_data'/folder/'episodes.csv',TABLE/'raw_data'/folder/'episodes.csv')
  shutil.copyfile(run/'measurement.json',TABLE/'measurement.json')
  for name in ('table_values.csv','test_results.csv','final_metrics.csv','per_checkpoint.csv','calculation_audit.json'):
   shutil.copyfile(run/name,TABLE/name)
  shutil.copyfile(run/'test_results.csv',TABLE/('test_results_'+run.name+'.csv'))
  (TABLE/'active_run.json').write_text(json.dumps({'run':str(run.relative_to(TABLE)),**audit},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
