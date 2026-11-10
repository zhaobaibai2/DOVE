"""Run all seven frozen Full-DOVE inference modes from the current manuscript."""
import argparse,csv,hashlib,json,math,os,shutil,subprocess,sys,time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
ROOT=Path(__file__).resolve().parents[1];TABLE=ROOT/'02_data/raw_data/TABLES/TAB-03'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def identity(j):return (j['model'],tuple(j['flags']),j['episodes'])
def pids_running(pids):
 for pid in pids:
  try:args=[x.decode(errors='replace') for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
  except OSError:continue
  if len(args)>1 and Path(args[1]).name=='eval_table03.py':return True
 return False
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4);p.add_argument('--reuse-from',type=Path,help='Explicitly reuse completed, config-verified local live measurements');p.add_argument('--validate',action='store_true',help='Compare actual outcomes to original archived episodes');p.add_argument('--publish-after-pid',nargs='*',type=int,default=[],help=argparse.SUPPRESS);a=p.parse_args()
 if not 1<=a.workers<=6:p.error('--workers must be between 1 and 6')
 spec=json.loads((TABLE/'experiment_v2.json').read_text());jobs=spec['jobs'];run=TABLE/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 shutil.copy2(TABLE/'experiment_v2.json',run/'experiment.json')
 env=os.environ.copy();env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',DOVE_EPISODE_PROGRESS='1',DOVE_NO_TENSORBOARD='1',DOVE_CORE_METRICS_ONLY='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 env.pop('DOVE_TRAFFIC_SEED_BASE',None);env.pop('DOVE_EPISODE_START',None)
 cached={};cache_records={}
 if a.reuse_from:
  base=a.reuse_from.resolve();meta=json.loads((base/'measurement.json').read_text());snapshot=base/'experiment.snapshot.json'
  if meta.get('smoke') or meta.get('source')!='live_model_simulation' or meta['spec_sha256']!=sha(snapshot) or meta['evaluator_sha256']!=sha(ROOT/'scripts/eval_pvp_policies_fixed_v3.py'):raise ValueError('缓存不是同一评测代码与已校验完整配置的真实测试')
  old=json.loads(snapshot.read_text())
  for oldjob in old['jobs']:
   path=base/(oldjob['folder']+'_'+oldjob['checkpoint'])/'episodes.csv'
   if path.exists():cached[identity(oldjob)]=(path,oldjob)
 lock=Lock();events=[]
 def progress(j,ep,count,arrived,source):
  other=int(ep['reason'] not in ('success','crash','out_of_route','timeout'));text=(f"{j['label']} 第{j['checkpoint']}组 {count}/{1 if a.smoke else j['episodes']} arrive={ep['arrive']} crash={ep['crash']} out_of_route={ep['out_of_route']} timeout={ep['timeout']} other={other} success={arrived/count:.1%}")
  event={'record_type':'episode','setting':j['label'],'checkpoint':j['checkpoint'],'episode':count,'arrive':ep['arrive'],'crash':ep['crash'],'out_of_route':ep['out_of_route'],'timeout':ep['timeout'],'other':other,'success':arrived/count,'reward_raw':ep['reward'],'reward':ep['reward']*j['reward_factor'],'route_completion':ep['route_completion'],'measurement_source':source,'terminal_text':text if source=='live' else ''}
  events.append(event)
  terminal=run/'terminal.csv';new_file=not terminal.exists()
  with terminal.open('a',newline='',encoding='utf-8') as f:
   writer=csv.DictWriter(f,fieldnames=list(event))
   if new_file:writer.writeheader()
   writer.writerow(event)
  if source=='live':print(text,flush=True)
 def check_rows(rows,n):
  if len(rows)!=n or [int(x['episode_idx']) for x in rows]!=list(range(n)):raise ValueError('回合记录不完整或重复，未发布')
  for r in rows:
   if any(not math.isfinite(float(r[k])) for k in ('reward','success','crash','route_completion')):raise ValueError('回合指标缺失或非有限值')
 def evaluate(j):
  model=TABLE/j['model'];n=1 if a.smoke else j['episodes'];out=run/(j['folder']+'_'+j['checkpoint']);out.mkdir()
  if sha(model)!=j['model_sha256']:raise ValueError('模型哈希校验失败')
  cache=cached.get(identity(j)) if not a.smoke else None
  if cache:
   path,oldjob=cache
   if oldjob['model_sha256']!=j['model_sha256']:raise ValueError('缓存模型与当前配置不一致')
   rows=read(path);check_rows(rows,n);shutil.copy2(path,out/'episodes.csv');arrived=0
   with lock:
    for i,r in enumerate(rows,1):
     ep={'arrive':int(float(r['success'])),'crash':int(float(r['crash'])),'out_of_route':int(float(r['out_of_route'])),'timeout':int(float(r['timeout'])),'reason':r['reason'],'reward':float(r['reward']),'route_completion':float(r['route_completion'])};arrived+=ep['arrive'];progress(j,ep,i,arrived,'reused_complete_live_run')
    text=f"已完成 {j['label']} 第{j['checkpoint']}组 arrive={arrived}/{n} success={arrived/n:.1%}"
    events.append({'record_type':'reuse_summary','setting':j['label'],'checkpoint':j['checkpoint'],'measurement_source':'reused_complete_live_run','terminal_text':text});print(text,flush=True)
    cache_records[(j['folder'],j['checkpoint'])]={'source':str(path),'sha256':sha(path)}
   source='reused_complete_live_run'
  else:
   cmd=[sys.executable,'-u',str(ROOT/'scripts/eval_pvp_policies_fixed_v3.py'),'--log_dir',str(model.parent),'--policy',model.name,'--maps','104','--horizon','1000','--num_episodes',str(n),'--output_csv',str(out/'summary.csv'),'--episode_output_csv',str(out/'episodes.csv'),*j['flags']]
   count=arrived=0;tail=deque(maxlen=12)
   with subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) as proc:
    for line in proc.stdout:
     tail.append(line.rstrip())
     if not line.startswith('DOVE_EPISODE '):continue
     ep=json.loads(line[len('DOVE_EPISODE '):]);count+=1;arrived+=ep['arrive']
     with lock:progress(j,ep,count,arrived,'live')
    if proc.wait():raise RuntimeError(j['label']+' 测试失败：'+'\n'.join(tail))
   rows=read(out/'episodes.csv');check_rows(rows,n)
   if count!=n:raise ValueError('终端进度与实际回合数不一致')
   source='live'
  validation={}
  if a.validate:
   original=read(TABLE/'reference/inference_v2/raw_data'/j['folder']/(j['checkpoint']+'.csv'))[:n]
   validation={k:sum(float(x[k])!=float(y[k]) for x,y in zip(rows,original)) for k in ('success','crash','route_completion')}
  return j,rows,{'folder':j['folder'],'checkpoint':j['checkpoint'],'model_sha256':j['model_sha256'],'source':source,'episodes':n,'episodes_sha256':sha(out/'episodes.csv'),'archive_outcome_mismatches':validation,**cache_records.get((j['folder'],j['checkpoint']),{})}
 with ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(evaluate,jobs))
 write(run/'terminal.csv',events);published={}
 for j,rows,_ in results:
  for r in rows:published.setdefault(j['folder'],[]).append({'checkpoint':j['checkpoint'],'episode':r['episode_idx'],'success':r['success'],'crash':r['crash'],'route_completion':r['route_completion'],'reward':float(r['reward'])*j['reward_factor'],'reward_raw':r['reward'],'reward_factor':j['reward_factor']})
 for folder,rows in published.items():write(run/'raw_data'/folder/'episodes.csv',rows)
 audit={'schema':spec['schema'],'source':'live_model_simulation','smoke':a.smoke,'jobs':[x[2] for x in results],'modes':7,'checkpoints':5,'table_episodes':sum(map(len,published.values())),'fresh_episodes':sum(x[2]['episodes'] for x in results if x[2]['source']=='live'),'reused_live_episodes':sum(x[2]['episodes'] for x in results if x[2]['source']!='live'),'spec_sha256':sha(run/'experiment.json'),'evaluator_sha256':sha(ROOT/'scripts/eval_pvp_policies_fixed_v3.py'),'runner_sha256':sha(Path(__file__)),'configuration':'Original map104, traffic .06, horizon1000, 100 continuous episodes per job. Explicit reuse never counts as newly rerun simulation.','conversion':'Only fixed historical factors from experiment.json; raw reward retained in per-job and public episode CSV.'}
 (run/'measurement.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 cmd=[sys.executable,str(TABLE/'calculate.py'),'--protocol','inference','--data-root',str(run/'raw_data'),'--output-dir',str(run),'--terminal-csv',str(run/'terminal.csv')]
 if a.smoke:cmd.append('--smoke')
 subprocess.run(cmd,check=True,cwd=ROOT)
 if not a.smoke:
  # Finish the user's already running legacy process without interrupting it or
  # letting its last publication overwrite the new seven-mode table.
  while pids_running(a.publish_after_pid):time.sleep(5)
  for folder in published:
   target=TABLE/'raw_data'/folder;target.mkdir(exist_ok=True);shutil.copy2(run/'raw_data'/folder/'episodes.csv',target/'episodes.csv')
  for name in ('table_values.csv','test_results.csv','final_metrics.csv','per_checkpoint.csv','comparison.csv','calculation_audit.json','measurement.json'):shutil.copy2(run/name,TABLE/name)
  shutil.copy2(run/'test_results.csv',TABLE/('test_results_'+run.name+'.csv'))
  shutil.copy2(TABLE/'experiment_v2.json',TABLE/'experiment.json')
  (TABLE/'active_run.json').write_text(json.dumps({'run':str(run.relative_to(TABLE)),**audit},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
