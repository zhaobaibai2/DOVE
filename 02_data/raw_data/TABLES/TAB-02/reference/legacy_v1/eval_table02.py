"""Recompute TAB-02 from frozen intervention pairs and every original model."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys, io
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TABLE=ROOT/'02_data/raw_data/TABLES/TAB-02'
import numpy as np
from scripts import eval_pvp_policies_fixed_v3 as ev
import jax
import jax.numpy as jnp
PREFIX={'Positive-only':'run11_noer','ER_wo_CPCal':'run13_eronly','Full_DOVE':'run12_full'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def kernels(agent):
 times=np.linspace(0,agent.num_timesteps-1,8,dtype=np.int32)
 def energy_one(policy,obs,pos,neg):
  o=obs[None,:];base=ev.random_key_from_data(o)
  def at(i,t):
   noise=jax.random.normal(jax.random.fold_in(base,i+7919),(1,2))
   xp=agent.diffusion.q_sample(t,pos[None,:],noise);xn=agent.diffusion.q_sample(t,neg[None,:],noise)
   return jnp.mean((agent.predict_noise(policy,o,t,xp)-noise)**2),jnp.mean((agent.predict_noise(policy,o,t,xn)-noise)**2)
  return jax.vmap(at)(jnp.arange(8,dtype=jnp.int32),jnp.asarray(times))
 def sample_one(policy,obs):
  o=obs[None,:];base=ev.random_key_from_data(o)
  return jax.vmap(lambda k: jnp.clip(agent.get_action(jax.random.fold_in(base,k+12345),policy,o)[0],-1.,1.))(jnp.arange(32,dtype=jnp.int32))
 def critic(q,obs,pos,neg):
  p1,_=agent.q(q[0],obs,pos);p2,_=agent.q(q[1],obs,pos);n1,_=agent.q(q[0],obs,neg);n2,_=agent.q(q[1],obs,neg)
  return jnp.ravel((p1+p2)*.5),jnp.ravel((n1+n2)*.5)
 return jax.jit(jax.vmap(energy_one,in_axes=(None,0,0,0))),jax.jit(jax.vmap(sample_one,in_axes=(None,0))),jax.jit(critic)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--validate',action='store_true',help='Compare first full actor checkpoint of every group to archived pair rows');a=p.parse_args()
 if a.batch_size<1:p.error('--batch-size must be positive')
 spec=json.loads((TABLE/'experiment.json').read_text());pairpath=TABLE/spec['pair_file']
 if sha(pairpath)!=spec['pair_sha256']:raise ValueError('样本校验失败')
 ev.require_jax_gpu_ready();data=np.load(pairpath);run=TABLE/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 events=[];models_used=[];validations=[]
 for group in PREFIX:
  models=[m for m in spec['models'] if m['group']==group]
  if a.smoke:models=[next(m for m in models if m['actor'] and m['critic'])]
  with redirect_stdout(io.StringIO()):
   agent=ev.build_agent(data['actor_obs'].shape[1],ev.load_run_args(TABLE/'models'/group))
  energy,sample,critic=kernels(agent)
  actorrows=[];criticrows=[]
  for num,m in enumerate(models,1):
   path=TABLE/m['path']
   if sha(path)!=m['sha256']:raise ValueError('模型校验失败')
   payload=ev.load_policy_payload(path);comp=ev._extract_guidance_components_from_payload(payload)
   if comp is None:raise ValueError('模型缺少诊断所需参数')
   models_used.append(m);checkpoint=PREFIX[group]+'_'+path.stem
   ar=[]
   if m['actor']:
    n=4 if a.smoke else spec['actor_pairs']
    for start in range(0,n,a.batch_size):
     stop=min(start+a.batch_size,n);o=data['actor_obs'][start:stop];pos=data['actor_pos'][start:stop];neg=data['actor_neg'][start:stop]
     ep,en=jax.device_get(energy(comp['policy_obj'],o,pos,neg));acts=np.asarray(jax.device_get(sample(comp['policy_obj'],o)),np.float32)
     ep=np.asarray(ep,dtype=np.float64).mean(axis=1);en=np.asarray(en,dtype=np.float64).mean(axis=1)
     for i,aa in enumerate(acts):
      distances=np.linalg.norm(aa[:,None,:]-aa[None,:,:],axis=-1);minimum=float(np.linalg.norm(aa-neg[i],axis=-1).min())
      ar.append({'checkpoint':checkpoint,'pair':start+i,'energy_pos':float(ep[i]),'energy_neg':float(en[i]),'pairsat_e':float(ep[i]+spec['margin']<=en[i]),'ragr':float(minimum<=spec['rejected_radius']),'sample_diversity':float(distances[np.triu_indices(32,1)].mean()),'min_dist_to_rejected':minimum})
    actorrows.extend(ar)
    if a.validate and not any(v['group']==group for v in validations):
     with (TABLE/'reference/raw_data'/group/'actor_pairs.csv').open(newline='',encoding='utf-8-sig') as f:reference=[r for r in csv.DictReader(f) if r['checkpoint']==checkpoint]
     if len(reference)!=len(ar):raise ValueError('验证样本数不一致')
     errors={k:max(abs(float(x[k])-float(y[k])) for x,y in zip(ar,reference)) for k in ('pairsat_e','ragr','sample_diversity')}
     if errors['pairsat_e']!=0 or errors['ragr']!=0 or errors['sample_diversity']>2e-6:raise ValueError(f'原公式核对失败 {group}: {errors}')
     validations.append({'group':group,'checkpoint':checkpoint,'max_abs_error':errors})
   cr=[]
   if m['critic']:
    n=4 if a.smoke else spec['critic_pairs'];qp,qn=jax.device_get(critic(comp['target_q_params'],data['critic_obs'][:n],data['critic_pos'][:n],data['critic_neg'][:n]))
    cr=[{'policy_name':path.name,'pair':i,'qbar_pos':float(x),'qbar_neg':float(y),'pairacc_c':float(y-x>0)} for i,(x,y) in enumerate(zip(qp,qn))];criticrows.extend(cr)
   text=f"{group} 第{num}/{len(models)}组"
   if ar:text+=f" PairSat_E={np.mean([r['pairsat_e'] for r in ar]):.3f} OVR_E={np.mean([r['ragr'] for r in ar]):.3f} Diversity={np.mean([r['sample_diversity'] for r in ar]):.3f}"
   if cr:text+=f" PairAcc_C={np.mean([r['pairacc_c'] for r in cr]):.3f}"
   print(text,flush=True);events.append({'record_type':'checkpoint','setting':group,'checkpoint':path.stem,'actor_pairs':len(ar),'critic_pairs':len(cr),'pairsat_e':np.mean([r['pairsat_e'] for r in ar]) if ar else '', 'ovr_e':np.mean([r['ragr'] for r in ar]) if ar else '', 'diversity':np.mean([r['sample_diversity'] for r in ar]) if ar else '', 'pairacc_c':np.mean([r['pairacc_c'] for r in cr]) if cr else '', 'terminal_text':text})
  write(run/'raw_data'/group/'actor_pairs.csv',actorrows);write(run/'raw_data'/group/'critic_pairs.csv',criticrows)
 write(run/'terminal.csv',events)
 audit={'source':'live_checkpoint_diagnostics','smoke':a.smoke,'models':models_used,'validations':validations,'pairs_sha256':sha(pairpath),'experiment_sha256':sha(TABLE/'experiment.json'),'code_sha256':sha(Path(__file__)),'formula_origin':'Reconstructed from original energy/sampler/critic functions; diversity verified against archived per-pair measurements. Missing actor audit script was not recovered.'}
 (run/'measurement.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 cmd=[sys.executable,str(TABLE/'calculate.py'),'--data-root',str(run/'raw_data'),'--output-dir',str(run),'--terminal-csv',str(run/'terminal.csv')]
 if a.smoke:cmd.append('--smoke')
 subprocess.run(cmd,check=True)
 if not a.smoke:
  for group in PREFIX:
   for name in ('actor_pairs.csv','critic_pairs.csv'):shutil.copyfile(run/'raw_data'/group/name,TABLE/'raw_data'/group/name)
  for name in ('table_values.csv','test_results.csv','final_metrics.csv','per_checkpoint.csv','calculation_audit.json','measurement.json'):shutil.copyfile(run/name,TABLE/name)
  shutil.copyfile(run/'test_results.csv',TABLE/('test_results_'+run.name+'.csv'))
  (TABLE/'active_run.json').write_text(json.dumps({'run':str(run.relative_to(TABLE)),'source':audit['source']},indent=2)+'\n')
if __name__=='__main__':main()
