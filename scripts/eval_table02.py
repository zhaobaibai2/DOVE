"""Recompute current-paper TAB-02 (Table III): VLB, HSR, RMR0.5 and critic ordering."""
import argparse,csv,hashlib,io,json,os,re,shutil,subprocess,sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cuda')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
TABLE=ROOT/'02_data/raw_data/TABLES/TAB-02'
import numpy as np
import jax
import jax.numpy as jnp
from scripts import eval_pvp_policies_fixed_v3 as ev
from experiments.run_elbo_faithfulness_audit import build_schedule,build_eval_fn,compute_pair_metrics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sortkey(m):return int(re.search(r'policy-(\d+)-',m['filename'])[1])
def sample_kernel(agent,count):
 def one(policy,obs):
  o=obs[None,:];base=ev.random_key_from_data(o)
  return jax.vmap(lambda k:agent.get_action(jax.random.fold_in(base,k+12345),policy,o)[0])(jnp.arange(count,dtype=jnp.uint32))
 return jax.jit(jax.vmap(one,in_axes=(None,0)))
def critic_kernel(agent):
 @jax.jit
 def evaluate(q,obs,pos,neg):
  p1,_=agent.q(q[0],obs,pos);p2,_=agent.q(q[1],obs,pos);n1,_=agent.q(q[0],obs,neg);n2,_=agent.q(q[1],obs,neg)
  return jnp.ravel((p1+p2)*.5),jnp.ravel((n1+n2)*.5)
 return evaluate
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--validate',action='store_true',help='Check VLB indicators against archived per-pair measurements');a=p.parse_args()
 if a.batch_size<1:p.error('--batch-size must be positive')
 spec=json.loads((TABLE/'experiment.json').read_text())
 if spec['schema']!='rq2_vlb_hsr_rmr_v2':raise ValueError('实验配置版本不一致')
 pairpath=TABLE/spec['pair_file']
 if sha(pairpath)!=spec['pair_sha256']:raise ValueError('配对样本校验失败')
 data=np.load(pairpath);obs=data['actor_obs'];pos=data['actor_pos'];neg=data['actor_neg']
 with (TABLE/'provenance/rq2_v2/elbo_64noise/pair_manifest.csv').open(newline='') as f:frozen=list(csv.DictReader(f))
 ids=json.loads((TABLE/'inputs/pair_ids.json').read_text())['actor']
 if [(x['source'],int(x['index'])) for x in ids]!=[(Path(x['source']).name,int(x['index'])) for x in frozen]:raise ValueError('与论文固定配对行号不一致')
 pairhash=hashlib.sha256('\n'.join(f"{x['source']}:{x['index']}" for x in frozen).encode()).hexdigest()
 if pairhash!=spec['pair_manifest_sha256']:raise ValueError('论文配对清单校验失败')
 noise=np.random.RandomState(spec['noise_seed']).standard_normal((256,20,64,2)).astype(np.float32)
 if hashlib.sha256(noise.tobytes()).hexdigest()!=spec['noise_sha256']:raise ValueError('共享噪声校验失败')
 schedule=build_schedule(20,2,'beta1_posterior');ev.require_jax_gpu_ready()
 run=TABLE/'evaluations'/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 shutil.copy2(TABLE/'experiment.json',run/'experiment.json');np.save(run/'shared_noise.npy',noise)
 events=[];used=[];validations=[];n=4 if a.smoke else 256
 for group in spec['groups']:
  models=sorted([m for m in spec['models'] if m['group']==group],key=sortkey)
  if a.smoke:models=models[:1]
  with redirect_stdout(io.StringIO()):agent=ev.build_agent(obs.shape[1],ev.load_run_args(TABLE/'models'/group))
  vlb_fn=build_eval_fn(agent,schedule['alphas_cumprod']);sample_fn=sample_kernel(agent,64);critic_fn=critic_kernel(agent)
  pairrows=[]
  for number,m in enumerate(models,1):
   model=TABLE/m['path']
   if sha(model)!=m['sha256']:raise ValueError('模型校验失败')
   comp=ev._extract_guidance_components_from_payload(ev.load_policy_payload(model))
   if comp is None:raise ValueError('模型缺少所需参数')
   ep_parts=[];en_parts=[];hs=[];rs=[]
   for start in range(0,n,a.batch_size):
    end=min(start+a.batch_size,n);o=obs[start:end];pp=pos[start:end];nn=neg[start:end]
    ep,en=jax.device_get(vlb_fn(comp['policy_obj'],o,pp,nn,noise[start:end]));ep_parts.append(np.asarray(ep,np.float64));en_parts.append(np.asarray(en,np.float64))
    actions=np.clip(np.asarray(jax.device_get(sample_fn(comp['policy_obj'],o)),np.float32),-1.,1.)
    gap=pp-nn;norm2=np.sum(gap*gap,axis=1);mid=(pp+nn)*.5
    projection=np.sum((actions-mid[:,None,:])*gap[:,None,:],axis=-1)/norm2[:,None]
    hs.extend(np.sum(projection>0,axis=1).tolist());dist=np.linalg.norm(actions-nn[:,None,:],axis=-1)
    rs.extend(np.sum(dist<.5*np.sqrt(norm2)[:,None],axis=1).tolist())
   ep=np.concatenate(ep_parts);en=np.concatenate(en_parts)
   metrics=compute_pair_metrics(en-ep,pos[:n],neg[:n],schedule,.05,'l2')
   qp,qn=jax.device_get(critic_fn(comp['target_q_params'],obs[:n],pos[:n],neg[:n]));ind=(qn-qp>0).astype(int)
   rows=[{'checkpoint':model.stem,'pair_id':i,'source':frozen[i]['source'],'source_index':int(frozen[i]['index']),'delta_vlb':float(metrics['delta_vlb'][i]),'rho_vlb_indicator':int(metrics['vlb_ok'][i]),'hsr_count_of_64':int(hs[i]),'rmr_0p5_count_of_64':int(rs[i]),'samples':64,'qbar_pos':float(qp[i]),'qbar_neg':float(qn[i]),'pairacc_c_indicator':int(ind[i])} for i in range(n)]
   pairrows.extend(rows);used.append(m)
   out=run/'checkpoints'/group;out.mkdir(parents=True,exist_ok=True);write(out/(model.stem+'_pairs.csv'),rows);np.savez_compressed(out/(model.stem+'_energies.npz'),positive_mse=ep,negative_mse=en)
   if a.validate and not a.smoke:
    candidates=list((TABLE/'reference/rq2_v2'/group).glob(model.stem+'*_pairs.csv'))
    if candidates:
     with candidates[0].open(newline='') as f:old=list(csv.DictReader(f))
     key='rho_vlb_indicator' if group=='CPCal_only' else 'vlb_ok'
     mismatch=sum(int(float(x[key]))!=y['rho_vlb_indicator'] for x,y in zip(old,rows))
     if len(old)!=n or mismatch:raise ValueError(f'{group} VLB 逐样本核对失败')
     if group=='CPCal_only':
      for field in ('hsr_count_of_64','rmr_0p5_count_of_64','pairacc_c_indicator'):
       if any(int(float(x[field]))!=y[field] for x,y in zip(old,rows)):raise ValueError(f'{group} {field} 核对失败')
     validations.append({'group':group,'checkpoint':model.stem,'pairs':n,'vlb_mismatches':mismatch})
   values={'rho_vlb':float(np.mean(metrics['vlb_ok'])),'hsr':sum(hs)/(n*64),'rmr_0p5':sum(rs)/(n*64),'pairacc_c_mean':float(np.mean(ind))}
   text=f"{group} 第{number}/{len(models)}组 rho_VLB={values['rho_vlb']:.3f} HSR={values['hsr']:.3f} RMR_0.5={values['rmr_0p5']:.3f} PairAcc_C_mean={values['pairacc_c_mean']:.3f}"
   print(text,flush=True);events.append({'record_type':'checkpoint','setting':group,'checkpoint':model.stem,'pairs':n,'samples':n*64,**values,'terminal_text':text});write(run/'terminal.csv',events)
  write(run/'raw_data'/group/'pairs.csv',pairrows)
 audit={'schema':spec['schema'],'source':'live_checkpoint_diagnostics','smoke':a.smoke,'models':used,'validations':validations,'pairs_per_checkpoint':n,'samples_per_pair':64,'noise_sha256':spec['noise_sha256'],'spec_sha256':sha(run/'experiment.json'),'code_sha256':sha(Path(__file__)),'paper_tex_sha256':spec['paper_tex_sha256'],'critic_note':'Recomputed on the same 256 intervention pairs and actor checkpoint memberships, as specified by DOVE_v3.tex.'}
 (run/'measurement.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
 cmd=[sys.executable,str(TABLE/'calculate.py'),'--data-root',str(run/'raw_data'),'--output-dir',str(run),'--terminal-csv',str(run/'terminal.csv')]
 if a.smoke:cmd.append('--smoke')
 subprocess.run(cmd,check=True)
 if not a.smoke:
  for group in spec['groups']:
   target=TABLE/'raw_data'/group;target.mkdir(exist_ok=True);shutil.copyfile(run/'raw_data'/group/'pairs.csv',target/'pairs.csv')
  for name in ('table_values.csv','test_results.csv','final_metrics.csv','per_checkpoint.csv','comparison.csv','calculation_audit.json','measurement.json'):shutil.copyfile(run/name,TABLE/name)
  shutil.copyfile(run/'test_results.csv',TABLE/('test_results_'+run.name+'.csv'))
  (TABLE/'active_run.json').write_text(json.dumps({'run':str(run.relative_to(TABLE)),'schema':spec['schema'],'source':audit['source']},indent=2)+'\n')
if __name__=='__main__':main()
