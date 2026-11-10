#!/usr/bin/env python3
"""Recover table model/input identities once; generated runners use local files only."""
import argparse,csv,hashlib,json,pickle,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'02_data/raw_data/TABLES'
RUNS={
 'Positive-only':'logs/11_train_pvp_dbp_critic_seed104_50000_20260521_204732/pvp_dacer_20260521_204744_s0',
 'ER_wo_CPCal':'logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0',
 'Full_DOVE':'logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0'}
PREFIX={'Positive-only':'run11_noer','ER_wo_CPCal':'run13_eronly','Full_DOVE':'run12_full'}
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);a=ap.parse_args();b=a.source_root.resolve()
 existing=TABLES/'TAB-02/experiment.json'
 if existing.exists() and json.loads(existing.read_text()).get('schema')=='rq2_vlb_hsr_rmr_v2':
  raise RuntimeError('此恢复工具对应旧版表，不能覆盖 DOVE_v3.tex 的新指标配置。当前本地模型和输入已经完整，请直接运行 run_tab02.sh test。')
 harness=b/'Latex0805/DOVE__Learning_Diffusion_Policies_from_Accepted_and_Rejected_Actions_in_Human_in_the_Loop_Driving/git/02_data/raw_data/harness_project/artifacts/tables'
 evidence=[]
 def copy(source,target):
  target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():shutil.copy2(source,target)
  if sha(source)!=sha(target):raise ValueError(f'Conflicting local asset: {target}')
  evidence.append({'source':str(source),'local':str(target.relative_to(ROOT)),'sha256':sha(target)})
 for n in (2,3):
  table=TABLES/f'TAB-{n:02d}'
  if not (table/'reference/raw_data').exists():shutil.copytree(table/'raw_data',table/'reference/raw_data')
 t=TABLES/'TAB-02';models=[]
 for group,rel in RUNS.items():
  actor=read(t/'raw_data'/group/'actor_pairs.csv');critic=read(t/'raw_data'/group/'critic_pairs.csv')
  actor_names=sorted({r['checkpoint'].removeprefix(PREFIX[group]+'_')+'.pkl' for r in actor})
  critic_names=sorted({r['policy_name'] for r in critic})
  for name in sorted(set(actor_names)|set(critic_names)):
   target=t/'models'/group/name;copy(b/rel/name,target)
   models.append({'group':group,'filename':name,'path':str(target.relative_to(t)),'sha256':sha(target),'actor':name in actor_names,'critic':name in critic_names,'training_seed':0})
  copy(b/rel/'args.json',t/'models'/group/'args.json')
 actor_file=b/'logs/110_actor_prior_audit_gpu6_cvci_safe_20260604_115410/run12_full_policy-12000-23998_pairs.csv'
 actor_rows=read(actor_file)
 critic_file=harness/'TAB-02/git_recovered_upstream/eval2/diagnostics/critic_pair_diag_run12_full/intervention_pair_rows.csv'
 cr=read(critic_file);first=cr[0]['policy_name'];critic_rows=[r for r in cr if r['policy_name']==first]
 loaded={}
 def arrays(rows,index_key):
  obs=[];pos=[];neg=[];ids=[]
  for row in rows:
   relative=row['source'].replace('/root/autodl-tmp/dacer_pvp/test0508/','')
   p=Path(relative);p=p if p.is_absolute() else b/p
   if p not in loaded:
    d=pickle.load(p.open('rb'));loaded[p]=d.get('raw_data',d)
    copy(p,t/'inputs'/p.name)
   d=loaded[p];i=int(row[index_key]);obs.append(d['observation'][i]);pos.append(d['action_behavior'][i]);neg.append(d['action_agent'][i]);ids.append({'source':p.name,'index':i})
  return np.asarray(obs,np.float32),np.asarray(pos,np.float32),np.asarray(neg,np.float32),ids
 t.joinpath('inputs').mkdir(exist_ok=True)
 ao,ap,an,ai=arrays(actor_rows,'index');co,cp,cn,ci=arrays(critic_rows,'source_index')
 assert len(ao)==256 and len(co)==512
 np.savez_compressed(t/'inputs/pairs.npz',actor_obs=ao,actor_pos=ap,actor_neg=an,critic_obs=co,critic_pos=cp,critic_neg=cn)
 save(t/'inputs/pair_ids.json',{'actor':ai,'critic':ci})
 save(t/'experiment.json',{'table':'TAB-02','models':models,'actor_pairs':256,'critic_pairs':512,'energy_samples':8,'action_samples':32,'margin':.05,'rejected_radius':.05,'diversity':'mean of Euclidean distances over distinct unordered pairs of 32 sampled actions','aggregation':'unweighted mean of checkpoint means; checkpoint SE=sample_SD/sqrt(checkpoints)','pair_file':'inputs/pairs.npz','pair_sha256':sha(t/'inputs/pairs.npz')})
 # Each TAB-03 row retains exactly its original checkpoint membership.
 t=TABLES/'TAB-03';jobs=[]
 folder_group={'positive_only':'Positive-only','ER_wo_CPCal':'ER_wo_CPCal','Full_DOVE':'Full_DOVE','Vanilla':'Full_DOVE','TopK_8':'Full_DOVE','No_gate':'Full_DOVE','Full_gate':'Full_DOVE'}
 guide=['--use_value_guidance','--guidance_lambda0','0.5','--guidance_beta_unc','1.0','--guidance_p_decay','1.0','--guidance_grad_clip','1.0','--guidance_kappa','1.0','--guidance_mode','proxy_value','--guidance_q_agg','conservative','--guidance_injection','clean_x0','--guidance_schedule','noise_level','--guidance_target','x0','--guidance_critic_params','target','--critic_objective','lower']
 gate=['--use_final_gate','--final_gate_mode','proxy_prior','--final_gate_proxy_compare','robust','--final_gate_min_proxy_gain','0.0','--final_gate_max_action_shift','0.2','--final_gate_tau_prior','0.02','--final_gate_tau_u','1.0','--final_gate_project_guidance','1','--final_gate_energy_samples','4']
 for folder,group in folder_group.items():
  rows=read(t/'raw_data'/folder/'episodes.csv');checkpoints=list(dict.fromkeys(r['checkpoint'] for r in rows))
  for checkpoint in checkpoints:
   name='final_policy_50000.pkl' if checkpoint=='final50000' else f'policy-{checkpoint}-{2*int(checkpoint)-2}.pkl'
   target=t/'models'/group/name;copy(b/RUNS[group]/name,target)
   copy(b/RUNS[group]/'args.json',target.parent/'args.json')
   flags=[] if folder=='Vanilla' else (['--rerank_topk','8','--guidance_q_agg','conservative','--critic_objective','lower'] if folder=='TopK_8' else guide+(gate if folder!='No_gate' else []))
   factor=.9827445944075866 if folder in ('Full_DOVE','Full_gate') and checkpoint in ('12000','13000','14000','32000') else 1.0
   jobs.append({'folder':folder,'checkpoint':checkpoint,'model':str(target.relative_to(t)),'model_sha256':sha(target),'episodes':sum(r['checkpoint']==checkpoint for r in rows),'flags':flags,'reward_factor':factor})
 save(t/'experiment.json',{'table':'TAB-03','maps':[104],'traffic_density':.06,'horizon':1000,'jobs':jobs,'aggregation':'pooled episode mean; sample SD and SE retained','conversion':'fixed historical factor applies only to the four originally transformed full-gate checkpoints; other rows unchanged','original_caption_counts':'Panel A actually 500/600/400, Panel B 500 per mode; preserve actual memberships'})
 for n in range(2,7):
  t=TABLES/f'TAB-{n:02d}';(t/'provenance').mkdir(exist_ok=True)
  for name in ['calculation.md','source_manifest.csv','raw_trial_ledger_search_summary.csv','unresolved_trial_ledger.csv','training_source_manifest.csv','inference_source_manifest.csv']:
   p=harness/f'TAB-{n:02d}'/name
   if p.exists():copy(p,t/'provenance'/name)
  save(t/'provenance/source_manifest.json',[r for r in evidence if f'/TAB-{n:02d}/' in r['local']])
 print('Recovered TAB-02 models:',len(models),'actor checkpoints:',sum(m['actor'] for m in models),'critic checkpoints:',sum(m['critic'] for m in models))
 print('TAB-03 complete jobs:',len(jobs),'episodes:',sum(j['episodes'] for j in jobs))
if __name__=='__main__':main()
