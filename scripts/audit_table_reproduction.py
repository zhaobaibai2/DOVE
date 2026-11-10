"""Audit fresh local measurements against the immutable archived table inputs."""
import csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'02_data/raw_data/TABLES'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def grouped(rows,key):
 out=defaultdict(list)
 for r in rows:out[r[key]].append(r)
 return out
def main():
 table=ROOT/'TAB-02';report=[]
 for group in ('Positive-only','ER_wo_CPCal','CPCal_only','Full_DOVE'):
  upstream={'Positive-only':'positive_only','ER_wo_CPCal':'er_without_cpcal','Full_DOVE':'full_dove'}.get(group)
  legacy={}
  if upstream:
   for r in read(table/'reference/rq2_v2/critic_legacy'/f'{upstream}.csv'):
    legacy[(Path(r['policy_name']).stem,Path(r['source']).name,int(r['source_index']))]=int(float(r['pairacc_c']))
  actual=grouped(read(table/'raw_data'/group/'pairs.csv'),'checkpoint')
  expected=50 if group in ('Positive-only','ER_wo_CPCal') else 46
  assert len(actual)==expected
  for checkpoint,rows in actual.items():
   candidates=list((table/'reference/rq2_v2'/group).glob(checkpoint+'*_pairs.csv'))
   assert len(candidates)==1,(group,checkpoint)
   old=read(candidates[0]);assert len(rows)==len(old)==256
   key='rho_vlb_indicator' if group=='CPCal_only' else 'vlb_ok'
   mismatches=sum(int(float(x[key]))!=int(y['rho_vlb_indicator']) for x,y in zip(old,rows));assert mismatches==0
   if group=='CPCal_only':
    for field in ('hsr_count_of_64','rmr_0p5_count_of_64','pairacc_c_indicator'):assert all(int(float(x[field]))==int(y[field]) for x,y in zip(old,rows))
   else:
    sampled=read(table/'reference/rq2_v2/resampled_actor'/upstream/(checkpoint+'_samples.csv'))
    assert len(sampled)==len(rows)
    for field in ('hsr_count_of_64','rmr_0p5_count_of_64'):assert all(int(x[field])==int(y[field]) for x,y in zip(sampled,rows))
    for r in rows:assert int(r['pairacc_c_indicator'])==legacy[(checkpoint,Path(r['source']).name,int(r['source_index']))]
   report.append({'group':group,'checkpoint':checkpoint,'pairs':len(rows),'vlb_mismatches':mismatches})
 (table/'provenance/rq2_v2/full_pair_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 table=ROOT/'TAB-03';active=json.loads((table/'active_run.json').read_text());run=table/active['run'];report=[]
 if active.get('schema')=='inference_seven_modes_v2':
  assert not active['smoke'] and active['table_episodes']==3500
  assert active['fresh_episodes']+active['reused_live_episodes']==3500
  spec=json.loads((table/'experiment_v2.json').read_text())
  for config in spec['configs']:
   folder=config['folder'];actual=grouped(read(table/'raw_data'/folder/'episodes.csv'),'checkpoint')
   assert set(actual)=={'12000','13000','14000','20000','32000'}
   for cp,rows in actual.items():
    original=read(table/'reference/inference_v2/raw_data'/folder/(cp+'.csv'));assert len(rows)==len(original)==100
    assert len({r['episode'] for r in rows})==100
    errors={k:sum(float(x[k])!=float(y[k]) for x,y in zip(rows,original)) for k in ('success','crash','route_completion')}
    report.append({'folder':folder,'checkpoint':cp,'episodes':100,'archive_outcome_mismatches':errors})
  events=read(table/'test_results.csv');assert sum(r['record_type']=='episode' for r in events)==3500;assert sum(r['record_type']=='summary' for r in events)==7
  (table/'provenance/inference_v2/full_episode_comparison.json').write_text(json.dumps({'run':active['run'],'comparison':report},ensure_ascii=False,indent=2)+'\n')
 else:
  assert not active['smoke'] and active['actual_episodes']==3100 and active['table_episodes']==3500
  for folder in ('positive_only','ER_wo_CPCal','Full_DOVE','Vanilla','TopK_8','No_gate','Full_gate'):
   actual=grouped(read(table/'raw_data'/folder/'episodes.csv'),'checkpoint');original=grouped(read(table/'reference/raw_data'/folder/'episodes.csv'),'checkpoint');assert set(actual)==set(original)
   for cp,rr in actual.items():
    oo=original[cp];assert len(rr)==len(oo)==100
    for x,y in zip(rr,oo):
     assert all(float(x[k])==float(y[k]) for k in ('success','crash','route_completion')),(folder,cp)
    ratios=[float(y['reward'])/float(x['reward']) for x,y in zip(rr,oo) if float(x['reward'])!=0]
    report.append({'folder':folder,'checkpoint':cp,'episodes':len(rr),'outcomes_identical':True,'legacy_to_current_return_ratio_min':min(ratios),'legacy_to_current_return_ratio_max':max(ratios),'legacy_max_return':max(float(x['reward']) for x in oo),'current_max_return':max(float(x['reward']) for x in rr)})
  events=read(table/'test_results.csv');assert sum(r['record_type']=='episode' for r in events)==3100;assert sum(r['record_type']=='summary' for r in events)==7
  for r in events:
   if r['record_type']=='episode':assert all(word not in r['terminal_text'] for word in ('.pkl','map=','reward='))
  (table/'provenance/full_episode_comparison.json').write_text(json.dumps({'run':active['run'],'comparison':report,'note':'Return ratios describe historical processing differences only. They do not modify fresh evaluation results.'},ensure_ascii=False,indent=2)+'\n')
 print('TAB-02: all four groups match archived VLB, 64-sample counts and critic ordering on the stated pairs')
 print('TAB-03: episode memberships, counts, archived comparison and terminal rows verified')
if __name__=='__main__':main()
