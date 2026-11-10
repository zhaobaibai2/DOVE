#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime
import csv, math, statistics
ROOT=Path(__file__).resolve().parent

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,default=ROOT/'raw_data')
parser.add_argument('--output-dir',type=Path,default=ROOT)
parser.add_argument('--terminal-csv',type=Path)
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--protocol',choices=('auto','legacy','inference'),default='auto')
OPTIONS=parser.parse_args();DATA_ROOT=OPTIONS.data_root.resolve();OUTPUT=OPTIONS.output_dir.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)

LEGACY_CONFIGS=(('A','Positive-only (DBC + critic)','positive_only',('0.442','0.362','0.887','324.99')),('A','ER w/o CPCal','ER_wo_CPCal',('0.680','0.320','0.852','315.25')),('A','Full DOVE','Full_DOVE',('0.940','0.060','0.966','363.36')),('B','Repaired actor without test-time guidance','Vanilla',('0.400','0.400','0.768','288.53')),('B','Top-K rerank (K=8)','TopK_8',('0.388','0.500','0.703','261.45')),('B','Projected guidance, always execute','No_gate',('0.472','0.490','0.658','244.14')),('B','Projected guidance with acceptance/fallback gate','Full_gate',('0.842','0.158','0.883','330.76')))
INFERENCE_CONFIGS=(('Inference','Actor only (no guidance)','Vanilla',('0.400','0.400','0.768','288.53')),('Inference','Top-8 rerank','TopK_8',('0.388','0.500','0.703','261.45')),('Inference','Guidance, always execute','No_gate',('0.472','0.490','0.658','244.14')),('Inference','Guidance, loss-only gate','Loss_only_gate',('0.492','0.470','0.647','240.55')),('Inference','Guidance, proxy-only gate','Proxy_only_gate',('0.770','0.230','0.854','322.30')),('Inference','Guidance, w/o disagreement','No_disagreement',('0.838','0.162','0.874','330.81')),('Inference','Guidance, full gate','Full_gate',('0.842','0.158','0.883','330.76')))
IS_INFERENCE=OPTIONS.protocol=='inference' or (OPTIONS.protocol=='auto' and all((DATA_ROOT/f/'episodes.csv').exists() for f in ('Loss_only_gate','Proxy_only_gate','No_disagreement')))
CONFIGS=INFERENCE_CONFIGS if IS_INFERENCE else LEGACY_CONFIGS
SCHEMA='inference_seven_modes_v2' if IS_INFERENCE else 'legacy_training_and_inference_v1'
METRICS=('success','crash','route_completion','reward')
DETAIL_NAME='per_checkpoint.csv'
EVIDENCE_NOTE='Local episode records; run.sh test loads the packaged models and generates fresh episodes.'
PROCESSING_NOTE='Pool episodes within each setting; retain sample SD and SE. Every stored reward already includes its fixed historical conversion exactly once. Preserve original checkpoint membership and 500/600/400 panel A counts.'
if IS_INFERENCE:
 PROCESSING_NOTE='Seven execution modes use the same five frozen Full_DOVE checkpoints with 100 consecutive episodes each; pool 500 episodes per mode. Retain sample SD, SE, and per-checkpoint metrics. Stored reward includes its fixed historical factor once; reward_raw remains unchanged. No training-ablation rows enter this table.'

# Self-contained: only the Python standard library and this table's local inputs.
def save_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def publish(values,details,lines):
 save_csv(OUTPUT/'final_metrics.csv',values)
 if details:save_csv(OUTPUT/DETAIL_NAME,details)
 events=[]
 terminal=OPTIONS.terminal_csv
 if terminal is None and OUTPUT==ROOT and DATA_ROOT==(ROOT/'raw_data').resolve() and (ROOT/'active_run.json').exists():
  active=json.loads((ROOT/'active_run.json').read_text());run=ROOT/active['run'];candidate=run/'terminal.csv'
  same=all((run/'raw_data'/folder/'episodes.csv').exists() and hashlib.sha256((run/'raw_data'/folder/'episodes.csv').read_bytes()).digest()==hashlib.sha256((DATA_ROOT/folder/'episodes.csv').read_bytes()).digest() for _,_,folder,_ in CONFIGS)
  if candidate.exists() and same:terminal=candidate
 if terminal:
  with Path(terminal).open(encoding='utf-8-sig',newline='') as f:events=list(csv.DictReader(f))
 if IS_INFERENCE:
  labels={folder:label for _,label,folder,_ in CONFIGS}
  for event in events:
   setting=event.pop('setting',None)
   if setting is not None:event['Setting']=labels.get(setting,setting)
 for row,line in zip(values,lines):
  print(line,flush=True);events.append({'record_type':'summary',**row,'terminal_text':line})
 save_csv(OUTPUT/'test_results.csv',events)
 if OUTPUT==ROOT:
  import shutil
  shutil.copyfile(OUTPUT/'test_results.csv',OUTPUT/('test_results_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv'))
 inputs=[{'file':str(p.relative_to(DATA_ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DATA_ROOT.rglob('*.csv'))]
 (OUTPUT/'calculation_audit.json').write_text(json.dumps({'table':ROOT.name,'schema':SCHEMA,'smoke':OPTIONS.smoke,'inputs':inputs,'metrics_source':EVIDENCE_NOTE,'processing':PROCESSING_NOTE},ensure_ascii=False,indent=2)+'\n')

def main():
 out=[];details=[]
 for panel,label,folder,reported in CONFIGS:
  with (DATA_ROOT/folder/'episodes.csv').open(encoding='utf-8-sig',newline='') as f:episodes=list(csv.DictReader(f))
  for checkpoint in sorted({r['checkpoint'] for r in episodes}):
   rr=[r for r in episodes if r['checkpoint']==checkpoint];details.append({'setting':label,'checkpoint':checkpoint,'episodes':len(rr),**{k:statistics.fmean(float(r[k]) for r in rr) for k in METRICS}})
  checkpoints=sorted({r['checkpoint'] for r in episodes})
  if IS_INFERENCE and not OPTIONS.smoke:
   if set(checkpoints)!={'12000','13000','14000','20000','32000'} or any(sum(r['checkpoint']==c for r in episodes)!=100 for c in checkpoints):raise ValueError('七模式表要求固定五个 checkpoint、每个 100 回合')
   if episodes and 'episode' in episodes[0] and len({(r['checkpoint'],r['episode']) for r in episodes})!=500:raise ValueError('重复回合记录')
  row={'panel':panel,'paper_setting':label,'n_checkpoints':len(checkpoints),'n_episodes':len(episodes),'checkpoint_membership':';'.join(checkpoints)};rounded=[]
  for metric in METRICS:
   values=[float(r[metric]) for r in episodes];m=statistics.fmean(values);sd=statistics.stdev(values) if len(values)>1 else 0.;se=sd/math.sqrt(len(values)) if len(values)>1 else 0.
   name='collision' if metric=='crash' else metric;row[name]=f'{m:.12g}';row[name+'_sample_sd']=f'{sd:.12g}';row[name+'_se']=f'{se:.12g}';rounded.append(f'{m:.{2 if metric=="reward" else 3}f}')
  row['reported_values']='/'.join(reported);row['rounded_values']='/'.join(rounded);row['matches_final_tex']=str(tuple(rounded)==reported).lower();row['matches_caption_500_episode_claim']=str(len(episodes)==500).lower();out.append(row)
 path=OUTPUT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 values=[{'Panel':r['panel'],'Setting':r['paper_setting'],'Episodes':r['n_episodes'],'Succ.↑':float(r['success']),'Crash↓':float(r['collision']),'Route Completion↑':float(r['route_completion']),'Return↑':float(r['reward'])} for r in out]
 lines=[f"{r['Setting']} arrive={r['Succ.↑']:.1%} crash={r['Crash↓']:.1%} route_completion={r['Route Completion↑']:.3f} Return↑={r['Return↑']:.2f}" for r in values]
 if IS_INFERENCE:
  comparison=[]
  for row,(_,_,_,reported) in zip(out,CONFIGS):
   for k,ref in zip(('success','collision','route_completion','reward'),reported):
    value=float(row[k]);digits=2 if k=='reward' else 3;comparison.append({'setting':row['paper_setting'],'metric':k,'recomputed':value,'paper':float(ref),'matches_paper':f'{value:.{digits}f}'==ref})
  save_csv(OUTPUT/'comparison.csv',comparison)
 publish(values,details,lines)
if __name__=='__main__':main()
