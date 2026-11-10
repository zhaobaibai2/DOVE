#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import argparse, hashlib, json
from datetime import datetime
import csv, math, statistics
ROOT=Path(__file__).resolve().parent

parser=argparse.ArgumentParser()
parser.add_argument('--data-root',type=Path,default=ROOT/'raw_data')
parser.add_argument('--output-dir',type=Path,default=ROOT)
parser.add_argument('--terminal-csv',type=Path)
parser.add_argument('--smoke',action='store_true')
OPTIONS=parser.parse_args();DATA_ROOT=OPTIONS.data_root.resolve();OUTPUT=OPTIONS.output_dir.resolve();OUTPUT.mkdir(parents=True,exist_ok=True)

CONFIGS=(('Positive-only','Positive-only','run11_noer','critic_pair_diag_run11_no_er',('0.074','0.615','0.048','0.789')),('ER_wo_CPCal','ER w/o CPCal','run13_eronly','critic_pair_diag_run13_no_cpcal',('0.580','0.054','0.024','0.345')),('Full_DOVE','Full DOVE','run12_full','critic_pair_diag_run12_full',('0.175','0.341','0.041','0.891')))
def mean(v):return statistics.fmean(v)
def se(v):return statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else 0.0
DETAIL_NAME='per_checkpoint.csv'
EVIDENCE_NOTE='Local actor and critic pair records; run.sh test recomputes these using the packaged checkpoint models.'
PROCESSING_NOTE='Checkpoint means are averaged with equal weight; SE uses sample SD across checkpoints. Actor 256 pairs, 8 energy samples, 32 sampled actions; critic 512 pairs and target critics.'

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
 if OPTIONS.terminal_csv:
  with Path(OPTIONS.terminal_csv).open(encoding='utf-8-sig',newline='') as f:events=list(csv.DictReader(f))
 for row,line in zip(values,lines):
  print(line,flush=True);events.append({'record_type':'summary',**row,'terminal_text':line})
 save_csv(OUTPUT/'test_results.csv',events)
 if OUTPUT==ROOT:
  import shutil
  shutil.copyfile(OUTPUT/'test_results.csv',OUTPUT/('test_results_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.csv'))
 inputs=[{'file':str(p.relative_to(DATA_ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(DATA_ROOT.rglob('*.csv'))]
 (OUTPUT/'calculation_audit.json').write_text(json.dumps({'table':ROOT.name,'smoke':OPTIONS.smoke,'inputs':inputs,'metrics_source':EVIDENCE_NOTE,'processing':PROCESSING_NOTE},ensure_ascii=False,indent=2)+'\n')

def main():
 out=[];details=[]
 for folder,paper_setting,actor_group,critic_group,reported in CONFIGS:
  root=DATA_ROOT/folder;actor=defaultdict(list)
  with (root/'actor_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):actor[r['checkpoint']].append(r)
  ast=[]
  for checkpoint,rows in sorted(actor.items()):
   if not OPTIONS.smoke and len(rows)!=256:raise ValueError(f'{folder}/{checkpoint}: expected 256 actor pairs')
   ast.append({'pairsat_e':mean([float(r['pairsat_e']) for r in rows]),'ovr_e':mean([float(r['ragr']) for r in rows]),'diversity':mean([float(r['sample_diversity']) for r in rows])})
  critic=defaultdict(list)
  with (root/'critic_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):critic[r['policy_name']].append(float(r['pairacc_c']))
  if not OPTIONS.smoke and any(len(v)!=512 for v in critic.values()):raise ValueError(f'{folder}: expected 512 critic pairs/checkpoint')
  for checkpoint,rr in sorted(actor.items()):details.append({'setting':paper_setting,'checkpoint':checkpoint,'kind':'actor','samples':len(rr),'pairsat_e':mean([float(r['pairsat_e']) for r in rr]),'ovr_e':mean([float(r['ragr']) for r in rr]),'diversity':mean([float(r['sample_diversity']) for r in rr])})
  for checkpoint,rr in sorted(critic.items()):details.append({'setting':paper_setting,'checkpoint':checkpoint,'kind':'critic','samples':len(rr),'pairacc_c':mean(rr)})
  cm=[mean(v) for _,v in sorted(critic.items())];vals=[mean([r[k] for r in ast]) for k in ('pairsat_e','ovr_e','diversity')]+[mean(cm)];rounded=tuple(f'{v:.3f}' for v in vals)
  out.append({'paper_setting':paper_setting,'actor_source_group':actor_group,'actor_checkpoints':len(ast),'actor_pairs_per_checkpoint':256,'actor_pair_rows':sum(map(len,actor.values())),'critic_source_group':critic_group,'critic_checkpoints':len(cm),'critic_pairs_per_checkpoint':512,'critic_pair_rows':sum(map(len,critic.values())),'pairsat_e':f'{vals[0]:.15g}','pairsat_e_checkpoint_se':f'{se([r["pairsat_e"] for r in ast]):.15g}','ovr_e':f'{vals[1]:.15g}','ovr_e_checkpoint_se':f'{se([r["ovr_e"] for r in ast]):.15g}','diversity':f'{vals[2]:.15g}','diversity_checkpoint_se':f'{se([r["diversity"] for r in ast]):.15g}','pairacc_c':f'{vals[3]:.15g}','pairacc_c_checkpoint_se':f'{se(cm):.15g}','rounded_values':'/'.join(rounded),'reported_values':'/'.join(reported),'matches_final_tex':str(rounded==reported).lower()})

 path=OUTPUT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 values=[{'Setting':r['paper_setting'],'PairSat_E':float(r['pairsat_e']),'OVR_E':float(r['ovr_e']),'Diversity':float(r['diversity']),'PairAcc_C':float(r['pairacc_c'])} for r in out]
 lines=[f"{r['Setting']} PairSat_E={r['PairSat_E']:.3f} OVR_E={r['OVR_E']:.3f} Diversity={r['Diversity']:.3f} PairAcc_C={r['PairAcc_C']:.3f}" for r in values]
 publish(values,details,lines)
if __name__=='__main__':main()
