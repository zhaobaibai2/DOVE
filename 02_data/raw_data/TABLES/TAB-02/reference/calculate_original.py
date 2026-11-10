#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import csv, math, statistics
ROOT=Path(__file__).resolve().parent
CONFIGS=(('Positive-only','Positive-only','run11_noer','critic_pair_diag_run11_no_er',('0.074','0.615','0.048','0.789')),('ER_wo_CPCal','ER w/o CPCal','run13_eronly','critic_pair_diag_run13_no_cpcal',('0.580','0.054','0.024','0.345')),('Full_DOVE','Full DOVE','run12_full','critic_pair_diag_run12_full',('0.175','0.341','0.041','0.891')))
def mean(v):return statistics.fmean(v)
def se(v):return statistics.stdev(v)/math.sqrt(len(v)) if len(v)>1 else 0.0
def main():
 out=[]
 for folder,paper_setting,actor_group,critic_group,reported in CONFIGS:
  root=ROOT/'raw_data'/folder;actor=defaultdict(list)
  with (root/'actor_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):actor[r['checkpoint']].append(r)
  ast=[]
  for checkpoint,rows in sorted(actor.items()):
   if len(rows)!=256:raise ValueError(f'{folder}/{checkpoint}: expected 256 actor pairs')
   ast.append({'pairsat_e':mean([float(r['pairsat_e']) for r in rows]),'ovr_e':mean([float(r['ragr']) for r in rows]),'diversity':mean([float(r['sample_diversity']) for r in rows])})
  critic=defaultdict(list)
  with (root/'critic_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
   for r in csv.DictReader(f):critic[r['policy_name']].append(float(r['pairacc_c']))
  if any(len(v)!=512 for v in critic.values()):raise ValueError(f'{folder}: expected 512 critic pairs/checkpoint')
  cm=[mean(v) for _,v in sorted(critic.items())];vals=[mean([r[k] for r in ast]) for k in ('pairsat_e','ovr_e','diversity')]+[mean(cm)];rounded=tuple(f'{v:.3f}' for v in vals)
  out.append({'paper_setting':paper_setting,'actor_source_group':actor_group,'actor_checkpoints':len(ast),'actor_pairs_per_checkpoint':256,'actor_pair_rows':sum(map(len,actor.values())),'critic_source_group':critic_group,'critic_checkpoints':len(cm),'critic_pairs_per_checkpoint':512,'critic_pair_rows':sum(map(len,critic.values())),'pairsat_e':f'{vals[0]:.15g}','pairsat_e_checkpoint_se':f'{se([r["pairsat_e"] for r in ast]):.15g}','ovr_e':f'{vals[1]:.15g}','ovr_e_checkpoint_se':f'{se([r["ovr_e"] for r in ast]):.15g}','diversity':f'{vals[2]:.15g}','diversity_checkpoint_se':f'{se([r["diversity"] for r in ast]):.15g}','pairacc_c':f'{vals[3]:.15g}','pairacc_c_checkpoint_se':f'{se(cm):.15g}','rounded_values':'/'.join(rounded),'reported_values':'/'.join(reported),'matches_final_tex':str(rounded==reported).lower()})
 if not all(r['matches_final_tex']=='true' for r in out):raise ValueError('paper values do not match')
 path=ROOT/'table_values.csv'
 with path.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(path)
if __name__=='__main__':main()
