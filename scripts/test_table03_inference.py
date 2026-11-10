"""Validate seven-mode memberships and standalone aggregation of archived episodes."""
import csv,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TABLE=ROOT/'02_data/raw_data/TABLES/TAB-03'
def read(path):
 with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 spec=json.loads((TABLE/'experiment_v2.json').read_text());jobs=spec['jobs']
 assert spec['schema']=='inference_seven_modes_v2' and len(jobs)==35
 assert {j['checkpoint'] for j in jobs}=={'12000','13000','14000','20000','32000'}
 assert all(j['model'].startswith('models/Full_DOVE/') and j['episodes']==100 for j in jobs)
 assert len({j['model'] for j in jobs})==5
 for mode,tau in [('Loss_only_gate','1.0'),('Proxy_only_gate','1.0'),('No_disagreement','-1.0')]:
  selected=[j for j in jobs if j['folder']==mode];assert len(selected)==5
  for j in selected:
   flags=j['flags'];assert flags[flags.index('--final_gate_tau_u')+1]==tau
   expected={'Loss_only_gate':'prior_only','Proxy_only_gate':'proxy_only','No_disagreement':'proxy_prior'}[mode]
   assert flags[flags.index('--final_gate_mode')+1]==expected
 with tempfile.TemporaryDirectory() as temp:
  out=Path(temp);standalone=out/'calculate.py';standalone.write_bytes((TABLE/'calculate.py').read_bytes())
  result=subprocess.run([sys.executable,str(standalone),'--protocol','inference','--data-root',str(TABLE/'reference/inference_v2/raw_data'),'--output-dir',str(out)],capture_output=True,text=True,check=True,cwd=temp)
  summary=read(out/'table_values.csv');comparison=read(out/'comparison.csv');events=read(out/'test_results.csv')
  assert len(summary)==7 and len(comparison)==28 and all(r['matches_paper']=='True' for r in comparison)
  assert all(int(r['n_episodes'])==500 and int(r['n_checkpoints'])==5 for r in summary)
  assert result.stdout.strip().splitlines()==[r['terminal_text'] for r in events]
  assert len(read(out/'per_checkpoint.csv'))==35
  # Removing one entire checkpoint must be rejected instead of silently reducing
  # the episode denominator or producing a plausible incomplete table.
  import shutil
  damaged=out/'data';shutil.copytree(TABLE/'reference/inference_v2/raw_data',damaged)
  path=damaged/'Loss_only_gate/episodes.csv';rows=[r for r in read(path) if r['checkpoint']!='12000']
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  failure=subprocess.run([sys.executable,str(standalone),'--protocol','inference','--data-root',str(damaged),'--output-dir',str(out/'bad')],capture_output=True,text=True)
  assert failure.returncode!=0
 print('Seven modes / five fixed models / 100 episodes verified; 28 historical cells match; incomplete input rejected')
if __name__=='__main__':main()
