"""Check standalone aggregations, archived results and console CSV agreement."""
import csv,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TABLES=ROOT/'02_data/raw_data/TABLES'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 with tempfile.TemporaryDirectory() as td:
  work=Path(td)
  for n in range(2,7):
   table=TABLES/f'TAB-{n:02d}';script=work/f'calculator{n}.py';script.write_bytes((table/'calculate.py').read_bytes());out=work/f'out{n}'
   data=table/('raw_data' if n==2 else ('reference/raw_data' if n==3 else 'raw_data'))
   proc=subprocess.run([sys.executable,str(script),'--data-root',str(data),'--output-dir',str(out)],capture_output=True,text=True,check=True,cwd=td)
   rows=read(out/'test_results.csv');lines=proc.stdout.strip().splitlines()
   if n==5:lines=lines[1:]
   assert lines==[r['terminal_text'] for r in rows],f'TAB-{n:02d} terminal/CSV mismatch'
   if n==2:
    import statistics
    actual=read(out/'table_values.csv')
    assert len(actual)==4
    for folder,rr in zip(('Positive-only','ER_wo_CPCal','CPCal_only','Full_DOVE'),actual):
     pairs=read(data/folder/'pairs.csv');nrows=len(pairs);denom=sum(int(x['samples']) for x in pairs)
     assert float(rr['rho_vlb'])==sum(int(x['rho_vlb_indicator']) for x in pairs)/nrows
     assert float(rr['hsr'])==sum(int(x['hsr_count_of_64']) for x in pairs)/denom
     assert float(rr['rmr_0p5'])==sum(int(x['rmr_0p5_count_of_64']) for x in pairs)/denom
     assert float(rr['pairacc_c_mean'])==sum(int(x['pairacc_c_indicator']) for x in pairs)/nrows
   else:assert read(table/'reference/table_values.csv')==read(out/'table_values.csv'),f'TAB-{n:02d} original aggregation changed' 
   assert len(read(out/'final_metrics.csv'))==len(rows)
   assert json.loads((out/'calculation_audit.json').read_text())['inputs']
   print(f'TAB-{n:02d}: standalone calculation and archived aggregation passed')
 for n in (2,3):
  table=TABLES/f'TAB-{n:02d}';spec=json.loads((table/'experiment.json').read_text())
  for m in spec.get('models',spec.get('jobs',[])):
   path=table/m.get('path',m.get('model',''));expected=m.get('sha256',m.get('model_sha256'))
   assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
 print('Packaged model checks passed')
if __name__=='__main__':main()
