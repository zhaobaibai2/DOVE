#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict
from pathlib import Path

ROOT = Path('/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508')
OUT_DIR = ROOT / 'logs/data0528-latex3/table_repro_full_raw_data'
POLICY_SUMMARY = ROOT / 'logs/data0528-latex3/all_checkpoints_policy_mode_summary.csv'
PAIR_DIR = ROOT / 'logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151'
DBP_PAIR = ROOT / 'logs/99_every_eval_script_and_pair_diag_seed104_20260524_092640/pair_diag/run10_stage1b_summary.csv'

TABLES = [
    ('TABLE IV', 'TABLE_IV_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE IV.sh'),
    ('TABLE V', 'TABLE_V_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE V.sh'),
    ('TABLE VI', 'TABLE_VI_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VI.sh'),
    ('TABLE VII', 'TABLE_VII_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VII.sh'),
    ('TABLE VIII', 'TABLE_VIII_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VIII.sh'),
]

TABLE_COLS = ['mode', 'policy', 'step', 'Succ.', 'Coll.', 'OOR', 'Timeout', 'Route', 'Reward', 'Delta C', 'GateAcc', 'Latency']
PAIR_COLS = ['source', 'variant', 'checkpoint', 'n_pairs', 'PairSat', 'V_E', 'IAR', 'RAGR', 'EnergyGap', 'MinDist', 'Margin', 'RAGR_delta', 'RAGR_k']


def natural_key(value: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]


def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def num(value):
    try:
        x = float(value) if value not in (None, '') else float('nan')
    except Exception:
        return float('nan')
    return x if math.isfinite(x) else float('nan')


def fmt(value, digits=4):
    if value is None:
        return '--'
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return '--'
    x = num(s)
    if math.isfinite(x):
        if abs(x - round(x)) < 1e-10 and abs(x) < 1000000:
            return str(int(round(x)))
        if abs(x) >= 100:
            return f'{x:.3f}'
        if abs(x) >= 1:
            return f'{x:.4f}'.rstrip('0').rstrip('.')
        return f'{x:.4f}'.rstrip('0').rstrip('0').rstrip('.') if abs(x) >= 1e-4 else f'{x:.4g}'
    s = s.replace('|', '\\|')
    if s.endswith('.pkl') or '/' in s:
        return f'`{s}`'
    return s


def md_table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---:' if h in {'step', 'n_pairs', 'checkpoint'} else '---' for h in headers) + ' |']
    out.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(out)


def parse_methods(script: Path):
    methods = []
    current = None
    for line in script.read_text(encoding='utf-8').splitlines():
        m = re.search(r'echo\s+"(\d+)\.\s+(.+?)"', line)
        if m:
            current = {'idx': m.group(1), 'label': m.group(2), 'manifest': '', 'inputs': []}
            methods.append(current)
            continue
        if current is None:
            continue
        m = re.search(r'row manifest:\s*([^" ]+)', line)
        if m:
            current['manifest'] = m.group(1)
        m = re.search(r'input csv \[checkpoint ([^\]]+)\]:\s*([^" ]+)', line)
        if m:
            current['inputs'].append({'checkpoint': m.group(1), 'path': m.group(2)})
    return methods


def combo(path_text: str):
    m = re.search(r'evaluation_matrix/([^/]+)/([^/]+)/', path_text)
    if m:
        return m.group(1), m.group(2)
    if 'run10_stage1b_summary' in path_text:
        return 'run10_dbp_only', 'pair_audit'
    if 'run12_full_policy' in path_text:
        return 'run12_full', 'pair_audit'
    if 'run13_nopv_policy' in path_text:
        return 'run13_nopv', 'pair_audit'
    return None, None


def combos_for(method):
    out = OrderedDict()
    for item in method['inputs']:
        run, mode = combo(item['path'])
        if run and mode:
            out.setdefault((run, mode), []).append(item)
    return out


def sort_key(row):
    return (
        -num(row.get('avg_success')),
        -num(row.get('avg_reward')),
        -num(row.get('avg_route_completion')),
        num(row.get('step')) if math.isfinite(num(row.get('step'))) else 1e18,
        natural_key(row.get('policy', '')),
    )


def table_row(row):
    policy = row.get('policy', '')
    if '/' in policy:
        policy = policy.split('/', 1)[1]
    return [
        fmt(row.get('mode')),
        fmt(policy),
        fmt(row.get('step')),
        fmt(row.get('avg_success')),
        fmt(row.get('avg_crash')),
        fmt(row.get('avg_out_of_route')),
        fmt(row.get('avg_timeout')),
        fmt(row.get('avg_route_completion')),
        fmt(row.get('avg_reward')),
        fmt(row.get('avg_final_gate_delta_c_mean')),
        fmt(row.get('avg_final_gate_accept_rate')),
        fmt(row.get('avg_inference_latency_mean')),
    ]


def policy_rows(summary, run, mode):
    prefix = run + '/'
    return sorted([r for r in summary if r.get('mode') == mode and r.get('policy', '').startswith(prefix)], key=sort_key)


def checkpoint_from_pair_path(path: Path):
    m = re.search(r'policy-(\d+)-', path.name)
    if m:
        return m.group(1)
    if 'stage1b' in path.name:
        return 'stage1b'
    return '--'


def pair_paths(run):
    if run == 'run10_dbp_only':
        return [DBP_PAIR]
    if run == 'run12_full':
        return sorted(PAIR_DIR.glob('run12_full_policy-*_summary.csv'), key=lambda p: natural_key(p.name))
    if run == 'run13_nopv':
        return sorted(PAIR_DIR.glob('run13_nopv_policy-*_summary.csv'), key=lambda p: natural_key(p.name))
    return []


def pair_table_rows(run):
    rows = []
    for path in pair_paths(run):
        for r in read_csv(path):
            rows.append([
                fmt(str(path.relative_to(ROOT))),
                fmt(r.get('variant')),
                fmt(checkpoint_from_pair_path(path)),
                fmt(r.get('n_pairs')),
                fmt(r.get('pairsat_e')),
                fmt(r.get('v_e')),
                fmt(r.get('iar')),
                fmt(r.get('ragr')),
                fmt(r.get('energy_gap_neg_minus_pos')),
                fmt(r.get('min_dist_to_rejected_mean')),
                fmt(r.get('er_margin')),
                fmt(r.get('ragr_delta')),
                fmt(r.get('ragr_k')),
            ])
    return rows


def render_table(table_name, out_name, script, summary):
    methods = parse_methods(script)
    out = [
        f'# {table_name} policy summary tables',
        '',
        f'- Source script: `{script.relative_to(ROOT)}`',
        f'- Summary source for policy tables: `{POLICY_SUMMARY.relative_to(ROOT)}`',
        '- Each method below uses the run/mode implied by the input CSV paths echoed in the shell script, then lists all matching policies from the summary source.',
        '',
    ]
    for method in methods:
        out += [f"## {method['idx']}. {method['label']}", '']
        if method['manifest']:
            out.append(f"- Row manifest: `{method['manifest']}`")
        echoed = ', '.join(i['checkpoint'] for i in method['inputs'])
        if echoed:
            out.append(f'- Checkpoints explicitly echoed in script: {echoed}')
        out.append('')
        for (run, mode), items in combos_for(method).items():
            if mode == 'pair_audit':
                rows = pair_table_rows(run)
                out += [f'### `{run}` pair audit summaries', '', f'- Rows: {len(rows)}', '', md_table(PAIR_COLS, rows), '']
                continue
            rows = policy_rows(summary, run, mode)
            out += [
                f'### `{run}` / `{mode}`',
                '',
                f'- Directory represented by script input: `eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/{run}/{mode}/`',
                f'- Policies listed: {len(rows)}',
                '',
                md_table(TABLE_COLS, [table_row(r) for r in rows]),
                '',
            ]
    target = OUT_DIR / out_name
    target.write_text('\n'.join(out).rstrip() + '\n', encoding='utf-8')
    return target, len(methods)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = read_csv(POLICY_SUMMARY)
    index = []
    for table_name, out_name, script in TABLES:
        target, method_count = render_table(table_name, out_name, script, summary)
        index.append([table_name, f'`{target.name}`', str(method_count)])
        print(f'wrote {target.relative_to(ROOT)}')
    readme = ['# Table repro compact policy tables', '', 'These five markdown files are grouped by the method labels echoed in TABLE IV-VIII scripts. Each method lists all policies in the matching run/mode directory using the compact table columns requested by the user.', '', md_table(['table', 'markdown', 'methods'], index), '']
    (OUT_DIR / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print(f'wrote {(OUT_DIR / "README.md").relative_to(ROOT)}')

if __name__ == '__main__':
    main()
