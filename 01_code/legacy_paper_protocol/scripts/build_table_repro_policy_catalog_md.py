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
EPISODE_SUMMARY = ROOT / 'logs/data0528-latex3/all_checkpoints_episode_mean_std.csv'
PAIR_DIR = ROOT / 'logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151'
DBP_PAIR = ROOT / 'logs/99_every_eval_script_and_pair_diag_seed104_20260524_092640/pair_diag/run10_stage1b_summary.csv'

TABLE_SCRIPTS = [
    ('TABLE IV', 'TABLE_IV_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE IV.sh'),
    ('TABLE V', 'TABLE_V_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE V.sh'),
    ('TABLE VI', 'TABLE_VI_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VI.sh'),
    ('TABLE VII', 'TABLE_VII_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VII.sh'),
    ('TABLE VIII', 'TABLE_VIII_full_raw_data.md', ROOT / 'scripts/table_repro/TABLE VIII.sh'),
]

COMPACT_COLS_POLICY = [
    'mode', 'policy', 'step', 'avg_success', 'avg_reward', 'avg_crash', 'avg_out_of_route', 'avg_timeout',
    'avg_route_completion', 'avg_cost', 'avg_closed_loop_distance', 'avg_final_gate_accept_rate',
    'avg_final_gate_delta_c_mean', 'avg_inference_latency_mean', 'source_csv',
]
COMPACT_COLS_EPISODE = [
    'mode', 'mode_display', 'policy', 'policy_basename', 'step', 'n_episodes', 'success_pm_std',
    'reward_pm_std', 'crash_pm_std', 'out_of_route_pm_std', 'timeout_pm_std', 'route_completion_pm_std',
    'cost_pm_std', 'closed_loop_distance_pm_std', 'final_gate_accept_rate_pm_std',
    'final_gate_delta_c_mean_pm_std', 'inference_latency_mean_pm_std', 'source_episode_csv',
]
COMPACT_COLS_PAIR = ['variant', 'source_csv', 'n_pairs', 'pairsat_e', 'v_e', 'iar', 'ragr', 'energy_gap_neg_minus_pos', 'min_dist_to_rejected_mean', 'er_margin', 'ragr_delta', 'ragr_k']

CATEGORY_PATTERNS = [
    ('Identity / source', re.compile(r'^(mode|mode_display|policy|policy_basename|step|n_episodes|source_csv|source_episode_csv|variant|n_pairs|source_file|checkpoint|method|row_id)$')),
    ('Outcome / task', re.compile(r'(success|crash|out_of_route|timeout|route_completion|reward|cost|native_cost|takeover_cost|closed_loop_distance|length)')),
    ('Intervention / takeover', re.compile(r'(takeover|intervention|autonomous|first_takeover|time_to_first)')),
    ('Vehicle dynamics', re.compile(r'(lateral|speed|accel|jerk|steering|action_saturation)')),
    ('Latency', re.compile(r'(latency)')),
    ('Final gate', re.compile(r'(final_gate)')),
    ('Guidance', re.compile(r'(guidance)')),
    ('Pair audit / EnergyRank', re.compile(r'(pair|pairsat|v_e|iar|ragr|energy|margin|min_dist|rejected|neg_minus_pos)')),
]

MODE_LABELS = {
    'vanilla': 'Vanilla / w/o GDS',
    'upvgds_no_gate': 'UPV-GDS w/o Gate',
    'upvgds_final_gate_full': 'Full UPV-GDS + Final Gate',
    'final_gate_no_disagreement': 'w/o disagreement gate',
    'final_gate_proxy_only': 'Proxy-only gate',
    'final_gate_prior_only': 'Prior-only gate',
    'topk_rerank': 'Top-K rerank',
}

def natural_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]

def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames or []), list(r)

def num(value):
    try:
        x = float(value) if value not in (None, '') else float('nan')
    except Exception:
        return float('nan')
    return x if math.isfinite(x) else float('nan')

def sort_perf(row):
    success = num(row.get('avg_success') or row.get('success_mean'))
    reward = num(row.get('avg_reward') or row.get('reward_mean'))
    route = num(row.get('avg_route_completion') or row.get('route_completion_mean'))
    step = num(row.get('step'))
    policy = row.get('policy') or row.get('policy_basename') or row.get('source_csv', '')
    return (-(success if math.isfinite(success) else -1), -(reward if math.isfinite(reward) else -1e300), -(route if math.isfinite(route) else -1), step if math.isfinite(step) else 1e18, natural_key(policy))

def esc(value):
    if value is None:
        return '--'
    s = str(value).strip()
    if not s or s.lower() in {'nan', 'none', 'null'}:
        return '--'
    try:
        x = float(s)
    except Exception:
        x = None
    if x is not None and math.isfinite(x):
        if abs(x - round(x)) < 1e-10 and abs(x) < 1000000:
            return str(int(round(x)))
        if abs(x) >= 100:
            return f'{x:.3f}'
        if abs(x) >= 1:
            return f'{x:.4f}'.rstrip('0').rstrip('.')
        return f'{x:.5f}'.rstrip('0').rstrip('.') if abs(x) >= 1e-4 else f'{x:.4g}'
    s = s.replace('\\', '\\\\').replace('|', '\\|').replace('\n', ' ')
    if '/' in s or s.endswith('.csv') or s.endswith('.pkl'):
        return f'`{s}`'
    return s

def md_table(cols, rows):
    if not cols:
        return ''
    out = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join('---:' if c in {'step', 'n_episodes', 'n_pairs'} or c.endswith('_n') else '---' for c in cols) + ' |']
    out.extend('| ' + ' | '.join(esc(r.get(c, '')) for c in cols) + ' |' for r in rows)
    return '\n'.join(out)

def category_cols(cols):
    cats = OrderedDict((name, []) for name, _ in CATEGORY_PATTERNS)
    cats['Other'] = []
    for c in cols:
        for name, pat in CATEGORY_PATTERNS:
            if pat.search(c):
                cats[name].append(c)
                break
        else:
            cats['Other'].append(c)
    return OrderedDict((k, v) for k, v in cats.items() if v)

def category_table(cols):
    rows = [{ 'category': name, 'columns': str(len(c)), 'fields': ', '.join(f'`{x}`' for x in c)} for name, c in category_cols(cols).items()]
    return md_table(['category', 'columns', 'fields'], rows)

def parse_table_script(path: Path):
    methods = []
    current = None
    for line in path.read_text(encoding='utf-8').splitlines():
        m = re.search(r'echo\s+"(\d+)\.\s+(.+?)"', line)
        if m:
            current = {'index': m.group(1), 'label': m.group(2), 'manifests': [], 'inputs': []}
            methods.append(current)
            continue
        if current is None:
            continue
        m = re.search(r'row manifest:\s*([^" ]+)', line)
        if m:
            current['manifests'].append(m.group(1))
        m = re.search(r'input csv \[checkpoint ([^\]]+)\]:\s*([^" ]+)', line)
        if m:
            current['inputs'].append({'checkpoint': m.group(1), 'path': m.group(2)})
    return methods

def combo_from_input(path_text):
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

def method_combos(method):
    seen = OrderedDict()
    for item in method['inputs']:
        run, mode = combo_from_input(item['path'])
        if run and mode:
            seen.setdefault((run, mode), []).append(item)
    return seen

def filter_rows(rows, run, mode):
    prefix = run + '/'
    return sorted([r for r in rows if r.get('mode') == mode and r.get('policy', '').startswith(prefix)], key=sort_perf)

def pair_rows_for(run):
    paths = []
    if run == 'run10_dbp_only':
        paths = [DBP_PAIR]
    elif run == 'run12_full':
        paths = sorted(PAIR_DIR.glob('run12_full_policy-*_summary.csv'), key=lambda p: natural_key(p.name))
    elif run == 'run13_nopv':
        paths = sorted(PAIR_DIR.glob('run13_nopv_policy-*_summary.csv'), key=lambda p: natural_key(p.name))
    rows = []
    for p in paths:
        cols, data = read_csv(p)
        for r in data:
            r = dict(r)
            r['source_file'] = str(p.relative_to(ROOT))
            rows.append(r)
    return rows

def render_policy_dataset(title, cols, rows, compact_cols):
    out = [f'#### {title}', '', f'- Rows: {len(rows)}', f'- Columns: {len(cols)}', '', '##### Compact sorted view', '', md_table([c for c in compact_cols if c in cols], rows), '', '##### Field categories', '', category_table(cols), '', '##### Complete fields by category', '']
    identity = [c for c in ['mode', 'mode_display', 'policy', 'policy_basename', 'step', 'n_episodes', 'source_csv', 'source_episode_csv'] if c in cols]
    for cat, cat_cols in category_cols(cols).items():
        use = []
        for c in identity + cat_cols:
            if c not in use:
                use.append(c)
        out += [f'###### {cat}', '', md_table(use, rows), '']
    return out

def render_table(table_name, out_name, script_path, policy_cols, policy_rows, ep_cols, ep_rows):
    methods = parse_table_script(script_path)
    out = [f'# {table_name} full policy catalog', '', f'- Source table script: `{script_path.relative_to(ROOT)}`', '- This file is organized by the method names printed in the table reproduction script.', '- For each method, every policy in the matching `run + mode` from `logs/data0528-latex3` is included, not only the few checkpoints echoed by the script.', '']
    for method in methods:
        out += [f"## {method['index']}. {method['label']}", '']
        if method['manifests']:
            out += ['- Row manifest(s): ' + ', '.join(f'`{m}`' for m in method['manifests'])]
        if method['inputs']:
            listed = ', '.join(f"{x['checkpoint']}" for x in method['inputs'])
            out += [f'- Checkpoints explicitly echoed by script: {listed}', '']
        combos = method_combos(method)
        if not combos:
            out += ['No matching run/mode could be inferred.', '']
            continue
        for (run, mode), echoed in combos.items():
            out += [f"### data source: `{run}` / `{mode}`", '', '- Echoed input CSV examples:', '']
            for item in echoed:
                out.append(f"  - checkpoint `{item['checkpoint']}`: `{item['path']}`")
            out.append('')
            if mode == 'pair_audit':
                rows = pair_rows_for(run)
                rows = sorted(rows, key=lambda r: natural_key(r.get('source_file', '')))
                cols = []
                for r in rows:
                    for c in r:
                        if c not in cols:
                            cols.append(c)
                out += render_policy_dataset('All related pair-audit summary rows', cols, rows, COMPACT_COLS_PAIR + ['source_file'])
            else:
                p_rows = filter_rows(policy_rows, run, mode)
                e_rows = filter_rows(ep_rows, run, mode)
                out += render_policy_dataset('All policies from all_checkpoints_policy_mode_summary.csv', policy_cols, p_rows, COMPACT_COLS_POLICY)
                out += render_policy_dataset('All policies from all_checkpoints_episode_mean_std.csv', ep_cols, e_rows, COMPACT_COLS_EPISODE)
    target = OUT_DIR / out_name
    target.write_text('\n'.join(out).rstrip() + '\n', encoding='utf-8')
    return target, methods

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    policy_cols, policy_rows = read_csv(POLICY_SUMMARY)
    ep_cols, ep_rows = read_csv(EPISODE_SUMMARY)
    index_rows = []
    for table, out_name, script in TABLE_SCRIPTS:
        target, methods = render_table(table, out_name, script, policy_cols, policy_rows, ep_cols, ep_rows)
        index_rows.append({'table': table, 'markdown': target.name, 'methods': str(len(methods))})
        print(f'wrote {target.relative_to(ROOT)}')
    readme = ['# Table repro full policy catalog index', '', 'Each markdown file is grouped by the method labels echoed in TABLE IV-VIII scripts. Under each method, rows are expanded to the full matching run/mode catalog from `logs/data0528-latex3`, so you can choose policies later.', '', md_table(['table', 'markdown', 'methods'], index_rows), '']
    (OUT_DIR / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print(f'wrote {(OUT_DIR / "README.md").relative_to(ROOT)}')

if __name__ == '__main__':
    main()
