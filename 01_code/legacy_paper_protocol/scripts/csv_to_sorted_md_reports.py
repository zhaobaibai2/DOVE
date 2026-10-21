#!/usr/bin/env python3
"""Generate readable Markdown reports for CSV files in data0528 log dirs."""
from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path('/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508')
TARGET_DIRS = [ROOT / 'logs/data0528--doc', ROOT / 'logs/data0528-latex3']

MODE_LABELS = {
    'vanilla': 'Vanilla / w/o GDS',
    'upvgds_no_gate': 'UPV-GDS w/o Gate',
    'upvgds_final_gate_full': 'Full UPV-GDS + Final Gate',
    'final_gate_no_disagreement': 'w/o disagreement gate',
    'final_gate_proxy_only': 'Proxy-only gate',
    'final_gate_prior_only': 'Prior-only gate',
    'topk_rerank': 'Top-K rerank',
    'topk_rerank_single_policy': 'Top-K rerank',
}

METRIC_LABELS = {
    'avg_success': 'Succ.', 'success_mean': 'Succ.', 'success_pm_std': 'Succ.',
    'avg_crash': 'Coll.', 'crash_mean': 'Coll.', 'crash_pm_std': 'Coll.',
    'avg_out_of_route': 'OOR', 'out_of_route_mean': 'OOR', 'out_of_route_pm_std': 'OOR',
    'avg_timeout': 'Timeout', 'timeout_mean': 'Timeout', 'timeout_pm_std': 'Timeout',
    'avg_route_completion': 'Route', 'route_completion_mean': 'Route', 'route_completion_pm_std': 'Route',
    'avg_reward': 'Reward', 'reward_mean': 'Reward', 'reward_pm_std': 'Reward',
    'avg_cost': 'Cost', 'cost_mean': 'Cost', 'cost_pm_std': 'Cost',
    'avg_final_gate_delta_c_mean': 'Delta C', 'final_gate_delta_c_mean_mean': 'Delta C', 'final_gate_delta_c_mean_pm_std': 'Delta C',
    'avg_final_gate_accept_rate': 'GateAcc', 'final_gate_accept_rate_mean': 'GateAcc', 'final_gate_accept_rate_pm_std': 'GateAcc',
    'avg_inference_latency_mean': 'Latency', 'inference_latency_mean_mean': 'Latency', 'inference_latency_mean_pm_std': 'Latency',
}

KEY_CANDIDATES = [
    'mode', 'mode_display', 'policy', 'policy_basename', 'step', 'n_episodes',
    'avg_success', 'success_pm_std', 'success_mean',
    'avg_crash', 'crash_pm_std', 'crash_mean',
    'avg_out_of_route', 'out_of_route_pm_std', 'out_of_route_mean',
    'avg_timeout', 'timeout_pm_std', 'timeout_mean',
    'avg_route_completion', 'route_completion_pm_std', 'route_completion_mean',
    'avg_reward', 'reward_pm_std', 'reward_mean',
    'avg_cost', 'cost_pm_std', 'cost_mean',
    'avg_final_gate_delta_c_mean', 'final_gate_delta_c_mean_pm_std', 'final_gate_delta_c_mean_mean',
    'avg_final_gate_accept_rate', 'final_gate_accept_rate_pm_std', 'final_gate_accept_rate_mean',
    'avg_inference_latency_mean', 'inference_latency_mean_pm_std', 'inference_latency_mean_mean',
]

LOWER_IS_BETTER = {'avg_crash', 'avg_out_of_route', 'avg_timeout', 'avg_cost', 'avg_inference_latency_mean',
                   'crash_mean', 'out_of_route_mean', 'timeout_mean', 'cost_mean', 'inference_latency_mean_mean'}

CATEGORY_PATTERNS = [
    ('Identity / source', re.compile(r'^(mode|mode_display|policy|policy_basename|step|n_episodes|source_csv|source_episode_csv)$')),
    ('Outcome / task', re.compile(r'(success|crash|out_of_route|timeout|route_completion|reward|cost|native_cost|closed_loop_distance|length)')),
    ('Intervention / takeover', re.compile(r'(takeover|intervention|autonomous|first_takeover|time_to_first)')),
    ('Vehicle dynamics', re.compile(r'(lateral|speed|accel|jerk|steering|action_saturation)')),
    ('Latency', re.compile(r'(latency)')),
    ('Final gate', re.compile(r'(final_gate)')),
    ('Guidance', re.compile(r'(guidance)')),
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def as_float(value: str | None) -> float:
    if value is None or value == '':
        return float('nan')
    try:
        x = float(value)
    except Exception:
        return float('nan')
    return x if math.isfinite(x) else float('nan')


def natural_key(value: str) -> list[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]


def row_sort_key(row: dict[str, str]) -> tuple:
    mode = row.get('mode_display') or MODE_LABELS.get(row.get('mode', ''), row.get('mode', ''))
    success = as_float(row.get('avg_success') or row.get('success_mean'))
    reward = as_float(row.get('avg_reward') or row.get('reward_mean'))
    route = as_float(row.get('avg_route_completion') or row.get('route_completion_mean'))
    step = as_float(row.get('step'))
    policy = row.get('policy_basename') or row.get('policy', '')
    return (natural_key(mode), -(success if math.isfinite(success) else -1), -(reward if math.isfinite(reward) else -1e300), -(route if math.isfinite(route) else -1), step if math.isfinite(step) else 1e18, natural_key(policy))


def mode_label(row: dict[str, str]) -> str:
    return row.get('mode_display') or MODE_LABELS.get(row.get('mode', ''), row.get('mode', 'unknown'))


def fmt_value(value: str, col: str) -> str:
    if value is None or value == '':
        return '--'
    s = str(value).strip()
    if '$\\pm$' in s or '$\\pm$' in s or '\\pm' in s:
        return s
    x = as_float(s)
    if s.lower() in {'nan', 'none', 'null'}:
        return '--'
    if math.isfinite(x):
        if col in {'step', 'n_episodes'} or col.endswith('_n') or col.endswith('_count'):
            return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f'{x:.3f}'
        if abs(x) >= 1000:
            return f'{x:.3f}'
        if abs(x) >= 10:
            return f'{x:.3f}'
        return f'{x:.4f}'.rstrip('0').rstrip('.') if abs(x) >= 0.001 else f'{x:.3g}'
    if col in {'policy', 'policy_basename'}:
        return f'`{s}`'
    if col.startswith('source'):
        return f'`{s}`'
    return s.replace('|', '\\|')


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ''
    aligns = []
    for h in headers:
        aligns.append('---:' if h in {'rank', 'n', 'step', 'n_episodes'} or h.endswith('_n') else '---')
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(aligns) + ' |']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return '\n'.join(out)


def pretty_header(col: str) -> str:
    return METRIC_LABELS.get(col, col)


def compact_key_table(fieldnames: list[str], rows: list[dict[str, str]], limit: int | None = None) -> str:
    cols = [c for c in KEY_CANDIDATES if c in fieldnames]
    if not cols:
        cols = fieldnames[:min(12, len(fieldnames))]
    display_headers = [pretty_header(c) for c in cols]
    body = []
    selected = rows[:limit] if limit else rows
    for r in selected:
        body.append([fmt_value(r.get(c, ''), c) for c in cols])
    return md_table(display_headers, body)


def best_rows_by_mode(fieldnames: list[str], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    for r in sorted(rows, key=lambda x: natural_key(mode_label(x))):
        grouped.setdefault(mode_label(r), []).append(r)
    best = []
    for group_rows in grouped.values():
        best.append(sorted(group_rows, key=row_sort_key)[0])
    return best


def categorize_columns(fieldnames: list[str]) -> OrderedDict[str, list[str]]:
    cats: OrderedDict[str, list[str]] = OrderedDict((name, []) for name, _ in CATEGORY_PATTERNS)
    cats['Other'] = []
    for col in fieldnames:
        placed = False
        for name, pat in CATEGORY_PATTERNS:
            if pat.search(col):
                cats[name].append(col)
                placed = True
                break
        if not placed:
            cats['Other'].append(col)
    return OrderedDict((k, v) for k, v in cats.items() if v)


def category_note(cats: OrderedDict[str, list[str]]) -> str:
    rows = []
    for name, cols in cats.items():
        rows.append([name, str(len(cols)), ', '.join(f'`{c}`' for c in cols)])
    return md_table(['category', 'columns', 'fields'], rows)


def table_for_columns(cols: list[str], rows: list[dict[str, str]], fieldnames: list[str]) -> str:
    identity = [c for c in ['mode_display', 'mode', 'policy_basename', 'policy', 'step', 'n_episodes'] if c in fieldnames]
    use_cols = []
    for c in identity + cols:
        if c not in use_cols:
            use_cols.append(c)
    body = [[fmt_value(r.get(c, ''), c) for c in use_cols] for r in rows]
    return md_table([pretty_header(c) for c in use_cols], body)


def render_report(path: Path) -> str:
    fieldnames, rows = read_csv(path)
    rows = sorted(rows, key=row_sort_key)
    cats = categorize_columns(fieldnames)
    rel = path.relative_to(ROOT)
    title = path.stem.replace('_', ' ')
    out: list[str] = []
    out.append(f'# {title}')
    out.append('')
    out.append(f'- Source CSV: `{rel}`')
    out.append(f'- Rows: {len(rows)}')
    out.append(f'- Columns: {len(fieldnames)}')
    if 'mode' in fieldnames:
        modes = sorted({mode_label(r) for r in rows}, key=natural_key)
        out.append(f'- Modes: {len(modes)}')
    out.append('')
    out.append('## Column categories')
    out.append('')
    out.append(category_note(cats))
    out.append('')

    if rows and 'mode' in fieldnames:
        out.append('## Best row per mode and run')
        out.append('')
        out.append(compact_key_table_with_run(fieldnames, best_rows_by_mode_and_run(fieldnames, rows, path)))
        out.append('')

    out.append('## Sorted summary')
    out.append('')
    if 'mode' in fieldnames:
        grouped_runs = grouped_by_mode_and_run(rows, path)
        for mode_name, runs in grouped_runs.items():
            out.append(f'### {mode_name}')
            out.append('')
            for run_name, group_rows in runs.items():
                out.append(f'#### {run_name}')
                out.append('')
                out.append(compact_key_table(fieldnames, group_rows))
                out.append('')
        grouped = OrderedDict((f'{mode_name} / {run_name}', group_rows) for mode_name, runs in grouped_runs.items() for run_name, group_rows in runs.items())
    else:
        out.append(compact_key_table(fieldnames, rows))
        out.append('')
        grouped = OrderedDict([('All rows', rows)])

    if 'mode' in fieldnames:
        out.append('## Rows by mode and run')
        out.append('')
        for mode_name, runs in grouped_by_mode_and_run(rows, path).items():
            out.append(f'### {mode_name}')
            out.append('')
            for run_name, group_rows in runs.items():
                out.append(f'#### {run_name}')
                out.append('')
                out.append(compact_key_table(fieldnames, group_rows))
                out.append('')

    out.append('## All CSV fields')
    out.append('')
    out.append('Every source column is included below. Wide CSVs are split by field family with identity columns repeated for readability.')
    out.append('')
    for cat_name, cols in cats.items():
        if cat_name == 'Identity / source':
            continue
        out.append(f'### {cat_name}')
        out.append('')
        for label, group_rows in grouped.items():
            if len(grouped) > 1:
                out.append(f'#### {label}')
                out.append('')
            out.append(table_for_columns(cols, group_rows, fieldnames))
            out.append('')
    if 'Identity / source' in cats:
        out.append('### Identity / source')
        out.append('')
        out.append(table_for_columns(cats['Identity / source'], rows, fieldnames))
        out.append('')
    return '\n'.join(out).rstrip() + '\n'




def run_group(row: dict[str, str], source_path: Path | None = None) -> str:
    policy = row.get('policy', '')
    if '/' in policy:
        return policy.split('/', 1)[0]
    source = row.get('source_csv') or row.get('source_episode_csv') or ''
    match = re.search(r'(run\d+_[^/]+)', source)
    if match:
        return match.group(1)
    if source_path and 'data0528--doc' in str(source_path):
        parent = source_path.parent.name
        return parent
    return 'single_run'


def grouped_by_mode_and_run(rows: list[dict[str, str]], source_path: Path | None = None) -> OrderedDict[str, OrderedDict[str, list[dict[str, str]]]]:
    grouped: OrderedDict[str, OrderedDict[str, list[dict[str, str]]]] = OrderedDict()
    for r in rows:
        mode = mode_label(r)
        run = run_group(r, source_path)
        grouped.setdefault(mode, OrderedDict()).setdefault(run, []).append(r)
    ordered: OrderedDict[str, OrderedDict[str, list[dict[str, str]]]] = OrderedDict()
    for mode in sorted(grouped.keys(), key=natural_key):
        ordered[mode] = OrderedDict()
        for run in sorted(grouped[mode].keys(), key=natural_key):
            ordered[mode][run] = sorted(grouped[mode][run], key=row_sort_key)
    return ordered


def best_rows_by_mode_and_run(fieldnames: list[str], rows: list[dict[str, str]], source_path: Path | None = None) -> list[dict[str, str]]:
    best: list[dict[str, str]] = []
    for mode, runs in grouped_by_mode_and_run(rows, source_path).items():
        for run, group_rows in runs.items():
            row = sorted(group_rows, key=row_sort_key)[0].copy()
            row['_run_group'] = run
            best.append(row)
    return best


def compact_key_table_with_run(fieldnames: list[str], rows: list[dict[str, str]]) -> str:
    cols = ['_run_group'] + [c for c in KEY_CANDIDATES if c in fieldnames]
    headers = ['run'] + [pretty_header(c) for c in cols[1:]]
    body = [[fmt_value(r.get(c, ''), c) for c in cols] for r in rows]
    return md_table(headers, body)

def write_indexes(csvs: list[Path]) -> None:
    by_dir: OrderedDict[Path, list[Path]] = OrderedDict()
    for csv_path in csvs:
        by_dir.setdefault(csv_path.parent, []).append(csv_path)

    for directory, paths in by_dir.items():
        rows = []
        for csv_path in sorted(paths, key=lambda p: natural_key(p.name)):
            fieldnames, data = read_csv(csv_path)
            cats = categorize_columns(fieldnames)
            md_name = csv_path.with_suffix('.md').name
            rows.append([
                f'`{csv_path.name}`',
                f'`{md_name}`',
                str(len(data)),
                str(len(fieldnames)),
                ', '.join(cats.keys()),
            ])
        out = [
            '# CSV Markdown report index',
            '',
            f'- Directory: `{directory.relative_to(ROOT)}`',
            f'- CSV files: {len(paths)}',
            '',
            md_table(['CSV', 'Markdown', 'rows', 'columns', 'categories'], rows),
            '',
        ]
        (directory / 'README.md').write_text('\n'.join(out), encoding='utf-8')
        print(f'wrote {(directory / "README.md").relative_to(ROOT)}')

def main() -> None:
    csvs = []
    for d in TARGET_DIRS:
        csvs.extend(sorted(d.rglob('*.csv'), key=lambda p: natural_key(str(p.relative_to(ROOT)))))
    for csv_path in csvs:
        md_path = csv_path.with_suffix('.md')
        md_path.write_text(render_report(csv_path), encoding='utf-8')
        print(f'wrote {md_path.relative_to(ROOT)}')
    write_indexes(csvs)

if __name__ == '__main__':
    main()
