#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path('/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508')
OUT_DIR = ROOT / 'logs/data0528-latex3/table_repro_full_raw_data'
OLD_ROOTS = [
    '/root/autodl-tmp/dacer_pvp/test0508/',
    '/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/',
]

TABLES = [
    {
        'name': 'TABLE IV',
        'slug': 'TABLE_IV',
        'title': 'Table IV main task full raw data',
        'script': 'scripts/table_repro/TABLE IV.sh',
        'manifest': 'latex3/table_inputs/main_task_rows.csv',
        'computed': 'latex3/table_outputs/main_task_rows_computed.csv',
        'tex_rows': 'latex3/table_outputs/main_task_rows_tex_rows.tex',
        'trend': 'latex3/table_outputs/main_task_rows_trend_audit.md',
        'source_key': 'source_episode_csv',
    },
    {
        'name': 'TABLE V',
        'slug': 'TABLE_V',
        'title': 'Table V training-side ablation full raw data',
        'script': 'scripts/table_repro/TABLE V.sh',
        'manifest': 'latex3/table_inputs/train_ablation_rows.csv',
        'computed': 'latex3/table_outputs/train_ablation_rows_computed.csv',
        'tex_rows': 'latex3/table_outputs/train_ablation_rows_tex_rows.tex',
        'trend': 'latex3/table_outputs/train_ablation_rows_trend_audit.md',
        'source_key': 'source_episode_csv',
    },
    {
        'name': 'TABLE VI',
        'slug': 'TABLE_VI',
        'title': 'Table VI test-time ablation full raw data',
        'script': 'scripts/table_repro/TABLE VI.sh',
        'manifest': 'latex3/table_inputs/testtime_ablation_rows.csv',
        'computed': 'latex3/table_outputs/testtime_ablation_rows_computed.csv',
        'tex_rows': 'latex3/table_outputs/testtime_ablation_rows_tex_rows.tex',
        'trend': 'latex3/table_outputs/testtime_ablation_rows_trend_audit.md',
        'source_key': 'source_episode_csv',
    },
    {
        'name': 'TABLE VII',
        'slug': 'TABLE_VII',
        'title': 'Table VII deployment diagnostics full raw data',
        'script': 'scripts/table_repro/TABLE VII.sh',
        'manifest': 'latex3/table_inputs/diag_rows.csv',
        'computed': 'latex3/table_outputs/diag_rows_computed.csv',
        'tex_rows': 'latex3/table_outputs/diag_rows_tex_rows.tex',
        'trend': 'latex3/table_outputs/diag_rows_trend_audit.md',
        'source_key': 'source_episode_csv',
    },
    {
        'name': 'TABLE VIII',
        'slug': 'TABLE_VIII',
        'title': 'Table VIII actor-prior pair audit full raw data',
        'script': 'scripts/table_repro/TABLE VIII.sh',
        'manifest': 'latex3/table_inputs/pair_audit_rows.csv',
        'computed': 'latex3/table_outputs/pair_audit_rows_computed.csv',
        'tex_rows': 'latex3/table_outputs/pair_audit_rows_tex_rows.tex',
        'trend': 'latex3/table_outputs/pair_audit_rows_trend_audit.md',
        'source_key': 'source_summary_csv',
    },
]

CATEGORY_PATTERNS = [
    ('Identity / source', re.compile(r'^(table|row_id|display|checkpoint|source_csv|source_episode_csv|source_summary_csv|source_file|source_row|policy_name|policy_path|policy|policy_basename|variant|mode|mode_display|step|map_seed|episode_idx|reason)$')),
    ('Outcome / task', re.compile(r'(success|crash|out_of_route|timeout|route_completion|reward|cost|native_cost|takeover_cost|closed_loop_distance|length)')),
    ('Intervention / takeover', re.compile(r'(takeover|intervention|autonomous|first_takeover|time_to_first)')),
    ('Vehicle dynamics', re.compile(r'(lateral|speed|accel|jerk|steering|action_saturation)')),
    ('Latency', re.compile(r'(latency)')),
    ('Final gate', re.compile(r'(final_gate)')),
    ('Guidance', re.compile(r'(guidance)')),
    ('Pair audit / EnergyRank', re.compile(r'(pair|pairsat|v_e|iar|ragr|energy|margin|min_dist|rejected|neg_minus_pos)')),
]

BRIEF_COLS = [
    'display', 'checkpoint', 'source_file', 'source_row', 'policy_name', 'policy', 'variant', 'mode', 'step', 'episode_idx',
    'success', 'crash', 'out_of_route', 'timeout', 'route_completion', 'reward', 'cost',
    'final_gate_delta_c_mean', 'final_gate_accept_rate', 'inference_latency_mean',
    'n_pairs', 'pairsat_e', 'v_e', 'iar', 'ragr', 'energy_gap_neg_minus_pos',
]


def natural_key(value):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', str(value))]


def resolve(raw: str) -> Path:
    text = (raw or '').strip()
    for old in OLD_ROOTS:
        if text.startswith(old):
            text = text[len(old):]
            break
    p = Path(text)
    if not p.is_absolute():
        p = ROOT / p
    return p


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def fnum(value: str | None) -> float:
    try:
        x = float(value) if value not in (None, '') else float('nan')
    except Exception:
        return float('nan')
    return x if math.isfinite(x) else float('nan')


def sort_row(row: dict[str, str]):
    display = row.get('display', '')
    mode = row.get('mode', '')
    checkpoint = row.get('checkpoint', '') or row.get('step', '')
    ep = row.get('episode_idx', '') or row.get('source_row', '')
    success = fnum(row.get('success') or row.get('success_mean') or row.get('avg_success'))
    reward = fnum(row.get('reward') or row.get('reward_mean') or row.get('avg_reward'))
    return (natural_key(display), natural_key(mode), natural_key(checkpoint), -(success if math.isfinite(success) else -1), -(reward if math.isfinite(reward) else -1e300), natural_key(ep))


def esc(value: object) -> str:
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
        if abs(x - round(x)) < 1e-9 and abs(x) < 1000000:
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


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ''
    align = ['---:' if h in {'source_row', 'episode_idx', 'step', 'checkpoint', 'n_pairs'} or h.endswith('_n') else '---' for h in headers]
    out = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(align) + ' |']
    out.extend('| ' + ' | '.join(r) + ' |' for r in rows)
    return '\n'.join(out)


def categorize(cols: list[str]) -> OrderedDict[str, list[str]]:
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


def category_table(cats: OrderedDict[str, list[str]]) -> str:
    rows = [[name, str(len(cols)), ', '.join(f'`{c}`' for c in cols)] for name, cols in cats.items()]
    return md_table(['category', 'columns', 'fields'], rows)


def rows_table(cols: list[str], rows: list[dict[str, str]]) -> str:
    return md_table(cols, [[esc(r.get(c, '')) for c in cols] for r in rows])


def brief_cols(cols: list[str]) -> list[str]:
    out = [c for c in BRIEF_COLS if c in cols]
    if not out:
        out = cols[:min(12, len(cols))]
    return out


def prefixed_raw_rows(spec: dict[str, str], source_key: str) -> tuple[list[str], list[dict[str, str]], Path]:
    source = resolve(spec[source_key])
    header, rows = read_csv(source)
    meta_cols = ['display', 'row_id', 'checkpoint', 'source_file', 'source_row']
    out_rows = []
    for i, row in enumerate(rows, 1):
        enriched = {
            'display': spec.get('display', ''),
            'row_id': spec.get('row_id', ''),
            'checkpoint': spec.get('checkpoint', ''),
            'source_file': str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
            'source_row': str(i),
        }
        enriched.update(row)
        out_rows.append(enriched)
    return meta_cols + [c for c in header if c not in meta_cols], out_rows, source


def render_csv_file(title: str, path: Path) -> list[str]:
    if not path.exists():
        return [f'### {title}', '', f'Missing file: `{path.relative_to(ROOT)}`', '']
    cols, rows = read_csv(path)
    rows = sorted(rows, key=sort_row)
    cats = categorize(cols)
    out = [f'### {title}', '', f'- File: `{path.relative_to(ROOT)}`', f'- Rows: {len(rows)}', f'- Columns: {len(cols)}', '', '#### Field categories', '', category_table(cats), '', '#### Compact view', '', rows_table(brief_cols(cols), rows), '']
    out.append('#### Complete fields')
    out.append('')
    for name, cat_cols in cats.items():
        out.append(f'##### {name}')
        out.append('')
        out.append(rows_table(cat_cols, rows))
        out.append('')
    return out


def render_table(config: dict[str, str]) -> str:
    manifest_path = ROOT / config['manifest']
    manifest_cols, specs = read_csv(manifest_path)
    source_key = config['source_key']
    specs = sorted(specs, key=sort_row)
    out: list[str] = []
    out += [f"# {config['title']}", '', f"- Table script: `{config['script']}`", f"- Manifest: `{config['manifest']}`", f"- Computed aggregation: `{config['computed']}`", f"- Output generated from manifest source CSVs; raw sections preserve every original source column.", '']

    out += ['## Manifest rows', '', rows_table(manifest_cols, specs), '']
    out += ['## Table aggregation outputs', '']
    out += render_csv_file('Computed rows, all fields', ROOT / config['computed'])
    tex_path = ROOT / config['tex_rows']
    if tex_path.exists():
        out += ['### LaTeX table rows', '', '```tex', tex_path.read_text(encoding='utf-8').rstrip(), '```', '']
    trend_path = ROOT / config['trend']
    if trend_path.exists():
        out += ['### Trend audit', '', trend_path.read_text(encoding='utf-8').rstrip(), '']

    out += ['## Source CSV inventory', '']
    inv_rows = []
    total_raw = 0
    all_cols = []
    raw_by_spec = []
    for spec in specs:
        cols, rows, source = prefixed_raw_rows(spec, source_key)
        total_raw += len(rows)
        raw_by_spec.append((spec, cols, rows, source))
        for c in cols:
            if c not in all_cols:
                all_cols.append(c)
        inv_rows.append([spec.get('display',''), spec.get('row_id',''), spec.get('checkpoint',''), str(len(rows)), str(len(cols)), f'`{source.relative_to(ROOT)}`'])
    out += [md_table(['display', 'row_id', 'checkpoint', 'rows', 'columns', 'source'], inv_rows), '', f'- Total raw rows included below: {total_raw}', '']

    out += ['## Compact raw data, grouped by table row and checkpoint', '']
    for spec, cols, rows, source in raw_by_spec:
        rows = sorted(rows, key=sort_row)
        out += [f"### {spec.get('display','')} / checkpoint {spec.get('checkpoint','')}", '', f"- Source: `{source.relative_to(ROOT)}`", f"- Rows: {len(rows)}", '', rows_table(brief_cols(cols), rows), '']

    out += ['## Complete raw data, all original columns', '', 'Each subsection below repeats `display`, `row_id`, `checkpoint`, `source_file`, and `source_row` so every raw row remains traceable to the exact CSV and table line.', '']
    for spec, cols, rows, source in raw_by_spec:
        rows = sorted(rows, key=sort_row)
        cats = categorize(cols)
        out += [f"### {spec.get('display','')} / checkpoint {spec.get('checkpoint','')}", '', f"- Source: `{source.relative_to(ROOT)}`", f"- Rows: {len(rows)}", f"- Columns: {len(cols)}", '', '#### Field categories', '', category_table(cats), '']
        identity = [c for c in ['display', 'row_id', 'checkpoint', 'source_file', 'source_row', 'policy_name', 'policy', 'variant', 'mode', 'step', 'episode_idx'] if c in cols]
        for cat_name, cat_cols in cats.items():
            table_cols = []
            for c in identity + cat_cols:
                if c not in table_cols:
                    table_cols.append(c)
            out += [f'#### {cat_name}', '', rows_table(table_cols, rows), '']
    return '\n'.join(out).rstrip() + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for cfg in TABLES:
        out_path = OUT_DIR / f"{cfg['slug']}_full_raw_data.md"
        content = render_table(cfg)
        out_path.write_text(content, encoding='utf-8')
        manifest_cols, specs = read_csv(ROOT / cfg['manifest'])
        raw_count = 0
        for spec in specs:
            _, rows, _ = prefixed_raw_rows(spec, cfg['source_key'])
            raw_count += len(rows)
        index_rows.append([cfg['name'], f'`{out_path.name}`', cfg['manifest'], str(len(specs)), str(raw_count)])
        print(f'wrote {out_path.relative_to(ROOT)}')
    readme = ['# Table repro full raw data index', '', 'These files are generated from `scripts/table_repro/TABLE IV.sh` through `TABLE VIII.sh` via their actual LaTeX table manifests. Each table MD includes the manifest, computed aggregation, and every row/field from every source CSV referenced by that table.', '', md_table(['table', 'markdown', 'manifest', 'manifest rows', 'raw rows'], index_rows), '']
    (OUT_DIR / 'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print(f'wrote {(OUT_DIR / "README.md").relative_to(ROOT)}')

if __name__ == '__main__':
    main()
