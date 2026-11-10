# TAB-04 calculation from archived row-level inputs

## Authoritative inputs

- Training: `raw_logs/collection_repair_train_rows_30475.csv` (30,475 rows).
- Frozen execution: `raw_logs/frozen_execution_rows_2629.csv` (2,629 rows).
- Connected trace: `raw_connected_trace/realcar_actual_control_trace_20260624_164122_actual_main_block_0_919s_source_actual_tags_combined.csv` (8,411 samples).
- SHA-256 values and original paths are recorded in `source_manifest.csv`.

`processed_analysis/` is not an input to this rebuild.

## Selection and formulas

- Collection: `phase_name == "Data Collection"`.
- Repair adaptation: `phase_name == "PVP Learning"`.
- Frozen execution: all rows in the archived evaluation CSV.
- Gate denominators include only rows satisfying `gate_diagnostics_logged == 1` because non-diagnostic rows contain zero placeholders.
- Gate accept rate = `sum(gate_accepted) / diagnostic-row count`; fallback is analogous.
- Duration = `max(timestamp) - min(timestamp)`. The trace uses `relative_time_s`; other rows use `ts`.
- P95 uses linear interpolation at position `(n - 1) * 0.95`, equivalent to NumPy's default percentile method.
- Mean trace speed is the arithmetic mean of all 8,411 `actual_speed_mps` samples; maximum is the literal maximum with no smoothing, clipping, or trimming.

## Recalculated paper rows

| Row | Raw rows/samples | Gate evals | Accept/fallback | P95 timing | Duration/speed |
|---|---:|---:|---:|---:|---:|
| Collection | 7514 | 375 | 0.005333333/0.994666667 | process 157.575654984 ms; loop 161.474120617 ms | n/a |
| Repair adaptation | 22362 | 1104 | 0.746376812/0.253623188 | process 231.696915627 ms; loop 238.205575943 ms | n/a |
| Frozen execution | 2629 | 131 | 0.984732824/0.015267176 | process 289.547443390 ms; infer 286.312532425 ms; loop 305.992412567 ms | 447.151437759 s |
| Connected trace | 8411 | n/a | n/a | n/a | 919.651245 s; mean 2.083194635; max 3.833333; P95 3.111111 m/s |

## Evidence gap

The paper labels `2.08/3.11 m/s` as mean/maximum speed. The row-level trace gives mean `2.083194635` m/s and literal maximum `3.833333` m/s. The paper's `3.11` instead matches the raw trace P95 `3.111111` m/s after rounding. No source-backed maximum-speed filtering rule was found, so the maximum is not rewritten or silently clipped.
