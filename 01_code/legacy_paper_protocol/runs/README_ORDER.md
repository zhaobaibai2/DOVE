# DOVE paper-protocol experiment scripts

The authoritative release protocol is defined in
`experiments/dove_paper_protocol.py`: fixed training maps 104--123,
held-out maps 200--209, and independent training seeds 0--4.  The map and
seed integer IDs are explicit release choices because the manuscript states
the 20/10/5 counts but does not print the IDs.  Every paper run uses five
initial HITL episodes per training map and the inclusive interaction budget
`N_init + N_rep = 50,000`.

Generate the complete 25-row DOVE/ablation/PVP/H-DSAC task matrix without
starting any experiment:

`python experiments/run_training_matrix.py`

Rows for **DOVE**, **Positive-only**, and **ER_without_CPCal** have
commands.  **PVP** and **H-DSAC** remain marked
`adapter_required` until their external sources have a verified adapter for
the identical map split, supervisor interface, and inclusive budget.

Some filenames below retain `seed104`/`doveer` for compatibility with old
logs; their displayed names and new output directories use the paper terms.

All scripts activate `${CONDA_ENV:-dacer_pvp}` through `${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}` and check `nvidia-smi`.

Stage1a uses steering wheel + render. Stage2 training scripts default to steering wheel + render through:

`--controller steering_wheel --manual_control --use_render`

Override this for headless/offline runs with `CONTROL_FLAGS=""`. The DBP-only offline script defaults to no manual/render flags; set `CONTROL_FLAGS="--controller steering_wheel --manual_control --use_render"` if you explicitly need interactive rendering there.

Evaluation scripts are closed-loop policy tests and intentionally use no render:

`--no_use_render`

Their defaults are the 10 held-out release maps (200--209) with two episodes
per map, i.e. 20 held-out episodes for each independently trained seed.
`runs/43_run_intervention_pair_diagnostics_seed104.sh` is the deliberate
exception: it is a training-distribution audit on one diagnostic map/seed, as
described in RQ2, and is not a held-out performance evaluation.

## 0. Environment

1. `bash runs/00_check_dacer_pvp_env.sh`

Confirms conda, Python, `nvidia-smi`, and JAX GPU backend.

## 1. Data Collection

2. `bash runs/01_stage1a_collect_seed104_5pass_only.sh`

Collects five human-controlled episodes on each of the 20 training maps for
`${TRAINING_SEED:-0}`. It prints the `DEMO_ROOT` to reuse.

## 2. Training Ablations

3. `DEMO_ROOT=/path/to/stage1a/data bash runs/10_train_dbp_only_seed104.sh`

Positive-only diffusion initialization checkpoint.  For the matched-budget
paper reference, use the generated `positive_only_diffusion_reference` task.

4. `DEMO_ROOT=/path/to/stage1a/data bash runs/11_train_pvp_dbp_critic_seed104_50k.sh`

Legacy diffusion-plus-PVP-critic diagnostic. This is not the paper row named PVP.

5. `DEMO_ROOT=/path/to/stage1a/data bash runs/12_train_doveer_er_seed104_50k.sh`

Main DOVE training. It enables EnergyRank and CPCal; deployment guidance and
the acceptance/fallback gate remain test-time operations.

6. `DEMO_ROOT=/path/to/stage1a/data bash runs/13_train_doveer_er_no_pv_constraint_seed104_50k.sh`

Paper training ablation: ER_without_CPCal (typeset as ``ER w/o CPCal'' in
the final TeX table).

## 3. Test-Time Ablations For Any Trained Policy

Use a policy produced by training, for example:

`POLICY="$(find logs/12_train_DOVE_* -name 'final_policy_*.pkl' | sort | tail -n 1)"`

7. `POLICY=/path/to/policy.pkl bash runs/20_eval_vanilla_policy_seed104.sh`

Vanilla diffusion action, no UPV-GDS, no Final Gate.

8. `POLICY=/path/to/policy.pkl bash runs/21_eval_upvgds_no_gate_policy_seed104.sh`

DOVE guided proposal executed directly (deployment-gate ablation).

9. `POLICY=/path/to/policy.pkl bash runs/22_eval_upvgds_final_gate_full_policy_seed104.sh`

Full deployment: UPV-GDS + Final Gate.

10. `POLICY=/path/to/policy.pkl bash runs/23_eval_final_gate_no_disagreement_policy_seed104.sh`

Final Gate ablation with disagreement gate disabled.

11. `POLICY=/path/to/policy.pkl bash runs/24_eval_final_gate_proxy_only_policy_seed104.sh`

Final Gate proxy-only ablation.

12. `POLICY=/path/to/policy.pkl bash runs/25_eval_final_gate_prior_only_policy_seed104.sh`

Final Gate prior-only ablation.

13. `POLICY=/path/to/policy.pkl bash runs/26_eval_topk_rerank_policy_seed104.sh`

Critic top-K reranking baseline.

Or run the main test-time ablations together, including the top-K reranking baseline:

`POLICY=/path/to/policy.pkl bash runs/30_eval_all_main_for_policy_seed104.sh`

## Suggested First 50K Round

1. `bash runs/00_check_dacer_pvp_env.sh`
2. `bash runs/01_stage1a_collect_seed104_5pass_only.sh`
3. `DEMO_ROOT=/path/to/stage1a/data bash runs/11_train_pvp_dbp_critic_seed104_50k.sh`
4. `POLICY="$(find logs/11_train_pvp_dbp_critic_seed104_50000_* -name 'final_policy_50000.pkl' | sort | tail -n 1)" bash runs/30_eval_all_main_for_policy_seed104.sh`
5. `DEMO_ROOT=/path/to/stage1a/data bash runs/12_train_doveer_er_seed104_50k.sh`
6. `POLICY="$(find logs/12_train_DOVE_* -name 'final_policy_*.pkl' | sort | tail -n 1)" bash runs/30_eval_all_main_for_policy_seed104.sh`

## 4. All-Checkpoint Evaluation Matrix

The `20`--`26` scripts are **single-policy wrappers**: they require `POLICY=/path/to/policy.pkl` and therefore each CSV contains exactly one policy row. This is expected and is useful after a validation-best checkpoint has already been selected.

To evaluate **every checkpoint in one training directory** under the same inference modes, do not pass `--policy`. Use:

```bash
RUN_DIR=/path/to/pvp_dacer_xxx_s0 MAPS="104" NUM_EPISODES=20 RUN_TOPK=0 \
  bash runs/40_eval_all_policies_all_modes_seed104.sh
```

This writes one CSV per inference mode plus:

- `all_modes_all_policies_summary.csv`
- `best_policy_by_mode.csv`
- `best_policy_by_mode.md`

`RUN_TOPK=0` is the default because top-k reranking is much slower than vanilla/guided eval. Set `RUN_TOPK=1` only when you intentionally want to run top-k for every checkpoint:

```bash
RUN_DIR=/path/to/pvp_dacer_xxx_s0 MAPS="104" NUM_EPISODES=20 RUN_TOPK=1 TOPK=8 \
  bash runs/40_eval_all_policies_all_modes_seed104.sh
```

To sweep the latest run 11/12/13 checkpoint directories vertically in one command:

```bash
MAPS="104" NUM_EPISODES=20 RUN_TOPK=0 RUN_IDS="11 12 13" \
  bash runs/41_eval_training_runs_all_policies_seed104.sh
```

Use the all-checkpoint matrix for validation/checkpoint selection and debugging final-checkpoint collapse. For paper test results, select checkpoints on held-out validation maps/seeds first, then evaluate the selected checkpoints once on held-out test maps/seeds.



## Exhaustive script-driven eval matrix

Use `runs/47_run_every_eval_script_and_diagnostics_gpu_jax_seed104.sh` when the
goal is to run every single-policy eval/ablation bash in `runs/` for every
checkpoint of every selected training run.  This is different from the older
fixed-mode runner: it discovers `runs/[0-9][0-9]_eval_*_policy_seed104.sh` and
loops over all policies.

Fast sweep example:

```bash
NUM_EPISODES=10 RUN_IDS="10 11 12 13" RUN_TOPK=0 DIAG_ALL_POLICIES=1 \
  bash runs/47_run_every_eval_script_and_diagnostics_gpu_jax_seed104.sh
```

Outputs are written under:

```text
logs/99_every_eval_script_and_pair_diag_seed*/closed_loop_matrix/run*/<mode>/<policy>/
```

Use `closed_loop_matrix/all_runs_table_ready_episode_mean_std.csv` for
closed-loop paper-table mean/std values, and `pair_diag_run*/intervention_pair_summary.csv`
for PairSat / V_E / IAR diagnostics.
