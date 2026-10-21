# 0508 MetaDrive Comparison Workspace

This workspace is intentionally trimmed. It keeps only the code needed to compare:

- local UPV-GDS / PVP-DACER (`relax`, `relax_env`, `scripts`)
- original DACER core subset (`baselines/dacer_source`)
- PVP MetaDrive subset (`baselines/pvp`)
- H-DSAC MetaDrive subset (`baselines/hdsac`)
- HACO MetaDrive subset (`baselines/haco`)

Smoke test:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=runs/mplconfig \
python experiments/run_metadrive_smoke.py --methods all --seed 104 --steps 1
```

The command above is only a one-map import/environment smoke test. Final-paper
tasks use `experiments/dove_paper_protocol.py` as the single source of truth:
20 training maps (104--123), 10 held-out maps (200--209), five independent
training seeds (0--4), and two held-out episodes per map for each training seed.
Generate the paper task matrix without running it via:

```bash
python experiments/run_training_matrix.py
```

The HACO environment runs on MetaDrive 0.4.3 after compatibility patches. The HACO trainer still requires `ray/rllib`, which is not installed in the current `dacer_pvp` conda environment.

Experiment entrypoints:

- `run_source_integrity_check.py`: imports, dependency status, and one-step smoke.
- `run_metadrive_smoke.py`: minimal one-step method smoke.
- `run_scene_simulation_matrix.py`: named scene simulation matrix.
- `run_training_matrix.py`: generates baseline training command CSV/shell.
- `run_checkpoint_eval_matrix.py`: generates checkpoint evaluation command CSV/shell.
- `run_upvgds_ablation_matrix.py`: generates UPV-GDS inference ablation commands.
- `run_policy_ablation_training_matrix.py`: generates UPV-GDS/PVP-DACER training ablation commands.
- `run_comparison_matrix.py`: top-level command printer covering integrity, smoke, scene simulation, training, checkpoint eval, and ablation matrices.
- `run_pdf_full_suite.py`: generates the full absolute-path command suite for the PDF experiment plan.
- `run_pdf_closed_loop_main.py`: PDF Sec.8 main closed-loop driving evaluation.
- `run_pdf_scene_protocol.py`: PDF seen/unseen map and traffic-density scene protocol.
- `run_pdf_guidance_source_ablation.py`: PDF Sec.9.1 guidance source ablation.
- `run_pdf_uncertainty_ablation.py`: PDF Sec.9.2 uncertainty gate ablation.
- `run_pdf_guidance_target_ablation.py`: PDF Sec.9.3 guidance target/schedule ablation.
- `run_pdf_critic_choice_ablation.py`: PDF Sec.9.4 critic choice ablation.
- `run_pdf_data_semantics_ablation.py`: PDF Sec.9.5 data semantics training ablation.
- `run_pdf_mechanism_action_shift.py`: PDF Sec.10.1 action shift mechanism check.
- `run_pdf_mechanism_proxy_risk_drop.py`: PDF Sec.10.2 proxy-risk reduction mechanism check.
- `run_pdf_mechanism_uncertainty_gate.py`: PDF Sec.10.3 uncertainty gate mechanism check.
- `run_pdf_failure_case_visualization.py`: PDF Sec.10.4 video/case command generator.
- `aggregate_pdf_metrics.py`: PDF Sec.11 paper-table metric aggregation.

See `../EXPERIMENT_STATUS_0509.md` for detailed completed/pending/next-step notes.
See `../PDF_EXPERIMENTS_0509.md` for PDF-specific experiment coverage and absolute commands.
