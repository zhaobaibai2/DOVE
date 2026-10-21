# MetaDrive implementation

This tree contains the JAX implementation used for MetaDrive training and frozen-policy evaluation.

## Main modules

- `relax/algorithm/pvp_dacer.py`: PVP-DACER training state, EnergyRank same-state actor ranking, and CPCal critic ordering.
- `relax/network/`: diffusion actor and twin distributional proxy-risk critics.
- `relax/buffer/` and `relax/trainer/`: intervention-pair replay and online training.
- `relax/utils/diffusion.py`: reverse diffusion, critic guidance, and bounded action refinement.
- `relax_env/`: MetaDrive HIL wrappers and takeover handling.
- `scripts/train_pvp_dacer_metadrive_off.py`: demonstration loading/collection, BC warm-up, and online training.
- `scripts/eval_pvp_policies_fixed_v3.py`: frozen-policy and inference-gate evaluation.
- `experiments/run_pdf_*.py`: paper evaluation and ablation command generators.

The upstream DACER-only algorithm file and baseline repositories were left out. The DOVE modules retain the diffusion actor/critic backbone required by the method; see the repository-level third-party notice.

## Environment

Use Python 3.10 or 3.11. Install the JAX build appropriate for the CUDA driver first; GPU wheels vary by platform. Then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

MetaDrive interactive collection needs a display and a supported controller. For headless evaluation, use the evaluation script's `--help` options and disable rendering. These scripts do not download or include demonstrations or checkpoints.

## Training and evaluation

Supply an existing Stage 1a demonstration directory with `--demo_root`, or use the script's collection options to collect demonstrations in MetaDrive:

```bash
cd simulation
python -u scripts/train_pvp_dacer_metadrive_off.py --help
python -u scripts/train_pvp_dacer_metadrive_off.py \
  --demo_root /path/to/demonstrations \
  --start_seed 100 --num_scenarios 20 --traffic_density 0.06 \
  --stage1b_updates 10000 --stage1b_lambda_bc 50 \
  --total_step 50000 --policy_mode hybrid_dacer --critic_objective cost \
  --lambda_pv 1 --B 2 --lambda_bc 12 --lambda_rl 0.1 --lambda_reg 2 \
  --use_energy_rank --lambda_er 1 --er_margin 0.05 \
  --er_min_action_gap 0.03 --er_positive_action behavior \
  --er_use_primal_dual --er_budget 0.01 --er_dual_lr 0.001 --er_eta_init 1 \
  --lambda_pv_constraint 1 --pv_constraint_margin 0.1 \
  --log_dir outputs/dove_main
```

Training uses maps 100–119. Frozen-policy evaluation defaults to the disjoint held-out maps 200–209 and rejects training-map seeds:

```bash
python -u scripts/eval_pvp_policies_fixed_v3.py \
  --log_dir outputs/dove_main \
  --maps 200 201 202 203 204 205 206 207 208 209 \
  --num_episodes 5
```

The run writes checkpoints and logs under the selected output directory. `experiments/run_pdf_closed_loop_main.py` and neighboring `run_pdf_*.py` files generate paper-protocol commands using the same held-out map set; set checkpoint and output paths for your run before executing them.

## Dependencies

`requirements.txt` lists the simulation dependencies. Install JAX/JAXlib with a CUDA-compatible build as described in the JAX installation instructions for your system.
