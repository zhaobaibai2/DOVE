# DOVE

**DOVE (Diffusion with Observed Vetoes)** learns from human takeovers during driving. Each intervention provides a same-state pair: the policy's rejected action and the human's accepted correction. The project uses that pair to shape a diffusion policy, train proxy-risk critics, and guide action selection.

- **Project page:** https://zhaobaibai2.github.io/DOVE/
- **Repository:** https://github.com/zhaobaibai2/DOVE

## Videos: where to find them

All video clips used by the project page are under [`assets/videos/`](assets/videos/). These are illustrative clips for viewing; they are not the demonstration buffers required for training.

### Driving and training demonstrations — `assets/videos/demos/`

| Video | What it shows |
| --- | --- |
| [`route_overview.mp4`](assets/videos/demos/route_overview.mp4) | Overview of the physical driving route and platform. |
| [`expert_demonstration.mp4`](assets/videos/demos/expert_demonstration.mp4) | Human/expert demonstration collection used to explain offline initialization. |
| [`offline_success.mp4`](assets/videos/demos/offline_success.mp4) | The initial policy succeeding under its original training setup. |
| [`distribution_shift_failure.mp4`](assets/videos/demos/distribution_shift_failure.mp4) | The same initial policy failing after the obstacle is moved. |
| [`online_hitl_adaptation.mp4`](assets/videos/demos/online_hitl_adaptation.mp4) | A supervisor taking over while the policy is repaired online. |

### Supplementary policy comparisons — `assets/videos/comparisons/`

These clips show an earlier H-DSAC setup and baseline failure examples. They are additional platform demonstrations, not the four-scenario DOVE evaluation reported on the page.

| Scenario | Policy clip | Baseline failure clip |
| --- | --- | --- |
| Obstacle avoidance | [`obstacle_policy.mp4`](assets/videos/comparisons/obstacle_policy.mp4) | [`obstacle_baseline_failure.mp4`](assets/videos/comparisons/obstacle_baseline_failure.mp4) |
| Pedestrian crossing | [`pedestrian_policy.mp4`](assets/videos/comparisons/pedestrian_policy.mp4) | [`pedestrian_baseline_failure.mp4`](assets/videos/comparisons/pedestrian_baseline_failure.mp4) |
| Turning | [`turning_policy.mp4`](assets/videos/comparisons/turning_policy.mp4) | [`turning_baseline_failure.mp4`](assets/videos/comparisons/turning_baseline_failure.mp4) |

## What is in the repository?

| Path | Purpose |
| --- | --- |
| [`simulation/`](simulation/README.md) | JAX implementation for MetaDrive training, evaluation, and paper experiments. |
| [`robot/`](robot/README.md) | PyTorch and ROS 2 implementation for the physical unmanned ground vehicle (UGV). |
| `assets/images/`, `assets/videos/` | Figures, posters, and videos used by the project page. |
| `index.html`, `assets/css/`, `assets/js/` | Static project website. |
| `data/main_results.csv`, `data/main_results.json` | Aggregated result tables in CSV and JSON formats. |
| [`docs/experiment_notes.md`](docs/experiment_notes.md) | Additional notes on the experiment protocol. |

The simulation and physical-vehicle implementations are separate: simulation uses JAX and MetaDrive; the UGV package uses PyTorch and ROS 2. Demonstration buffers, trained checkpoints, vehicle sensor/perception software, and the low-level CAN control stack are not included.

## Code map

### MetaDrive simulation (`simulation/`)

- `scripts/train_pvp_dacer_metadrive_off.py` loads or collects demonstrations, performs behavior-cloning warm-up, then runs online human-in-the-loop training.
- `scripts/eval_pvp_policies_fixed_v3.py` evaluates a saved policy without further training. Its default held-out maps are 200–209; the documented training maps are 100–119.
- `relax/algorithm/pvp_dacer.py` contains the training algorithm, including EnergyRank actor preference learning and CPCal critic ordering from same-state intervention pairs.
- `relax/network/` defines the diffusion actor and twin proxy-risk critics. `relax/buffer/` and `relax/trainer/` store intervention data and run updates.
- `relax/utils/diffusion.py` implements reverse diffusion and critic-guided action refinement; `relax_env/` connects the algorithm to MetaDrive and human takeovers.
- `experiments/run_pdf_*.py` and `experiments/aggregate_pdf_metrics.py` prepare paper-protocol runs and aggregate their metrics.

### Physical UGV (`robot/`)

- `src/car_dacer_torch/car_dacer_torch/car_dacer_torch.py` is the ROS 2 policy/training node.
- `torch_networks.py`, `torch_diffusion.py`, and `torch_algorithm.py` implement the PyTorch diffusion policy, critics, diffusion sampling, and learning updates.
- `torch_replay_buffer.py` stores human actions and intervention pairs; `safety_manager.py` applies the configured action and safety checks.
- `offline_train_doveer.py` trains from recorded Route 1 buffers. It supports the full DOVE mode and a `positive_only` comparison mode.
- `car_dacer_torch/configs/` contains route and experiment YAML settings. `src/car_interfaces/` defines the ROS 2 messages this package imports.
- `tools/summarize_real_metrics.py` summarizes evaluation CSV logs.

The main ROS 2 executables are registered in `src/car_dacer_torch/setup.py`:

| Executable | Use |
| --- | --- |
| `car_dacer_torch` | Run the main human-in-the-loop collection/training node. |
| `offline_train_doveer` | Train from saved Route 1 buffer data. |
| `car_dacer_torch_eval_det` | Run deterministic policy evaluation. |
| `car_dacer_torch_eval_noisy` | Run evaluation with action noise. |
| `car_dacer_torch_skip` | Load an available model and start directly in the online-learning phase. |

The ROS 2 package supplies the DOVE node and custom messages only. It expects the target vehicle's ROS 2 distribution, PyTorch/CUDA build, sensor and localization topics, and CAN/control stack to be installed and configured separately.

## Run the MetaDrive code

Use Python 3.10 or 3.11. Install a JAX/JAXlib build that matches your platform and CUDA driver, then install the listed dependencies:

```bash
cd simulation
python -m pip install -r requirements.txt
python -u scripts/train_pvp_dacer_metadrive_off.py --help
python -u scripts/eval_pvp_policies_fixed_v3.py --help
```

Training needs MetaDrive and either an existing demonstration directory or interactive demonstration collection. For example, with a prepared demonstration directory:

```bash
python -u scripts/train_pvp_dacer_metadrive_off.py \
  --demo_root /path/to/demonstrations \
  --start_seed 100 --num_scenarios 20 --traffic_density 0.06 \
  --total_step 50000 --policy_mode hybrid_dacer --critic_objective cost \
  --use_energy_rank --lambda_er 1 --er_margin 0.05 \
  --log_dir outputs/dove_main
```

Then evaluate the saved run on held-out maps:

```bash
python -u scripts/eval_pvp_policies_fixed_v3.py \
  --log_dir outputs/dove_main \
  --maps 200 201 202 203 204 205 206 207 208 209 \
  --num_episodes 5 --no_use_render
```

Training logs and checkpoints are written under `--log_dir`. For the complete paper configuration, data-collection options, and all command-line settings, see [`simulation/README.md`](simulation/README.md). The repository does not provide the demonstration directory or pretrained checkpoints.

## Run the ROS 2 vehicle code

Build from the `robot/` directory in an environment with the target vehicle's ROS 2 and PyTorch installations. The Python requirements file does not install ROS 2 or PyTorch; install versions compatible with the vehicle separately.

```bash
cd robot
rosdep install --from-paths src --ignore-src -r -y
python -m pip install -r requirements.txt
colcon build --packages-up-to car_dacer_torch
source install/setup.bash
```

Select a configuration with `DACER_CONFIG_PATH`, then launch the installed ROS 2 node. For example, to collect Route 1 human-intervention data:

```bash
export DACER_CONFIG_PATH="$PWD/src/car_dacer_torch/car_dacer_torch/configs/01_route1_hil_collect.yaml"
ros2 run car_dacer_torch car_dacer_torch
```

To train offline from a buffer collected on Route 1, use the adaptation configuration and point `--buffer` to the recorded file or directory:

```bash
export DACER_CONFIG_PATH="$PWD/src/car_dacer_torch/car_dacer_torch/configs/02_route1_adapt_doveer_full.yaml"
ros2 run car_dacer_torch offline_train_doveer \
  --config "$DACER_CONFIG_PATH" \
  --buffer /path/to/route1-buffer.pkl \
  --mode full
```

To run frozen deterministic evaluation on Route 2, select the Route 2 config and evaluation executable:

```bash
export DACER_CONFIG_PATH="$PWD/src/car_dacer_torch/car_dacer_torch/configs/04_route2_eval_doveer_full.yaml"
ros2 run car_dacer_torch car_dacer_torch_eval_det
```

The YAML files correspond to these steps:

| Config | Purpose |
| --- | --- |
| `00_shadow_safety.yaml` | Stage 0 shadow-mode runtime check; does not send CAN commands. |
| `01_route1_hil_collect.yaml` | Collect Route 1 human-intervention pairs. |
| `02_route1_adapt_doveer_full.yaml` | Offline adaptation from the collected Route 1 buffer. |
| `06_route1_train_doveer_full_online.yaml` | Continue with online DOVE training on Route 1. |
| `07_route1_eval_doveer_full.yaml` | Frozen evaluation on the seen Route 1. |
| `04_route2_eval_doveer_full.yaml` | Frozen evaluation on Route 2. |
| `09_route1_train_doveer_full_from_scratch.yaml` | Run the Route 1 collect, warm-up, and online-training flow from scratch. |

To see offline-training options, run `ros2 run car_dacer_torch offline_train_doveer --help`. Review the topic names, route, CAN interface, and vehicle settings before connecting to a platform; no vehicle logs or model checkpoints are bundled. See [`robot/README.md`](robot/README.md) for build details.

## Preview the project page locally

```bash
python3 -m http.server 8000
```

Open http://localhost:8000 in a browser.
