# DOVE

Code, selected models, experimental data, and figure entry points for **DOVE: Learning Diffusion Policies from Accepted and Rejected Actions in Human-in-the-Loop Driving**.

[Project website](https://zhaobaibai2.github.io/DOVE/)

The existing project website, illustrations, and videos are retained. The research bundle follows the author's final local organization:

| Directory | Contents |
| --- | --- |
| `01_code/` | Simulation and vehicle implementations, legacy experiment runners, and environment information |
| `02_data/` | Final plotting inputs, manuscript table transcriptions, selected-model configurations, and release-data downloader |
| `03_plotting/plot/` | One Python entry point per paper figure asset |
| `03_plotting/output/` | Generated figure PDFs and PNGs |

## Figures

```bash
python -m pip install -r 01_code/environment/requirements_plotting.txt
python 03_plotting/plot/plot_sim_reward.py
python 03_plotting/plot/plot_sim_takeover.py
python 03_plotting/plot/plot_real_takeover.py
```

Each other `plot_*.py` file runs independently. Three scripts redraw final curve values; the other eight export archived paper artwork. The vehicle takeover CSV contains final figure vertices, rather than raw telemetry. Outputs are written to `03_plotting/output/`.

## Complete data and selected models

Large logs, archived experiment inputs, selected comparison models, and two example vehicle buffers are distributed as segmented assets in the repository's data release. This avoids GitHub's Git file-size limit while retaining the local `02_data` layout.

```bash
python 02_data/download_data.py
python 02_data/download_data.py --verify-only
```

The downloader verifies SHA-256 checksums and restores the files listed in `02_data/DATA_MANIFEST.csv`. Release details and per-part checksums are in `02_data/DATA_RELEASE.json`. Allow disk space for the extracted data and cached archive parts.

The author's **main DOVE simulation checkpoint is not distributed**. Its configuration is retained. Ablation checkpoints, comparison-method models, and selected vehicle models are included in the data release. A single selected model is retained per method/variant; original demonstration files and the two vehicle replay examples are data, not intermediate model checkpoints.

## Implementations

See [simulation instructions](01_code/existing_public_release/simulation/README.md) and [source layout](01_code/existing_public_release/README.md). The primary simulation implementation uses JAX/MetaDrive; the vehicle implementation uses ROS 2/PyTorch. Install the appropriate ML framework and vehicle middleware separately. Legacy runners require reviewing their paths and protocol defaults before use.

## Reproduction scope

The bundle includes the final paper's figures and reported table values, together with recovered experimental inputs. This is a partial reproducibility release: figure execution and selected table calculations have been checked; complete retraining and all final-paper aggregate statistics have not been verified. Table transcriptions and website numbers are manuscript-reported values, rather than newly measured results. Some baseline implementations and experiment-to-paper mappings still require completion.
