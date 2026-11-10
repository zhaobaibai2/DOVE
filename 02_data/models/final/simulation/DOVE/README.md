# DOVE models used by TAB-01

Only the three checkpoints listed in `manifest.json` participate in the current evaluation:

- `seed0-final.pkl`
- `seed1-final.pkl`
- `seed2-final.pkl`

They were copied byte-for-byte from the original run12 training directory. `args.json` is the original training configuration. Model/config SHA-256 values and the full evaluation flags are recorded in `manifest.json`.

Run `bash scripts/run_pvp.sh test` from the project root. It evaluates all three models and automatically runs `calculate.py`, displaying episode metrics and the final summary.

See [the full data provenance and formulas](../../../../raw_data/TABLES/TAB-01/raw_data/DOVE/README.md) for source paths, archived measurements, the meaning of each uncertainty column, and the distinction between the old single-checkpoint paper row and the three-checkpoint calculation. Existing older evaluation CSV/TensorBoard files in this directory are not used by this pipeline.

## Filename mapping

| Current filename | Original checkpoint | Actual training seed |
| --- | --- | ---: |
| seed0-final.pkl | policy-11000-21998.pkl | 0 |
| seed1-final.pkl | policy-12000-23998.pkl | 0 |
| seed2-final.pkl | policy-13000-25998.pkl | 0 |

The seed/final words are requested filename labels. These are three intermediate checkpoints from the same seed-0 training run, not three independent seeded training runs or the last saved training checkpoints. `manifest.json` preserves the original names, sample/update steps, actual training seed, source run and unchanged payload hashes. Existing measured CSVs retain their original checkpoint names; readers support both names.
