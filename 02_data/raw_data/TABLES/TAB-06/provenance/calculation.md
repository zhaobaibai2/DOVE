# TAB-06 calculation

The preserved original script computes one row per source run. For each run: mean speed is the arithmetic mean of the logged speed; lateral MAE is mean(abs(lateral error)); jerk is the mean absolute second finite difference of speed divided by timestamp intervals. Method mean and sample SD use pandas groupby with ddof=1.

The exact source sample sizes are 4 DACER runs, 2 DOVE route-control runs, 3 IWR runs, and 3 PVP-TD3 runs. These are not 40 trial rows per method. The mapping from DACER source runs to the paper's Positive-only row is not established by a training/config manifest and remains explicit.
