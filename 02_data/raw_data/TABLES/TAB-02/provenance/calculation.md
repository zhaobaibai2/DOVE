# TAB-02 raw-data calculation

The calculation starts from the archived `*_pairs.csv` and `intervention_pair_rows.csv` files, not the already aggregated paper table.

For each actor checkpoint `c` and its 256 retained pairs: `PairSat_E(c)=mean(1[E_neg-E_pos >= m_E])`; `OVR_E(c)=mean(ragr)` where `ragr` is the rejected-action generation rate; `Div(c)=mean(sample_diversity)`. The reported center is the unweighted mean over checkpoint means. Checkpoint SE is sample-SD divided by `sqrt(number of checkpoints)`.

For each critic checkpoint and its 512 pair rows: `PairAcc_C(c)=mean(1[qbar_neg-qbar_pos > 0])`, stored as the binary `pairacc_c` column. The paper value is the mean of checkpoint means.

The exact raw memberships are actor 50/52/46 checkpoints at 256 pairs per checkpoint and critic 52/52/48 checkpoints at 512 pairs per checkpoint for Positive-only / ER w/o CPCal / Full DOVE.
