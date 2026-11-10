# TAB-03 calculation and provenance

All displayed values are pooled episode means. For each metric `x`, the stored diagnostics also report sample SD `sqrt(sum((x_i-mean)^2)/(n-1))` and SE `SD/sqrt(n)`. The final paper displays means only. Collision is the source column `crash`.

Panel A uses the repository training-ablation manifest; Panel B uses the test-time-ablation manifest. Every named CSV was copied into `raw_sources/` and hash-indexed before aggregation.

## Caption inconsistency

- Positive-only (DBC + critic): 5 checkpoints / 500 episodes.
- ER w/o CPCal: 6 checkpoints / 600 episodes.
- Full DOVE: 4 checkpoints / 400 episodes.

The final TeX says 5 checkpoints and 500 episodes for every Panel A configuration, but the exact source membership that reproduces the displayed values is 5/6/4 checkpoints and 500/600/400 episodes. This discrepancy is preserved, not normalized away.

The Positive-only pooled reward is `324.984769655...`; conventional two-decimal rounding gives `324.98`, while the final TeX prints `324.99`. The absolute difference is about 0.00523, so the source is saved as a near match and the printed value is not silently forced.
