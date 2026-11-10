# TAB-05 calculation

For each reported method, sum the four scenario success counts and verify the reported total out of 40. This reproduces aggregate arithmetic only. A scientifically valid reconstruction additionally requires 160 observed trial rows (method, scenario, trial identity/timestamp, success/failure outcome, failure reason, and raw-log link).

The archived real-car code documentation confirms that strict success/collision was not present in the ROS log and had to be written into a separate manual CSV. The preserved eval archive contains one repeated trial ID over control-step rows, not a multi-trial success ledger. Scene-name Markdown files explicitly say their labels were inferred from control curves and are not human truth. Therefore no mean, uncertainty, or per-scenario count is claimed as raw-data-derived, and no aggregate count is expanded into fake trials.

For scenario analysis only, `hypothetical/hypothetical_trial_ledger.csv` provides 160 explicitly marked rows. For each reported cell with S successes out of N trials, rows 1..S are assigned `AssumedSuccess=1` and rows S+1..N are assigned `AssumedSuccess=0`; `hypothetical_aggregate_check.csv` confirms all 16 reported aggregates. These are assumption values, not observations, contain no raw-log links, and are excluded from authoritative builds.

The separate `history_evidence/event_tag_inventory.csv` and `event_tag_long.csv` inventory 46 retained TensorBoard event paths (36 unique contents). None contains a strict per-trial success/failure/collision/outcome tag; route completion, intervention, takeover-rate, and count fields remain runtime proxies only.
