"""Single source of truth for the final DOVE manuscript protocol.

The manuscript fixes the *counts* (20/10 maps and five independent runs)
but does not print the integer IDs.  This release makes the split and run
seeds explicit so every generated command uses the same values.  The IDs
extend the map-104 and held-out-map-200 conventions already used by this
workspace; they must not be presented as recovered historical provenance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple


PAPER_METHOD_NAMES: Dict[str, str] = {
    "dove": "DOVE",
    "positive_only_diffusion_reference": "Positive-only",
    "er_without_cpcal": "ER_without_CPCal",
    "hdsac": "H-DSAC",
    "pvp": "PVP",
}

# Historical logs and the provenance model inventory use several descriptive
# spellings.  Keep those paths/manifests intact and normalize only at the
# paper-facing display boundary.
PAPER_METHOD_ALIASES: Dict[str, str] = {
    "DOVE": "DOVE",
    "Full DOVE": "DOVE",
    "Positive-only": "Positive-only",
    "Positive-only Diffusion": "Positive-only",
    "Positive-only diffusion reference": "Positive-only",
    "ER_without_CPCal": "ER_without_CPCal",
    "ER without CPCal": "ER_without_CPCal",
    "ER w/o CPCal": "ER_without_CPCal",
    "H-DSAC": "H-DSAC",
    "PVP": "PVP",
}


def canonical_paper_method_name(name: str) -> str:
    """Return the exact paper/Harness display name without renaming sources."""

    try:
        return PAPER_METHOD_ALIASES[name.strip()]
    except KeyError as exc:
        raise ValueError(f"unknown paper method alias: {name!r}") from exc


@dataclass(frozen=True)
class DOVEPaperProtocol:
    training_maps: Tuple[int, ...] = tuple(range(104, 124))
    heldout_maps: Tuple[int, ...] = tuple(range(200, 210))
    training_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    initial_collection_episodes_per_map: int = 5
    total_interaction_budget: int = 50_000
    heldout_episodes_per_training_seed: int = 20
    diffusion_steps: int = 20
    critic_hidden_layers: int = 2
    critic_hidden_width: int = 256
    diffusion_hidden_width: int = 128
    batch_size: int = 256
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 1.5e-4
    discount: float = 0.99
    polyak_tau: float = 5e-3
    actor_update_interval: int = 3
    behavior_prior_updates: int = 10_000
    critic_warmup_updates: int = 1_000
    updates_per_environment_step: int = 2
    lambda_bc: float = 12.0
    lambda_rl: float = 0.1
    lambda_reg: float = 2.0
    lambda_pv: float = 1.0
    proxy_bound: float = 2.0
    lambda_q: float = 0.05
    lambda_cal: float = 1.0
    cpcal_margin: float = 0.1
    energy_rank_margin: float = 0.05
    energy_constraint_slack: float = 0.01
    energy_dual_learning_rate: float = 1e-3
    energy_dual_initial_value: float = 1.0
    pair_retention_threshold: float = 0.03
    guidance_step_size: float = 0.5
    guidance_gradient_clip: float = 1.0
    gate_proxy_gain: float = 0.02
    gate_max_action_shift: float = 0.2

    @property
    def heldout_episodes_per_map(self) -> int:
        episodes, remainder = divmod(
            self.heldout_episodes_per_training_seed, len(self.heldout_maps)
        )
        if remainder:
            raise ValueError("held-out episodes must divide evenly across maps")
        return episodes

    def repair_environment_steps(self, initial_collection_steps: int) -> int:
        n_init = int(initial_collection_steps)
        if not 0 <= n_init < self.total_interaction_budget:
            raise ValueError(
                "initial collection steps must be in [0, total interaction budget)"
            )
        return self.total_interaction_budget - n_init

    def validate(self) -> None:
        if len(self.training_maps) != 20 or len(set(self.training_maps)) != 20:
            raise ValueError("DOVE paper protocol requires 20 unique training maps")
        if len(self.heldout_maps) != 10 or len(set(self.heldout_maps)) != 10:
            raise ValueError("DOVE paper protocol requires 10 unique held-out maps")
        if set(self.training_maps) & set(self.heldout_maps):
            raise ValueError("training and held-out map splits must be disjoint")
        if len(self.training_seeds) != 5 or len(set(self.training_seeds)) != 5:
            raise ValueError("DOVE paper protocol requires five independent seeds")
        if self.updates_per_environment_step != 2:
            raise ValueError("paper protocol requires two updates per repair step")
        if self.initial_collection_episodes_per_map != 5:
            raise ValueError("paper protocol requires five collection episodes per map")
        _ = self.heldout_episodes_per_map

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["training_maps"] = list(self.training_maps)
        result["heldout_maps"] = list(self.heldout_maps)
        result["training_seeds"] = list(self.training_seeds)
        result["heldout_episodes_per_map"] = self.heldout_episodes_per_map
        result["paper_method_names"] = dict(PAPER_METHOD_NAMES)
        result["paper_method_aliases"] = dict(PAPER_METHOD_ALIASES)
        return result


PROTOCOL = DOVEPaperProtocol()
PROTOCOL.validate()

TRAINING_MAPS = PROTOCOL.training_maps
HELDOUT_MAPS = PROTOCOL.heldout_maps
TRAINING_SEEDS = PROTOCOL.training_seeds


if __name__ == "__main__":
    print(json.dumps(PROTOCOL.to_dict(), indent=2, sort_keys=True))
