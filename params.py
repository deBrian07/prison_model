from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Level0Params:
    # Space
    grid_width: int
    grid_height: int
    # Agents
    n_prisoners: int
    # Movement
    moore: bool  # True for 8-neighborhood, False for 4-neighborhood
    allow_stay: bool  # whether staying in place is allowed on a move
    # Violence
    fight_start_prob: float  # probability a fight starts on eligible collision (0..1)
    death_probability: float  # probability the loser dies (0..1)
    # Distributions (all must be provided explicitly)
    internal_violence_mean: float
    internal_violence_std: float
    external_violence_mean: float
    external_violence_std: float
    strength_mean: float
    strength_std: float
    # Joining (conversion)
    violence_count_threshold_join: int
    external_violence_threshold_join: float
    # Gangs
    # Level 0 contains exactly 2 gangs by design.
    initial_affiliated_fraction: float  # fraction of agents initially in any gang (0..1)
    # Random seed
    seed: Optional[int] = None
