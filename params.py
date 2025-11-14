from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Level0Params:
    """Inputs for the Level 0 prison model.

    Grouped by purpose (grid, movement, violence, distributions, joining, gangs).
    All values should be provided explicitly; units are arbitrary but consistent.
    """

    # Grid size
    grid_width: int
    grid_height: int

    # Population size
    n_prisoners: int

    # Movement settings
    moore: bool  # True: 8-neighborhood; False: 4-neighborhood
    allow_stay: bool  # If True, agents may choose to remain in place

    # Violence settings
    fight_start_prob: float  # Chance a fight starts on an eligible collision (0..1)
    death_probability: float  # Chance the loser dies in a fight (0..1)

    # Trait distributions (drawn per agent)
    internal_violence_mean: float
    internal_violence_std: float
    external_violence_mean: float
    external_violence_std: float
    strength_mean: float
    strength_std: float

    # Joining (conversion) rules
    violence_count_threshold_join: int
    external_violence_threshold_join: float

    # Gangs
    # Exactly two gangs exist in Level 0
    initial_affiliated_fraction: float  # Fraction initially in any gang (0..1)

    # Reproducibility
    seed: Optional[int] = None
