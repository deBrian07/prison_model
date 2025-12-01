from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Level1Params:
    """Inputs for the Level 1 prison model."""

    grid_width: int
    grid_height: int

    n_prisoners: int

    moore: bool
    allow_stay: bool

    fight_start_prob: float
    death_probability: float

    strength_mean: float
    strength_std: float
    age_mean: float
    age_std: float
    sentence_mean: float
    sentence_std: float

    violence_count_threshold_join: int
    strength_threshold_join: float
    initial_affiliated_fraction: float
    n_initial_gangs: int
    fear_threshold: float

    strictness_violence_threshold: int
    isolation_duration: int

    seed: Optional[int] = None
