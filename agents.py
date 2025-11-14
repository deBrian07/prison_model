from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from mesa import Agent

"""Agent definitions for the prison model.

- Gang: simple container for gang identity, members, and reputation.
- Prisoner: a Mesa Agent with movement, interaction, and basic state.
"""


@dataclass
class Gang:
    """Represents a gang and tracks its members and reputation."""

    gang_id: int
    name: str
    members: set = field(default_factory=set)
    reputation: float = 0.0

    def size(self) -> int:
        """Current number of members (alive and affiliated)."""
        return len(self.members)


class Prisoner(Agent):
    """An inmate in the prison yard.

    Attributes of interest:
    - internal_violence: propensity to fight within the prison
    - external_violence: fear/pressure from outside (used for conversion)
    - strength: used to determine fight outcomes
    - gang_id: current gang affiliation (None means unaffiliated)
    - alive: whether the prisoner is still in the simulation
    """

    def __init__(
        self,
        model,
        *,
        internal_violence: float,
        external_violence: float,
        strength: float,
        gang_id: Optional[int] = None,
    ):
        super().__init__(model)
        self.internal_violence = internal_violence
        self.external_violence = external_violence
        self.strength = strength
        self.gang_id = gang_id

        self.violence_count = 0
        self.winning_fight_count = 0
        self.alive = True

        # Movement planned for this tick (set in plan_move, used in apply_move)
        self._planned_move: Optional[Tuple[int, int]] = None
        # Used to prevent duplicate interactions in a cell during a tick
        self._interacted_this_tick = False

    def in_yard(self) -> bool:
        """Returns True if the prisoner is alive and present on the grid."""
        return self.alive

    def can_interact(self) -> bool:
        """Whether the prisoner is eligible to interact this tick."""
        return self.in_yard()

    def plan_move(self) -> None:
        """Choose a legal target cell (or stay) uniformly at random."""
        if not self.alive:
            self._planned_move = None
            return
        neighbors = self.model.get_move_targets(self.pos)
        if not neighbors:
            self._planned_move = None
            return
        self._planned_move = self.random.choice(neighbors)

    def apply_move(self) -> None:
        """Execute any planned move and clear it."""
        if self._planned_move is None:
            return
        self.model.grid.move_agent(self, self._planned_move)
        self._planned_move = None

    def interact(self) -> None:
        """Trigger one round of pairwise interactions for this cell.

        Only the lowest-ID agent in a cell starts the interaction routine to
        avoid processing the same pairs multiple times.
        """
        self._interacted_this_tick = False
        if not self.can_interact():
            return
        cell_agents: List[Prisoner] = self.model.get_cell_prisoners(self.pos)
        if len(cell_agents) < 2:
            return
        # Let a single canonical agent drive interactions in this cell
        if self.unique_id != min(a.unique_id for a in cell_agents):
            return
        if self._interacted_this_tick:
            return
        self.model.resolve_cell_interactions(self.pos)

    def advance_day(self) -> None:
        """Daily cleanup stage. Level 0 has no extra effects."""
        return
