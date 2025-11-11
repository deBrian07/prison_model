from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from mesa import Agent


@dataclass
class Gang:
    gang_id: int
    name: str
    members: set = field(default_factory=set)

    def size(self) -> int:
        return len(self.members)


class Prisoner(Agent):
    def __init__(
        self,
        unique_id: int,
        model,
        *,
        internal_violence: float,
        external_violence: float,
        strength: float,
        gang_id: Optional[int] = None,
    ):
        super().__init__(unique_id, model)
        self.internal_violence = internal_violence
        self.external_violence = external_violence
        self.strength = strength
        self.gang_id = gang_id

        self.violence_count = 0
        self.winning_fight_count = 0
        self.alive = True

        self._planned_move: Optional[Tuple[int, int]] = None
        self._interacted_this_tick = False

    def in_yard(self) -> bool:
        return self.alive

    def can_interact(self) -> bool:
        return self.in_yard()

    def plan_move(self) -> None:
        if not self.alive:
            self._planned_move = None
            return
        neighbors = self.model.get_move_targets(self.pos)
        if not neighbors:
            self._planned_move = None
            return
        self._planned_move = self.random.choice(neighbors)

    def apply_move(self) -> None:
        if self._planned_move is None:
            return
        self.model.grid.move_agent(self, self._planned_move)
        self._planned_move = None

    def interact(self) -> None:
        self._interacted_this_tick = False
        if not self.can_interact():
            return
        cell_agents: List[Prisoner] = [
            a for a in self.model.grid.get_cell_list_contents([self.pos]) if isinstance(a, Prisoner)
        ]
        if len(cell_agents) < 2:
            return
        if self._interacted_this_tick:
            return
        self.model.resolve_cell_interactions(self.pos)

    def advance_day(self) -> None:
        # Level 0 has no isolation or sentences.
        return
