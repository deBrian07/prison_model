from __future__ import annotations

from typing import List, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

MAX_GANG_REPORTS = 6

# Allow imports to work both as a package and as local modules
try:  # package-style imports
    from .agents import PrisonerLevel1, Gang, IsolationCellMarker
    from .params import Level1Params
except Exception:  # running from this folder
    from agents import PrisonerLevel1, Gang, IsolationCellMarker
    from params import Level1Params


class PrisonModelLevel1(Model):
    """Level 1 variant with fear, isolation, sentences, and configurable gangs."""

    def __init__(self, p: Level1Params):
        super().__init__()
        self.params = p
        if p.seed is not None:
            self.random.seed(p.seed)
            np.random.seed(p.seed)

        if p.grid_width < 3 or p.grid_height < 3:
            raise ValueError("Grid must be at least 3x3 to create an isolation ring.")

        self.grid = MultiGrid(p.grid_width, p.grid_height, torus=False)
        self.isolation_cells = self._compute_isolation_cells()
        self.yard_cells = [
            (x, y)
            for x in range(self.grid.width)
            for y in range(self.grid.height)
            if (x, y) not in self.isolation_cells
        ]
        if not self.yard_cells:
            raise ValueError("No yard cells available once the isolation ring is created.")
        self._create_isolation_markers()

        self._init_gangs()

        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0
        self.total_releases_this_tick = 0

        self._init_agents()

        reporters = {
            "pct_unaffiliated": lambda m: m._pct_unaffiliated(),
            "pct_isolated": lambda m: m._pct_isolated(),
            "fights_per_tick": lambda m: m.total_fights_this_tick,
            "joins_per_tick": lambda m: m.total_joins_this_tick,
            "deaths_per_tick": lambda m: m.total_deaths_this_tick,
            "releases_per_tick": lambda m: m.total_releases_this_tick,
            "avg_fear_overall": lambda m: m._avg_fear_overall(),
            "avg_fear_unaffiliated": lambda m: m._avg_fear_unaffiliated(),
            "alive_count": lambda m: len(m._alive_prisoners()),
        }
        max_gang = max(1, min(MAX_GANG_REPORTS, int(self.params.n_initial_gangs)))
        for gid in range(1, max_gang + 1):
            reporters[f"pct_gang{gid}"] = lambda m, gid=gid: m._pct_in_gang(gid)

        self.datacollector = DataCollector(model_reporters=reporters)
        self.datacollector.collect(self)

    def _init_gangs(self) -> None:
        self.gangs = {}
        n = max(1, int(self.params.n_initial_gangs))
        for i in range(1, n + 1):
            self.gangs[i] = Gang(gang_id=i, name=f"Gang {i}")

    def _init_agents(self) -> None:
        p = self.params
        required = [
            "strength_mean",
            "strength_std",
            "age_mean",
            "age_std",
            "sentence_mean",
            "sentence_std",
            "fight_start_prob",
            "death_probability",
            "violence_count_threshold_join",
            "strength_threshold_join",
            "initial_affiliated_fraction",
            "fear_threshold",
            "strictness_violence_threshold",
            "isolation_duration",
        ]
        for name in required:
            if getattr(p, name, None) is None:
                raise ValueError(f"Missing required Level 1 parameter: {name}")

        n = p.n_prisoners
        n_gangs = max(1, int(p.n_initial_gangs))
        n_affil = int(round(n * p.initial_affiliated_fraction))
        base = n_affil // n_gangs
        remainder = n_affil % n_gangs
        gang_counts = [base + (1 if i < remainder else 0) for i in range(n_gangs)]

        indices = list(range(n))
        self.random.shuffle(indices)
        gang_assignments = {}
        cursor = 0
        for gid, count in zip(range(1, n_gangs + 1), gang_counts):
            assigned = indices[cursor : cursor + count]
            cursor += count
            for idx in assigned:
                gang_assignments[idx] = gid

        strength = np.random.normal(p.strength_mean, p.strength_std, n)
        ages = np.random.normal(p.age_mean, p.age_std, n)
        sentences = np.random.normal(p.sentence_mean, p.sentence_std, n)

        for i in range(n):
            age = float(max(1.0, ages[i]))
            sentence = int(max(1, round(sentences[i])))
            gang_id = gang_assignments.get(i)
            agent = PrisonerLevel1(
                model=self,
                strength=float(strength[i]),
                age=age,
                sentence_length=sentence,
                gang_id=gang_id,
            )
            self._place_agent_in_yard(agent)
            if gang_id is not None and gang_id in self.gangs:
                self.gangs[gang_id].members.add(agent.unique_id)

    def _compute_isolation_cells(self) -> set:
        cells = set()
        for x in range(self.grid.width):
            for y in range(self.grid.height):
                if x == 0 or y == 0 or x == self.grid.width - 1 or y == self.grid.height - 1:
                    cells.add((x, y))
        return cells

    def _create_isolation_markers(self) -> None:
        self.isolation_markers: dict[Tuple[int, int], IsolationCellMarker] = {}
        self.active_isolation_markers: set[Tuple[int, int]] = set()
        for cell in self.isolation_cells:
            marker = IsolationCellMarker(self)
            self.grid.place_agent(marker, cell)
            self.isolation_markers[cell] = marker
            self.active_isolation_markers.add(cell)

    def _deactivate_isolation_marker(self, cell: Tuple[int, int]) -> None:
        if cell not in self.active_isolation_markers:
            return
        marker = self.isolation_markers.get(cell)
        if marker is None:
            return
        try:
            self.grid.remove_agent(marker)
        except Exception:
            pass
        self.active_isolation_markers.discard(cell)

    def _activate_isolation_marker(self, cell: Tuple[int, int]) -> None:
        marker = self.isolation_markers.get(cell)
        if marker is None or cell in self.active_isolation_markers:
            return
        try:
            self.grid.place_agent(marker, cell)
        except Exception:
            pass
        else:
            self.active_isolation_markers.add(cell)

    def _place_agent_in_yard(self, agent: PrisonerLevel1) -> None:
        cell = self.random.choice(self.yard_cells)
        self.grid.place_agent(agent, cell)

    def step(self) -> None:
        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0
        self.total_releases_this_tick = 0

        self.agents.shuffle_do("plan_move")
        self.agents.shuffle_do("apply_move")
        self.agents.shuffle_do("interact")
        self.agents.shuffle_do("advance_day")
        self.datacollector.collect(self)

        self.running = len(self._alive_prisoners()) > 0

    def get_move_targets(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        if self.params.moore:
            neigh = [
                (x + dx, y + dy)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if not (dx == 0 and dy == 0)
            ]
        else:
            neigh = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        for nx, ny in neigh:
            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                if (nx, ny) not in self.isolation_cells:
                    neighbors.append((nx, ny))
        if self.params.allow_stay and (x, y) not in self.isolation_cells:
            neighbors.append((x, y))
        return neighbors

    def resolve_cell_interactions(self, pos: Tuple[int, int]) -> None:
        agents = self.get_cell_prisoners(pos)
        if len(agents) < 2:
            return
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a = agents[i]
                b = agents[j]
                if not (a.alive and b.alive):
                    continue
                self._pair_interaction(a, b)

    def _pair_interaction(self, a: PrisonerLevel1, b: PrisonerLevel1) -> None:
        if (a.gang_id is not None) and (a.gang_id == b.gang_id):
            return

        if (a.gang_id is None) ^ (b.gang_id is None):
            unaff = a if a.gang_id is None else b
            aff = b if a is unaff else a
            if unaff.alive and aff.alive and aff.gang_id is not None:
                if unaff.fear > self.params.fear_threshold:
                    self._assign_to_gang(unaff, aff.gang_id)
                    self.total_joins_this_tick += 1
                    return
                if self._should_convert(unaff, aff.gang_id):
                    self._assign_to_gang(unaff, aff.gang_id)
                    self.total_joins_this_tick += 1
                    return

        if self.random.random() < self.params.fight_start_prob:
            self._handle_fight(a, b)

    def _should_convert(self, unaff: PrisonerLevel1, target_gang_id: int) -> bool:
        if self.params.strength_std <= 0:
            return False
        z = (unaff.strength - self.params.strength_threshold_join) / max(
            1e-9, self.params.strength_std
        )
        total_danger = sum(g.danger for g in self.gangs.values()) + 1e-9
        target_danger = 0.0
        gang = self.gangs.get(target_gang_id)
        if gang is not None:
            target_danger = gang.danger
        danger_share = target_danger / total_danger if total_danger > 0 else 0.0
        x = z + (danger_share - 0.5)
        p = 1.0 / (1.0 + np.exp(-x))
        return self.random.random() < p

    def _handle_fight(self, a: PrisonerLevel1, b: PrisonerLevel1) -> None:
        self.total_fights_this_tick += 1

        if a.strength > b.strength:
            winner, loser = a, b
        elif b.strength > a.strength:
            winner, loser = b, a
        else:
            winner, loser = (a, b) if self.random.random() < 0.5 else (b, a)

        winner.winning_fight_count += 1

        a.violence_count += 1
        b.violence_count += 1
        if a.gang_id is not None and a.gang_id in self.gangs:
            self.gangs[a.gang_id].danger += 1
        if b.gang_id is not None and b.gang_id in self.gangs:
            self.gangs[b.gang_id].danger += 1

        if self.random.random() < self.params.death_probability:
            self._kill_agent(loser)

        for agent in (a, b):
            if agent.gang_id is None and agent.violence_count >= self.params.violence_count_threshold_join:
                other = b if agent is a else a
                if other.gang_id is not None:
                    self._assign_to_gang(agent, other.gang_id)
                    self.total_joins_this_tick += 1

        self._evaluate_isolation(a)
        self._evaluate_isolation(b)

    def _evaluate_isolation(self, agent: PrisonerLevel1) -> None:
        if not agent.alive or agent.pos is None:
            return
        if agent.is_isolated:
            return
        if self.params.strictness_violence_threshold <= 0:
            return
        if agent.violence_count >= self.params.strictness_violence_threshold:
            self._send_to_isolation(agent)

    def _send_to_isolation(self, agent: PrisonerLevel1) -> None:
        if not agent.alive or agent.pos is None:
            return
        if not self.isolation_cells:
            return
        cell = self.random.choice(list(self.isolation_cells))
        prev_pos = agent.pos
        self._deactivate_isolation_marker(cell)
        self.grid.move_agent(agent, cell)
        agent.pre_isolation_pos = prev_pos
        agent.start_isolation(self.params.isolation_duration, cell)

    def _release_from_isolation(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
        prev_cell = agent.last_isolation_cell
        target = agent.pre_isolation_pos
        if target is None or target in self.isolation_cells:
            cell = self.random.choice(self.yard_cells)
        else:
            cell = target
        self.grid.move_agent(agent, cell)
        agent.end_isolation()
        agent.pre_isolation_pos = None
        if prev_cell is not None:
            self._activate_isolation_marker(prev_cell)

    def _kill_agent(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
        prev_cell = agent.last_isolation_cell if agent.is_isolated else None
        agent.alive = False
        self.total_deaths_this_tick += 1
        try:
            self.grid.remove_agent(agent)
        except Exception:
            pass
        agent.remove()
        self._remove_from_gang(agent)
        agent.gang_id = None
        agent.pre_isolation_pos = None
        if prev_cell is not None:
            self._activate_isolation_marker(prev_cell)

    def _release_agent(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
        prev_cell = agent.last_isolation_cell if agent.is_isolated else None
        agent.mark_released()
        agent.alive = False
        self.total_releases_this_tick += 1
        try:
            self.grid.remove_agent(agent)
        except Exception:
            pass
        agent.remove()
        self._remove_from_gang(agent)
        agent.gang_id = None
        agent.pre_isolation_pos = None
        if prev_cell is not None:
            self._activate_isolation_marker(prev_cell)

    def _assign_to_gang(self, agent: PrisonerLevel1, gang_id: int) -> None:
        self._remove_from_gang(agent)
        agent.gang_id = gang_id
        gang = self.gangs.get(gang_id)
        if gang is not None:
            gang.members.add(agent.unique_id)
            gang.danger += agent.violence_count

    def _remove_from_gang(self, agent: PrisonerLevel1) -> None:
        if agent.gang_id is None:
            return
        gang = self.gangs.get(agent.gang_id)
        if gang is None:
            return
        if agent.unique_id in gang.members:
            gang.members.remove(agent.unique_id)
        gang.danger = max(0.0, gang.danger - agent.violence_count)

    def _alive_prisoners(self) -> List[PrisonerLevel1]:
        return [a for a in self.agents if isinstance(a, PrisonerLevel1) and a.alive]

    def _yard_prisoners(self) -> List[PrisonerLevel1]:
        return [a for a in self._alive_prisoners() if a.in_yard()]

    def _isolated_prisoners(self) -> List[PrisonerLevel1]:
        return [a for a in self._alive_prisoners() if a.is_isolated]

    def _pct_affiliated(self) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        affiliated = sum(1 for a in alive if a.gang_id is not None)
        return affiliated / len(alive)

    def _pct_unaffiliated(self) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        unaff = sum(1 for a in alive if a.gang_id is None)
        return unaff / len(alive)

    def _pct_isolated(self) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        iso = sum(1 for a in alive if a.is_isolated)
        return iso / len(alive)

    def _pct_in_gang(self, gang_id: int) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        in_gang = sum(1 for a in alive if a.gang_id == gang_id)
        return in_gang / len(alive)

    def _avg_fear_overall(self) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        fears = [a.fear for a in alive]
        return float(sum(fears) / len(fears)) if fears else 0.0

    def _avg_fear_unaffiliated(self) -> float:
        alive = self._alive_prisoners()
        unaff = [a.fear for a in alive if a.gang_id is None]
        if not unaff:
            return 0.0
        return float(sum(unaff) / len(unaff))

    def get_cell_prisoners(self, pos: Tuple[int, int]) -> List[PrisonerLevel1]:
        contents = list(self.grid.iter_cell_list_contents([pos]))
        if not contents:
            return []
        if len(contents) == 1 and isinstance(contents[0], list):
            items = contents[0]
        else:
            items = contents
        return [a for a in items if isinstance(a, PrisonerLevel1)]


# Default alias for consumers that previously imported PrisonModel
PrisonModel = PrisonModelLevel1
