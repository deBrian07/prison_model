from __future__ import annotations

from typing import List, Optional, Tuple
import itertools

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Allow imports to work both as a package and as local modules
try:  # package-style imports
    from .agents import Prisoner, PrisonerLevel1, Gang
    from .params import Level0Params, Level1Params
except Exception:  # running from this folder
    from agents import Prisoner, PrisonerLevel1, Gang
    from params import Level0Params, Level1Params


class PrisonModel(Model):
    """Level 0 prison gang model.

    What it does (short):
    - Places prisoners on a 2D bounded grid.
    - Each tick: move → interact (convert and/or fight) → advance day.
    - Tracks gang membership, fights, joins, deaths, and simple gang reputation.

    Assumptions made here (to keep Level 0 simple):
    - Fight winner: higher strength; ties broken randomly.
    - Violence counts: both fighters increment by 1 per fight.
    - Deaths: loser dies with probability `death_probability`.
    - Conversion order: attempt conversion first for unaffiliated vs affiliated; otherwise consider fights.

    Open items to confirm later:
    - Exact formula for strength-based join probability.
    - Initial gang split and any other tie-breaking preferences.
    """

    def __init__(self, p: Level0Params):
        super().__init__()
        self.params = p
        if p.seed is not None:
            self.random.seed(p.seed)
            np.random.seed(p.seed)

        # Non-wrapping grid (prison has walls)
        self.grid = MultiGrid(p.grid_width, p.grid_height, torus=False)

        # Mesa 3.x has no built-in staged scheduler; we call staged methods
        # on the AgentSet explicitly in step().

        # Exactly two gangs in Level 0
        self.gangs = {
            1: Gang(gang_id=1, name="Gang 1"),
            2: Gang(gang_id=2, name="Gang 2"),
        }

        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0

        self._init_agents()

        # Collect key model-level metrics each tick
        self.datacollector = DataCollector(
            model_reporters={
                "pct_gang1": lambda m: m._pct_in_gang(1),
                "pct_gang2": lambda m: m._pct_in_gang(2),
                "pct_unaffiliated": lambda m: m._pct_unaffiliated(),
                "fights_per_tick": lambda m: m.total_fights_this_tick,
                "joins_per_tick": lambda m: m.total_joins_this_tick,
                "alive_count": lambda m: len(m._alive_prisoners()),
            }
        )
        # Collect an initial baseline row so charts start aligned
        self.datacollector.collect(self)

    # ---------------------- Initialization ----------------------
    def _init_agents(self) -> None:
        """Create prisoners, set initial affiliations, and place them on the grid."""
        p = self.params

        # Sanity checks for required parameters
        for name in [
            "strength_mean",
            "strength_std",
            "fight_start_prob",
            "death_probability",
            "violence_count_threshold_join",
            "strength_threshold_join",
            "initial_affiliated_fraction",
        ]:
            if getattr(p, name, None) is None:
                raise ValueError(f"Missing required Level 0 parameter: {name}")

        n = p.n_prisoners
        n_affil = int(round(n * p.initial_affiliated_fraction))
        # Split the initially affiliated evenly across the two gangs
        n_affil_g1 = n_affil // 2
        n_affil_g2 = n_affil - n_affil_g1

        # Draw attributes
        strength = np.random.normal(p.strength_mean, p.strength_std, n)

        # Create agents
        idxs = list(range(n))
        self.random.shuffle(idxs)
        affil_idxs = set(idxs[:n_affil])
        g1_idxs = set(list(affil_idxs)[:n_affil_g1])
        g2_idxs = affil_idxs - g1_idxs

        for i in range(n):
            gid: Optional[int]
            if i in g1_idxs:
                gid = 1
            elif i in g2_idxs:
                gid = 2
            else:
                gid = None
            agent = Prisoner(
                model=self,
                strength=float(strength[i]),
                gang_id=gid,
            )
            # Place uniformly at random
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            if gid is not None:
                self.gangs[gid].members.add(agent.unique_id)

    # ---------------------- Step ----------------------
    def step(self) -> None:
        """Run one simulation day: move, interact, clean up, collect data."""
        # Reset tick counters
        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0

        # Execute stages explicitly using the model's AgentSet
        self.agents.shuffle_do("plan_move")
        self.agents.shuffle_do("apply_move")
        self.agents.shuffle_do("interact")
        self.agents.shuffle_do("advance_day")
        self.datacollector.collect(self)

        # Stop when everyone is dead
        if len(self._alive_prisoners()) == 0:
            self.running = False
        else:
            self.running = True

    # ---------------------- Movement ----------------------
    def get_move_targets(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return legal neighboring cells; include staying put if allowed."""
        x, y = pos
        neighbors = []
        # Neighborhood
        if self.params.moore:
            neigh = [
                (x + dx, y + dy)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if not (dx == 0 and dy == 0)
            ]
        else:
            neigh = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        # Filter to grid bounds
        for nx, ny in neigh:
            if 0 <= nx < self.grid.width and 0 <= ny < self.grid.height:
                neighbors.append((nx, ny))
        if self.params.allow_stay:
            neighbors.append((x, y))
        return neighbors

    # ---------------------- Interactions ----------------------
    def resolve_cell_interactions(self, pos: Tuple[int, int]) -> None:
        """Process each unordered pair in a cell exactly once."""
        agents = self.get_cell_prisoners(pos)
        if len(agents) < 2:
            return
        # Consider each unordered pair once
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a = agents[i]
                b = agents[j]
                if not (a.alive and b.alive):
                    continue
                self._pair_interaction(a, b)

    def _pair_interaction(self, a: Prisoner, b: Prisoner) -> None:
        """Decide whether two agents convert or fight, then apply effects."""
        # Same gang: no conversion and no violence
        if (a.gang_id is not None) and (a.gang_id == b.gang_id):
            return

        # Unaffiliated vs affiliated: attempt conversion first
        if (a.gang_id is None) ^ (b.gang_id is None):
            unaff = a if a.gang_id is None else b
            aff = b if a is unaff else a
            if unaff.alive and aff.alive and aff.gang_id is not None:
                if self._should_convert(unaff, aff.gang_id):
                    self._assign_to_gang(unaff, aff.gang_id)
                    self.total_joins_this_tick += 1
                    return

        # If no conversion occurred, consider violence
        if self.random.random() < self.params.fight_start_prob:
            self._handle_fight(a, b)

    def _handle_fight(self, a: Prisoner, b: Prisoner) -> None:
        """Resolve a fight: choose winner, update counts, handle possible death."""
        self.total_fights_this_tick += 1

        # Winner: stronger agent wins; ties random
        if a.strength > b.strength:
            winner, loser = a, b
        elif b.strength > a.strength:
            winner, loser = b, a
        else:
            winner, loser = (a, b) if self.random.random() < 0.5 else (b, a)

        winner.winning_fight_count += 1

        # Both participants' violence_count increment by 1 per fight
        a.violence_count += 1
        b.violence_count += 1
        if a.gang_id is not None and a.gang_id in self.gangs:
            self.gangs[a.gang_id].danger += 1
        if b.gang_id is not None and b.gang_id in self.gangs:
            self.gangs[b.gang_id].danger += 1

        # Loser may die with probability p
        if self.random.random() < self.params.death_probability:
            self._kill_agent(loser)

        # Update gang reputation for the winner's gang (if any):
        # base point = 1; upset bonus = +1 if winner was weaker on strength
        if winner.gang_id is not None:
            upset_bonus = 1.0 if winner.strength < loser.strength else 0.0
            g = self.gangs.get(winner.gang_id)
            if g is not None:
                g.reputation += 1.0 + upset_bonus

        # Join by violence-count threshold (only if opponent is affiliated)
        for agent in (a, b):
            if agent.gang_id is None and agent.violence_count >= self.params.violence_count_threshold_join:
                # If both are unaffiliated, there is no target gang; otherwise join opponent's gang.
                other = b if agent is a else a
                if other.gang_id is not None:
                    self._assign_to_gang(agent, other.gang_id)
                    self.total_joins_this_tick += 1

    def _kill_agent(self, agent: Prisoner) -> None:
        """Remove agent from the simulation and from gang membership."""
        if not agent.alive:
            return
        agent.alive = False
        self.total_deaths_this_tick += 1
        # Remove from grid and model registry
        try:
            self.grid.remove_agent(agent)
        except Exception:
            pass
        # Deregister agent from the model
        agent.remove()
        # Remove from gang membership if needed
        self._remove_from_gang(agent)
        agent.gang_id = None

    def _assign_to_gang(self, agent: Prisoner, gang_id: int) -> None:
        """Move agent to the given gang, updating membership sets."""
        # Remove from previous gang if present
        self._remove_from_gang(agent)
        agent.gang_id = gang_id
        gang = self.gangs.get(gang_id)
        if gang is not None:
            gang.members.add(agent.unique_id)
            gang.danger += agent.violence_count

    def _remove_from_gang(self, agent: Prisoner) -> None:
        if agent.gang_id is None:
            return
        gang = self.gangs.get(agent.gang_id)
        if gang is None:
            return
        if agent.unique_id in gang.members:
            gang.members.remove(agent.unique_id)
        gang.danger = max(0.0, gang.danger - agent.violence_count)

    # ---------------------- Conversion logic ----------------------
    def _should_convert(self, unaff: Prisoner, target_gang_id: int) -> bool:
        """Return True if an unaffiliated agent joins the target gang now.

        Triggers in Level 0:
        1) External violence vs threshold (mapped to a probability here).
        2) Violence-count threshold (handled after fights, not here).
        """
        # Trigger (2): handled in _handle_fight after fights update counts.

        # Trigger (1): probability depends on unaff strength relative to threshold,
        # and the target gang's reputation share.
        # Map to probability via a smooth logistic transform.
        if self.params.strength_std <= 0:
            return False

        # z-score of fear vs threshold
        z = (unaff.strength - self.params.strength_threshold_join) / (
            self.params.strength_std
        )

        # Reputation share of target gang among all gangs (with small epsilon)
        total_rep = sum(g.reputation for g in self.gangs.values()) + 1e-9
        target_rep = self.gangs.get(target_gang_id).reputation if target_gang_id in self.gangs else 0.0
        rep_share = target_rep / total_rep if total_rep > 0 else 0.0

        # Combine into probability via a simple logistic transform
        x = 1.0 * z + 1.0 * (rep_share - 0.5)
        # logistic function
        p = 1.0 / (1.0 + np.exp(-x))
        return self.random.random() < p

    # ---------------------- Metrics helpers ----------------------
    def _alive_prisoners(self) -> List[Prisoner]:
        """All living prisoners currently in the model."""
        return [a for a in self.agents if isinstance(a, Prisoner) and a.alive]

    def _pct_in_gang(self, gang_id: int) -> float:
        """Share of alive prisoners affiliated with a specific gang."""
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        in_g = sum(1 for a in alive if a.gang_id == gang_id)
        return in_g / len(alive)

    def _pct_unaffiliated(self) -> float:
        """Share of alive prisoners not in any gang."""
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        unaff = sum(1 for a in alive if a.gang_id is None)
        return unaff / len(alive)

    def get_cell_prisoners(self, pos: Tuple[int, int]) -> List[Prisoner]:
        """All Prisoner agents in the given grid cell."""
        contents = list(self.grid.iter_cell_list_contents([pos]))
        if not contents:
            return []
        # MultiGrid returns a list for cell contents; flatten if needed
        if len(contents) == 1 and isinstance(contents[0], list):
            items = contents[0]
        else:
            items = contents
        return [a for a in items if isinstance(a, Prisoner)]


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

        self._init_gangs()

        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0
        self.total_releases_this_tick = 0

        self._init_agents()

        self.datacollector = DataCollector(
            model_reporters={
                "pct_affiliated": lambda m: m._pct_affiliated(),
                "pct_unaffiliated": lambda m: m._pct_unaffiliated(),
                "pct_isolated": lambda m: m._pct_isolated(),
                "fights_per_tick": lambda m: m.total_fights_this_tick,
                "joins_per_tick": lambda m: m.total_joins_this_tick,
                "deaths_per_tick": lambda m: m.total_deaths_this_tick,
                "releases_per_tick": lambda m: m.total_releases_this_tick,
                "avg_fear_overall": lambda m: m._avg_fear_overall(),
                "avg_fear_unaffiliated": lambda m: m._avg_fear_unaffiliated(),
                "avg_same_gang_distance": lambda m: m._avg_same_gang_distance(),
                "alive_count": lambda m: len(m._alive_prisoners()),
            }
        )
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
        if agent.is_isolated:
            return
        if self.params.strictness_violence_threshold <= 0:
            return
        if agent.violence_count >= self.params.strictness_violence_threshold:
            self._send_to_isolation(agent)

    def _send_to_isolation(self, agent: PrisonerLevel1) -> None:
        if not self.isolation_cells:
            return
        cell = self.random.choice(list(self.isolation_cells))
        self.grid.move_agent(agent, cell)
        agent.start_isolation(self.params.isolation_duration)

    def _release_from_isolation(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
        cell = self.random.choice(self.yard_cells)
        self.grid.move_agent(agent, cell)
        agent.end_isolation()

    def _kill_agent(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
        agent.alive = False
        self.total_deaths_this_tick += 1
        try:
            self.grid.remove_agent(agent)
        except Exception:
            pass
        agent.remove()
        self._remove_from_gang(agent)
        agent.gang_id = None

    def _release_agent(self, agent: PrisonerLevel1) -> None:
        if not agent.alive:
            return
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

    def _avg_same_gang_distance(self) -> float:
        alive = [a for a in self._alive_prisoners() if a.gang_id is not None and a.pos is not None]
        if not alive:
            return 0.0
        distances = []
        by_gang: dict[int, List[Tuple[int, int]]] = {}
        for agent in alive:
            by_gang.setdefault(agent.gang_id, []).append(agent.pos)
        for coords in by_gang.values():
            if len(coords) < 2:
                continue
            for (x1, y1), (x2, y2) in itertools.combinations(coords, 2):
                distances.append(abs(x1 - x2) + abs(y1 - y2))
        if not distances:
            return 0.0
        return float(sum(distances) / len(distances))

    def get_cell_prisoners(self, pos: Tuple[int, int]) -> List[PrisonerLevel1]:
        contents = list(self.grid.iter_cell_list_contents([pos]))
        if not contents:
            return []
        if len(contents) == 1 and isinstance(contents[0], list):
            items = contents[0]
        else:
            items = contents
        return [a for a in items if isinstance(a, PrisonerLevel1)]
