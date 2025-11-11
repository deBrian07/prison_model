from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# Support running as a package or as standalone modules in the same directory
try:  # package-style imports
    from .agents import Prisoner, Gang
    from .params import Level0Params
except Exception:  # fallback to local-module imports when run from this folder
    from agents import Prisoner, Gang
    from params import Level0Params


class PrisonModel(Model):
    """
    Level 0 Mesa model following the provided outline strictly.

    Notes on open questions (need user decisions before finalizing behavior):
    - Join probability from external violence: exact mapping from
      `external_violence` vs (threshold, std) to probability is unspecified.
    - Does violence_count increase for both participants in a fight, or only
      the winner? (Level 1 specifies both; Level 0 text is ambiguous.)
    - If internal_violence ties in a fight, who is the winner?
    - On death during a fight, does the loser die with probability p, or
      is the death assignment symmetric/random?
    - Initial affiliation: what fraction of prisoners start unaffiliated vs.
      pre-affiliated, and how are they assigned to the two gangs?
    - Movement neighborhood: Von Neumann (4-neighbor) vs Moore (8-neighbor),
      and can agents choose to stay in place?
    - When both conversion and violence are eligible in a collision
      (unaffiliated vs affiliated), which is attempted first, or are both
      attempted (and in what order)?
    """

    def __init__(self, p: Level0Params):
        super().__init__()
        self.params = p
        if p.seed is not None:
            self.random.seed(p.seed)
            np.random.seed(p.seed)

        # Space (non-torus bounded grid as per "Boundaries of the prison")
        self.grid = MultiGrid(p.grid_width, p.grid_height, torus=False)

        # In Mesa 3.x there is no built-in StagedActivation; we will call
        # staged methods on the AgentSet explicitly in step().

        # Gangs: exactly two in Level 0
        self.gangs = {
            1: Gang(gang_id=1, name="Gang 1"),
            2: Gang(gang_id=2, name="Gang 2"),
        }

        self.total_deaths_this_tick = 0
        self.total_fights_this_tick = 0
        self.total_joins_this_tick = 0

        self._init_agents()

        # Data collector for Level 0 outputs
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
        # Collect an initial baseline row so charts have consistent lengths
        self.datacollector.collect(self)

    # ---------------------- Initialization ----------------------
    def _init_agents(self) -> None:
        p = self.params

        # Sanity checks for required parameters
        for name in [
            "internal_violence_mean",
            "internal_violence_std",
            "external_violence_mean",
            "external_violence_std",
            "strength_mean",
            "strength_std",
            "fight_start_prob",
            "death_probability",
            "violence_count_threshold_join",
            "external_violence_threshold_join",
            "initial_affiliated_fraction",
        ]:
            if getattr(p, name, None) is None:
                raise ValueError(f"Missing required Level 0 parameter: {name}")

        n = p.n_prisoners
        n_affil = int(round(n * p.initial_affiliated_fraction))
        # Ambiguity: how many start unaffiliated vs which gang? Ask user.
        # For now, we enforce that caller decides the fraction via params,
        # and we split the affiliated evenly across two gangs.
        n_affil_g1 = n_affil // 2
        n_affil_g2 = n_affil - n_affil_g1

        # Draw attributes
        internal = np.random.normal(p.internal_violence_mean, p.internal_violence_std, n)
        external = np.random.normal(p.external_violence_mean, p.external_violence_std, n)
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
                internal_violence=float(internal[i]),
                external_violence=float(external[i]),
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

        # Stop condition: run until all prisoners are dead
        if len(self._alive_prisoners()) == 0:
            self.running = False
        else:
            self.running = True

    # ---------------------- Movement ----------------------
    def get_move_targets(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
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
        # Same-gang: no violence, no conversion.
        if (a.gang_id is not None) and (a.gang_id == b.gang_id):
            return

        # Unaffiliated vs affiliated: try conversion first as per outline.
        if (a.gang_id is None) ^ (b.gang_id is None):
            unaff = a if a.gang_id is None else b
            aff = b if a is unaff else a
            if unaff.alive and aff.alive and aff.gang_id is not None:
                if self._should_convert(unaff, aff.gang_id):
                    self._assign_to_gang(unaff, aff.gang_id)
                    self.total_joins_this_tick += 1
                    return

        # If no conversion occurred, consider violence (not same gang).
        if self.random.random() < self.params.fight_start_prob:
            self._handle_fight(a, b)

    def _handle_fight(self, a: Prisoner, b: Prisoner) -> None:
        self.total_fights_this_tick += 1

        # Winner: stronger agent wins; ties random.
        if a.strength > b.strength:
            winner, loser = a, b
        elif b.strength > a.strength:
            winner, loser = b, a
        else:
            winner, loser = (a, b) if self.random.random() < 0.5 else (b, a)

        winner.winning_fight_count += 1

        # Ambiguity: violence_count updates in Level 0
        # Assuming both participants' violence_count increment by 1 per fight.
        a.violence_count += 1
        b.violence_count += 1

        # Death handling: loser may die with probability p
        if self.random.random() < self.params.death_probability:
            self._kill_agent(loser)

        # Update gang reputation for the winner's gang (if any):
        # base point = 1; upset bonus = +1 if winner was weaker on strength
        if winner.gang_id is not None:
            upset_bonus = 1.0 if winner.strength < loser.strength else 0.0
            g = self.gangs.get(winner.gang_id)
            if g is not None:
                g.reputation += 1.0 + upset_bonus

        # Joining by violence count threshold: ambiguous whether it requires collision
        for agent in (a, b):
            if agent.gang_id is None and agent.violence_count >= self.params.violence_count_threshold_join:
                # If both are unaffiliated, there's no target gang; requires an affiliated opponent
                # to define which gang to join. Otherwise, join the opponent's gang.
                other = b if agent is a else a
                if other.gang_id is not None:
                    self._assign_to_gang(agent, other.gang_id)
                    self.total_joins_this_tick += 1

    def _kill_agent(self, agent: Prisoner) -> None:
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
        if agent.gang_id is not None:
            gang = self.gangs.get(agent.gang_id)
            if gang is not None and agent.unique_id in gang.members:
                gang.members.remove(agent.unique_id)
        agent.gang_id = None

    def _assign_to_gang(self, agent: Prisoner, gang_id: int) -> None:
        # Remove from previous gang
        if agent.gang_id is not None:
            prev = self.gangs.get(agent.gang_id)
            if prev and agent.unique_id in prev.members:
                prev.members.remove(agent.unique_id)
        agent.gang_id = gang_id
        self.gangs[gang_id].members.add(agent.unique_id)

    # ---------------------- Conversion logic ----------------------
    def _should_convert(self, unaff: Prisoner, target_gang_id: int) -> bool:
        """
        Decide if an unaffiliated prisoner converts to the target gang during
        an unaffiliated vs affiliated collision.

        The outline specifies two triggers for Level 0 joining:
        1) External violence threshold + std -> probability of joining
        2) Violence count threshold -> join

        The exact mapping for (1) is unspecified; we need user input. For now,
        this method raises if needed parameters are absent.
        """
        # Trigger (2): handled in _handle_fight after fights update counts.

        # Trigger (1): probability depends on unaff fear (external_violence)
        # relative to threshold, and the target gang's reputation share.
        # Map to probability via a smooth logistic transform.
        if self.params.external_violence_std <= 0:
            return False

        # z-score of fear vs threshold
        z = (unaff.external_violence - self.params.external_violence_threshold_join) / (
            self.params.external_violence_std
        )

        # Reputation share of target gang among all gangs (including small epsilon)
        total_rep = sum(g.reputation for g in self.gangs.values()) + 1e-9
        target_rep = self.gangs.get(target_gang_id).reputation if target_gang_id in self.gangs else 0.0
        rep_share = target_rep / total_rep if total_rep > 0 else 0.0

        # Combine into probability
        # alpha and beta weight fear and reputation; tuned to 1.0 for Level 0 simplicity.
        x = 1.0 * z + 1.0 * (rep_share - 0.5)
        # logistic function
        p = 1.0 / (1.0 + np.exp(-x))
        return self.random.random() < p

    # ---------------------- Metrics helpers ----------------------
    def _alive_prisoners(self) -> List[Prisoner]:
        return [a for a in self.agents if isinstance(a, Prisoner) and a.alive]

    def _pct_in_gang(self, gang_id: int) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        in_g = sum(1 for a in alive if a.gang_id == gang_id)
        return in_g / len(alive)

    def _pct_unaffiliated(self) -> float:
        alive = self._alive_prisoners()
        if not alive:
            return 0.0
        unaff = sum(1 for a in alive if a.gang_id is None)
        return unaff / len(alive)

    def get_cell_prisoners(self, pos: Tuple[int, int]) -> List[Prisoner]:
        contents = list(self.grid.iter_cell_list_contents([pos]))
        if not contents:
            return []
        # MultiGrid returns a list for cell contents; flatten if needed
        if len(contents) == 1 and isinstance(contents[0], list):
            items = contents[0]
        else:
            items = contents
        return [a for a in items if isinstance(a, Prisoner)]
