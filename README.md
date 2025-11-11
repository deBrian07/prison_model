Prison Gang Model (Level 0) — Mesa

Contents
- Package: `mesa_impl`
  - `params.py`: Level 0 parameters dataclass
  - `agents.py`: `Prisoner` agent and `Gang` structure
  - `model.py`: `PrisonModel` implementing Level 0 dynamics

What’s implemented (Level 0)
- 2D bounded grid (no wrap), synchronous daily ticks via `StagedActivation`.
- Prisoners move randomly 1 step/day; neighborhood and stay-behavior are configurable.
- Two gangs only; membership tracked; dead agents removed from grid and gangs.
- On cell collisions, for each pair:
  - Violence can occur with user-set probability except same-gang pairs.
  - Fight winner determined by higher `internal_violence` (ties random).
  - Death on fight occurs with user-set probability (currently assigned to loser).
  - `violence_count` increments by 1 for both fighters.
  - Conversion allowed for unaffiliated vs affiliated pairs via:
    - external-violence threshold comparison (placeholder mapping; see questions), and
    - violence-count threshold when opposing an affiliated agent.
- Data collected per tick: % in gang 1, % in gang 2, % unaffiliated, number of fights, joins.

Open questions (please specify — no assumptions made beyond what's written)
1) Join probability from external violence:
   - You specified: threshold + standard deviation; “within that SD each person has a % chance (based on normal dist) of joining”.
   - Please provide the exact mapping formula or rule (e.g., prob = CDF or PDF transform, or step function).
2) Violence count increments in Level 0:
   - Should both fighters’ `violence_count` increase by 1 per fight, or only the winner’s? (Level 1 explicitly says both.)
3) Fight tie-breaking:
   - If `internal_violence` is equal, how do we pick the winner? Random acceptable?
4) Death assignment in a fight:
   - Should the loser die with probability `death_probability`, or should either participant die with that probability? If the latter, how allocated?
5) Initial affiliation:
   - What fraction starts affiliated vs unaffiliated, and how to split across Gang 1 vs 2? Current code expects a fraction and splits evenly.
6) Movement specifics:
   - Von Neumann (4-neighbor) or Moore (8-neighbor) moves? Can prisoners stay in place?
7) Conversion vs violence order for unaffiliated vs affiliated collisions:
   - Attempt conversion before violence, after, or both in some order?
8) “Timer -> disappear” line in Level 0 interactions:
   - Do Level 0 agents have a sentence timer? If yes, what distribution/parameters?

How to run (example outline)
- Instantiate `Level0Params` with your chosen values, then `PrisonModel(params)`.
- Call `model.step()` in a loop, and retrieve metrics via `model.datacollector.get_model_vars_dataframe()`.

