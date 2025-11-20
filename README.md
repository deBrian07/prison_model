Prison Gang Model (Level 0) — Mesa 3.x

Contents
- Package: `prison_model`
  - `params.py`: Level 0 parameters dataclass
  - `agents.py`: `Prisoner` agent and `Gang` structure
  - `model.py`: `PrisonModel` implementing Level 0 dynamics
  - Top-level `server.py`: Solara-based dashboard for Mesa 3.x

What’s implemented (Level 0)
- 2D bounded grid (no wrap), staged daily ticks via Model.agents (AgentSet) using `shuffle_do` for: plan_move → apply_move → interact → advance_day.
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
- Data collected per tick: % in gang 1, % in gang 2, % unaffiliated, fights per tick, joins per tick.

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

How to run (programmatic)
- Instantiate `Level0Params` with your chosen values, then `PrisonModel(params)`.
- Call `model.step()` in a loop; staged actions are handled internally via `self.agents.shuffle_do("...")`.
- Retrieve metrics via `model.datacollector.get_model_vars_dataframe()`.

How to run the dashboard (Mesa 3.x + Solara)
- Install dependencies (Mesa with viz extras): `pip install -r requirements.txt`.
- Launch the UI (serves on http://localhost:8765): `solara run server.py`.
- Controls:
  - Sliders/checkboxes configure Level 0 parameters (grid is fixed at 30x30; change in `server.py`).
  - Grid shows prisoners: red = Gang 1, blue = Gang 2, gray = unaffiliated.
  - Charts display group shares and fights/joins per tick.

Mesa 3.x migration notes (what changed here)
- No `mesa.time` schedulers: replaced with explicit staged calls on `model.agents` (AgentSet).
- Agent IDs are auto-managed by Mesa; `Prisoner` no longer takes `unique_id`.
- Removing agents uses `agent.remove()` after `grid.remove_agent(agent)`.
- MultiGrid cell access uses a helper to aggregate contents in 3.x (`model.get_cell_prisoners`).

Note
- External-violence-based conversion is wired but disabled pending your exact probability rule; the corresponding slider has no effect yet.

GitHub Pages (static build via Solara SSG)
- Workflow: `.github/workflows/gh-pages.yml` installs `solara-enterprise[ssg]`, runs `solara ssg server.py`, and publishes the `build/` output.
- Base path: `SOLARA_BASE_URL` is set to `/${REPO_NAME}/` for default GitHub Pages URLs (`https://<user>.github.io/<repo>/`). Override if using a custom domain.
- To publish: push to `main` or trigger the workflow manually, then enable GitHub Pages with source = GitHub Actions in the repo settings.

Railway deployment (full backend)
- A production-ready `Dockerfile` is included; it installs the Python dependencies and runs `solara run server.py --host 0.0.0.0 --port $PORT`.
- Steps to deploy:
  1. Push this repo to GitHub (or your preferred git host).
  2. In Railway, create a New Project → “Deploy from GitHub repo”, select this repo, and leave the default `Dockerfile` build plan.
  3. Railway injects `PORT`; no extra env vars are required, but you can change `MPLCONFIGDIR` or `SOLARA_BASE_URL` in the Railway dashboard if desired.
  4. After the build succeeds, Railway will expose the public URL where the Solara dashboard runs with a persistent backend.
- CLI alternative: install the Railway CLI, run `railway login`, `railway init`, then `railway up` to push the Docker build from your machine.

Level 1 dashboard usage
- Select “Level 1” from the toggle at the top of the Solara page to enable the expanded model (fear, isolation ring, multi-gang support).
- Additional sliders configure: # of gangs, age/sentence distributions, fear threshold, isolation strictness/duration, etc.
- Charts show overall affiliation + isolation shares, fights/joins/deaths/releases per tick, fear trends, spacing of gangs, and alive count.
