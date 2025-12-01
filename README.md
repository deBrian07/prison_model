Prison Gang Model (Level 1) — Mesa 3.x

Contents
- Package: `prison_model`
  - `params.py`: `Level1Params` dataclass for configuring runs.
  - `agents.py`: gang container, prisoner base, Level 1 prisoner, isolation marker.
  - `model.py`: `PrisonModelLevel1` (aliased to `PrisonModel`) with fear, isolation, and sentences.
  - `server.py`: Solara dashboard for Mesa 3.x.

Level 1 features
- 2D bounded grid with an isolation ring.
- Configurable gang count, initial affiliation share, and stay-in-place toggle. Movement uses a fixed Moore (8-neighbor) neighborhood.
- Fights with probabilistic starts and deaths; strength decides winners and feeds reputation/danger.
- Joining driven by fear, strength threshold, or accumulated violence when facing an affiliated agent.
- Isolation triggered by violence counts; timers move agents back to the yard or mark releases when sentences end.
- Metrics collected per tick: affiliation share, isolation share, fights/joins/deaths/releases, fear averages, alive count.

How to run programmatically
- Build `Level1Params`, then create `PrisonModel(params)` (alias for `PrisonModelLevel1`).
- Call `model.step()` in a loop; staged actions run via `model.agents.shuffle_do`.
- Pull results with `model.datacollector.get_model_vars_dataframe()`.

Dashboard (Mesa 3.x + Solara)
- Install dependencies: `pip install -r requirements.txt`.
- Launch the UI at http://localhost:8765: `solara run server.py`.
- Controls cover population size, gang count, movement, fight/death probabilities, strength/age/sentence distributions, join/fear thresholds, and isolation rules. Grid size is fixed at 30x30 in `server.py`.
- Grid legend: gang colors from `GANG_COLORS`, gray = unaffiliated, yellow = isolation cells.
- Charts show affiliation shares, fights/joins/deaths/releases per tick, fear trends, and alive count.

GitHub Pages (Solara SSG)
- Workflow builds with `solara ssg server.py` and publishes `build/`.
- Default base path uses `SOLARA_BASE_URL=/${REPO_NAME}/`; override if using a custom domain.
- Enable GitHub Pages with source = GitHub Actions after the workflow runs.

Railway deployment
- `Dockerfile` runs `solara run server.py --host 0.0.0.0 --port $PORT`.
- Deploy by pointing Railway at this repo; `PORT` is provided automatically. Adjust `MPLCONFIGDIR` or `SOLARA_BASE_URL` via Railway variables if needed.
