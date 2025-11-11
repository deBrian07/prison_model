from __future__ import annotations

from typing import Any, Dict

import solara
from mesa.visualization.solara_viz import SolaraViz
from mesa.visualization.components.altair_components import make_altair_space
from mesa.visualization.components.matplotlib_components import (
    make_mpl_plot_component,
)
from mesa.visualization.user_param import Slider

# Import local modules directly so this runs from this directory
from model import PrisonModel
from params import Level0Params
from agents import Prisoner


# Fixed grid dimensions for the dashboard. Change here to resize the view.
GRID_W = 30
GRID_H = 30


def agent_portrayal(agent: Prisoner) -> Dict[str, Any]:
    if not isinstance(agent, Prisoner):
        return {"id": 0}
    if not agent.alive:
        return {"id": agent.unique_id}
    color = "#888888"
    if agent.gang_id == 1:
        color = "#d62728"  # red
    elif agent.gang_id == 2:
        color = "#1f77b4"  # blue
    # Include rich tooltip data; X/Y duplicated in capitalized form to appear in tooltips
    x, y = agent.pos if agent.pos is not None else (None, None)
    # Optionally include gang reputation for affiliated agents
    gang_rep = 0.0
    if agent.gang_id is not None:
        g = agent.model.gangs.get(agent.gang_id)
        gang_rep = getattr(g, "reputation", 0.0) if g is not None else 0.0

    return {
        "id": agent.unique_id,
        "color": color,
        "gang": agent.gang_id if agent.gang_id is not None else 0,
        "X": x,
        "Y": y,
        "strength": round(agent.strength, 3),
        "internal_violence": round(agent.internal_violence, 3),
        "external_violence": round(agent.external_violence, 3),  # fear proxy
        "violence_count": agent.violence_count,
        "gang_reputation": round(gang_rep, 3),
        "alive": agent.alive,
    }


class AppPrisonModel(PrisonModel):
    """Thin wrapper exposing keyword params for Solara controls.

    This keeps the core model unchanged while enabling Solara's ModelCreator
    to instantiate/reset using individual parameters.
    """

    def __init__(
        self,
        *,
        n_prisoners: int,
        moore: bool,
        allow_stay: bool,
        fight_start_prob: float,
        death_probability: float,
        internal_violence_mean: float,
        internal_violence_std: float,
        external_violence_mean: float,
        external_violence_std: float,
        strength_mean: float,
        strength_std: float,
        violence_count_threshold_join: int,
        external_violence_threshold_join: float,
        initial_affiliated_fraction: float,
        seed: int | None = None,
    ) -> None:
        p = Level0Params(
            grid_width=GRID_W,
            grid_height=GRID_H,
            n_prisoners=int(n_prisoners),
            moore=bool(moore),
            allow_stay=bool(allow_stay),
            fight_start_prob=float(fight_start_prob),
            death_probability=float(death_probability),
            internal_violence_mean=float(internal_violence_mean),
            internal_violence_std=float(internal_violence_std),
            external_violence_mean=float(external_violence_mean),
            external_violence_std=float(external_violence_std),
            strength_mean=float(strength_mean),
            strength_std=float(strength_std),
            violence_count_threshold_join=int(violence_count_threshold_join),
            external_violence_threshold_join=float(external_violence_threshold_join),
            initial_affiliated_fraction=float(initial_affiliated_fraction),
            seed=seed,
        )
        super().__init__(p)


def default_model() -> AppPrisonModel:
    return AppPrisonModel(
        n_prisoners=200,
        moore=True,
        allow_stay=True,
        fight_start_prob=0.10,
        death_probability=0.05,
        internal_violence_mean=5.0,
        internal_violence_std=1.0,
        external_violence_mean=5.0,
        external_violence_std=1.0,
        strength_mean=5.0,
        strength_std=1.0,
        violence_count_threshold_join=3,
        external_violence_threshold_join=7.0,
        initial_affiliated_fraction=0.0,
        seed=None,
    )


# Solara user-adjustable parameters
model_params = {
    "n_prisoners": Slider("# Prisoners", value=200, min=10, max=600, step=1),
    "moore": {"type": "Checkbox", "label": "Moore neighborhood (8-neigh)", "value": True},
    "allow_stay": {"type": "Checkbox", "label": "Allow stay-in-place moves", "value": True},
    "fight_start_prob": Slider("Fight start probability", value=0.10, min=0.0, max=1.0, step=0.01),
    "death_probability": Slider("Death probability (loser)", value=0.05, min=0.0, max=1.0, step=0.01),
    "internal_violence_mean": Slider("Internal violence mean", value=5.0, min=0.0, max=10.0, step=0.1),
    "internal_violence_std": Slider("Internal violence std", value=1.0, min=0.0, max=5.0, step=0.1),
    "external_violence_mean": Slider("External violence mean", value=5.0, min=0.0, max=10.0, step=0.1),
    "external_violence_std": Slider("External violence std", value=1.0, min=0.0, max=5.0, step=0.1),
    "strength_mean": Slider("Strength mean", value=5.0, min=0.0, max=10.0, step=0.1),
    "strength_std": Slider("Strength std", value=1.0, min=0.0, max=5.0, step=0.1),
    "violence_count_threshold_join": Slider(
        "Violence-count threshold to join", value=3, min=0, max=20, step=1
    ),
    "external_violence_threshold_join": Slider(
        "External-violence threshold to join", value=7.0, min=0.0, max=10.0, step=0.1
    ),
    "initial_affiliated_fraction": Slider(
        "Initial affiliated fraction", value=0.0, min=0.0, max=1.0, step=0.05
    ),
    # Fixed parameters (grid size) are implied via AppPrisonModel and not user-exposed
}


@solara.component
def Page():
    model = default_model()
    # Space visualization with rich tooltips; enlarge the grid display
    def _enlarge_space(chart):
        return chart.properties(width=500, height=500)

    space_component = make_altair_space(
        agent_portrayal,
        post_process=_enlarge_space,
    )

    # Charts: affiliation shares, events per tick, and alive_count on the same page
    components = [
        (space_component, 0),
        make_mpl_plot_component(
            {"pct_gang1": "#d62728", "pct_gang2": "#1f77b4", "pct_unaffiliated": "#888888"},
            page=0,
        ),
        make_mpl_plot_component(
            {"fights_per_tick": "#000000", "joins_per_tick": "#2ca02c"},
            page=0,
        ),
        make_mpl_plot_component(
            "alive_count",
            page=0,
        ),
    ]

    return SolaraViz(
        model,
        components=components,
        model_params=model_params,
        name="Prison Gangs — Level 0",
    )

# Run with:  solara run server.py
