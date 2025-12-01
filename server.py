from __future__ import annotations

from typing import Any, Dict

import altair as alt
import solara
from solara.components import figure_altair as _figure_altair
from mesa.visualization.solara_viz import SolaraViz
from mesa.visualization.components.altair_components import make_altair_space
from mesa.visualization.components.matplotlib_components import (
    make_mpl_plot_component,
)
from mesa.visualization.user_param import Slider

"""Solara (Mesa 3.x) dashboard to visualize the Level 1 model."""

# Import local modules directly so this runs from this directory
from model import PrisonModelLevel1
from params import Level1Params
from agents import Prisoner, IsolationCellMarker


# Provide Altair utils fallback needed by Mesa with Altair 5.x
if not hasattr(alt.utils, "infer_vegalite_type_for_pandas"):
    alt.utils.infer_vegalite_type_for_pandas = alt.utils.infer_vegalite_type  # type: ignore[attr-defined]


# Patch Solara's FigureAltair to handle newer Altair/VegaLite MIME types (v6).
@solara.component
def _FigureAltairCompat(
    chart,
    on_click=None,
    on_hover=None,
):
    with alt.renderers.enable("mimetype"):
        bundle = chart._repr_mimebundle_()[0]
        keys = [
            "application/vnd.vegalite.v5+json",
            "application/vnd.vegalite.v4+json",
            "application/vnd.vegalite.v6+json",
            "application/vnd.vegalite.v6.json",
        ]
        spec = None
        for key in keys:
            if key in bundle:
                spec = bundle[key]
                break
        if spec is None:
            raise KeyError(f"{keys} not in mimebundle:\\n\\n{bundle}")
        return solara.widgets.VegaLite.element(
            spec=spec,
            on_click=on_click,
            listen_to_click=on_click is not None,
            on_hover=on_hover,
            listen_to_hover=on_hover is not None,
        )


# Apply monkey patch so Mesa's SpaceAltair uses the compatible implementation.
solara.FigureAltair = _FigureAltairCompat  # type: ignore[attr-defined]
_figure_altair.FigureAltair = _FigureAltairCompat  # type: ignore[attr-defined]


# Fixed grid dimensions for the dashboard. Change here to resize the view.
GRID_W = 30
GRID_H = 30
GANG_COLORS = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#bcbd22",
]
ISOLATION_COLOR = "#f4d03f"


def agent_portrayal(agent: Prisoner) -> Dict[str, Any]:
    """Encode how a Prisoner appears on the grid and in tooltips."""

    def _fmt(val, precision: int | None = None) -> str:
        if val is None:
            return "N/A"
        if precision is not None:
            try:
                return f"{float(val):.{precision}f}"
            except (TypeError, ValueError):
                return str(val)
        return str(val)

    if isinstance(agent, IsolationCellMarker):
        x, y = agent.pos if agent.pos is not None else (None, None)
        return {
            "id": f"isolation-{x}-{y}",
            "color": ISOLATION_COLOR,
            "legend_label": "Isolation Cell",
            "X": x,
            "Y": y,
            "Affiliation": "Isolation Cell",
            "Status": "Isolation Cell",
            "Strength": "N/A",
            "Violence Count": "N/A",
            "Winning Fights": "N/A",
            "Fear": "N/A",
            "Age": "N/A",
            "Sentence Remaining": "N/A",
            "gang": -1,
        }
    if not isinstance(agent, Prisoner):
        return {"id": 0}
    if not agent.alive:
        return {"id": agent.unique_id}
    color = "#888888"
    label = "Unaffiliated"
    status = "In Yard"
    if getattr(agent, "is_isolated", False):
        status = "In Isolation"
    if not agent.alive:
        status = "Deceased"
    if agent.gang_id is not None:
        color = GANG_COLORS[(agent.gang_id - 1) % len(GANG_COLORS)]
        gang = agent.model.gangs.get(agent.gang_id) if hasattr(agent.model, "gangs") else None
        label = getattr(gang, "name", None) or f"Gang {agent.gang_id}"
    else:
        gang = None
    # Include tooltip data; X/Y duplicated capitalized for Altair tooltips
    x, y = agent.pos if agent.pos is not None else (None, None)
    fear = getattr(agent, "fear", None)
    age = getattr(agent, "age", None)
    sentence_remaining = getattr(agent, "sentence_remaining", None)

    portrayal = {
        "id": agent.unique_id,
        "color": color,
        "legend_label": label,
        "gang": agent.gang_id if agent.gang_id is not None else 0,
        "X": x,
        "Y": y,
        "alive": agent.alive,
        "is_isolated": getattr(agent, "is_isolated", False),
    }
    tooltip_fields = {
        "Affiliation": label,
        "Status": status,
        "Strength": _fmt(agent.strength, 3),
        "Violence Count": _fmt(agent.violence_count),
        "Winning Fights": _fmt(agent.winning_fight_count),
        "Fear": _fmt(fear, 3),
        "Age": _fmt(age, 2),
        "Sentence Remaining": _fmt(sentence_remaining),
    }
    portrayal.update(tooltip_fields)
    return portrayal


class AppPrisonModelLevel1(PrisonModelLevel1):
    """Wrapper exposing keyword params for Level 1 dashboard controls."""

    def __init__(
        self,
        *,
        n_prisoners: int,
        n_initial_gangs: int,
        initial_affiliated_fraction: float,
        moore: bool,
        allow_stay: bool,
        fight_start_prob: float,
        death_probability: float,
        strength_mean: float,
        strength_std: float,
        age_mean: float,
        age_std: float,
        sentence_mean: float,
        sentence_std: float,
        violence_count_threshold_join: int,
        strength_threshold_join: float,
        fear_threshold: float,
        strictness_violence_threshold: int,
        isolation_duration: int,
        seed: int | None = None,
    ) -> None:
        p = Level1Params(
            grid_width=GRID_W,
            grid_height=GRID_H,
            n_prisoners=int(n_prisoners),
            n_initial_gangs=int(n_initial_gangs),
            initial_affiliated_fraction=float(initial_affiliated_fraction),
            moore=bool(moore),
            allow_stay=bool(allow_stay),
            fight_start_prob=float(fight_start_prob),
            death_probability=float(death_probability),
            strength_mean=float(strength_mean),
            strength_std=float(strength_std),
            age_mean=float(age_mean),
            age_std=float(age_std),
            sentence_mean=float(sentence_mean),
            sentence_std=float(sentence_std),
            violence_count_threshold_join=int(violence_count_threshold_join),
            strength_threshold_join=float(strength_threshold_join),
            fear_threshold=float(fear_threshold),
            strictness_violence_threshold=int(strictness_violence_threshold),
            isolation_duration=int(isolation_duration),
            seed=seed,
        )
        super().__init__(p)


def default_model_level1() -> AppPrisonModelLevel1:
    return AppPrisonModelLevel1(
        n_prisoners=200,
        n_initial_gangs=3,
        initial_affiliated_fraction=0.2,
        moore=True,
        allow_stay=True,
        fight_start_prob=0.1,
        death_probability=0.05,
        strength_mean=5.0,
        strength_std=1.0,
        age_mean=35.0,
        age_std=8.0,
        sentence_mean=365.0,
        sentence_std=60.0,
        violence_count_threshold_join=3,
        strength_threshold_join=7.0,
        fear_threshold=0.1,
        strictness_violence_threshold=6,
        isolation_duration=5,
        seed=None,
    )

model_params_level1 = {
    "n_prisoners": Slider("# Prisoners", value=200, min=50, max=600, step=10),
    "n_initial_gangs": Slider("# Initial gangs", value=3, min=1, max=6, step=1),
    "initial_affiliated_fraction": Slider(
        "Initial affiliated fraction", value=0.2, min=0.0, max=1.0, step=0.05
    ),
    "moore": {"type": "Checkbox", "label": "Moore neighborhood (8-neigh)", "value": True},
    "allow_stay": {"type": "Checkbox", "label": "Allow stay-in-place moves", "value": True},
    "fight_start_prob": Slider("Fight start probability", value=0.10, min=0.0, max=1.0, step=0.01),
    "death_probability": Slider("Death probability (loser)", value=0.05, min=0.0, max=1.0, step=0.01),
    "strength_mean": Slider("Strength mean", value=5.0, min=0.0, max=10.0, step=0.1),
    "strength_std": Slider("Strength std", value=1.0, min=0.0, max=5.0, step=0.1),
    "age_mean": Slider("Age mean", value=35.0, min=18.0, max=80.0, step=1.0),
    "age_std": Slider("Age std", value=8.0, min=1.0, max=25.0, step=1.0),
    "sentence_mean": Slider("Sentence mean (days)", value=365.0, min=30.0, max=2000.0, step=10.0),
    "sentence_std": Slider("Sentence std", value=60.0, min=5.0, max=500.0, step=5.0),
    "violence_count_threshold_join": Slider(
        "Violence-count threshold to join", value=3, min=0, max=20, step=1
    ),
    "strength_threshold_join": Slider(
        "Strength threshold to join", value=7.0, min=0.0, max=10.0, step=0.1
    ),
    "fear_threshold": Slider("Fear threshold to join", value=0.1, min=0.0, max=1.0, step=0.01),
    "strictness_violence_threshold": Slider(
        "Violence count for isolation", value=6, min=1, max=30, step=1
    ),
    "isolation_duration": Slider("Isolation duration (days)", value=5, min=1, max=30, step=1),
}


@solara.component
def Page():
    """Solara page: grid + charts + controls for the Level 1 model."""

    def _style_space_chart(chart):
        chart = chart.properties(width=250, height=250)
        data_values = getattr(getattr(chart, "data", None), "values", None)
        if not data_values and hasattr(chart, "layer"):
            for layer_chart in getattr(chart, "layer", []):
                layer_values = getattr(getattr(layer_chart, "data", None), "values", None)
                if layer_values:
                    data_values = layer_values
                    break
        if not data_values:
            return chart
        label_to_color = {}
        for entry in data_values:
            label = entry.get("legend_label")
            color = entry.get("color")
            if not label or not color or label in label_to_color:
                continue
            label_to_color[label] = color
        if not label_to_color:
            return chart
        domain = list(label_to_color.keys())
        color_range = [label_to_color[label] for label in domain]
        tooltip_fields = [
            alt.Tooltip("Affiliation:N"),
            alt.Tooltip("Status:N"),
            alt.Tooltip("Strength:N"),
            alt.Tooltip("Violence Count:N"),
            alt.Tooltip("Winning Fights:N"),
            alt.Tooltip("Fear:N"),
            alt.Tooltip("Age:N"),
            alt.Tooltip("Sentence Remaining:N"),
        ]
        return chart.encode(
            color=alt.Color(
                "legend_label:N",
                scale=alt.Scale(domain=domain, range=color_range),
                legend=alt.Legend(title="Affiliation"),
            ),
            tooltip=tooltip_fields,
        )

    space_component = make_altair_space(
        agent_portrayal,
        post_process=_style_space_chart,
    )

    components = [
        (space_component, 0),
        make_mpl_plot_component(
            {
                "pct_affiliated": "#d62728",
                "pct_unaffiliated": "#888888",
                "pct_isolated": ISOLATION_COLOR,
            },
            page=0,
        ),
        make_mpl_plot_component(
            {
                "fights_per_tick": "#000000",
                "joins_per_tick": "#2ca02c",
                "deaths_per_tick": "#d62728",
                "releases_per_tick": "#1f77b4",
            },
            page=0,
        ),
        make_mpl_plot_component(
            {
                "avg_fear_overall": "#9467bd",
                "avg_fear_unaffiliated": "#bcbd22",
            },
            page=0,
        ),
        make_mpl_plot_component("alive_count", page=0),
    ]

    return SolaraViz(
        default_model_level1(),
        components=components,
        model_params=model_params_level1,
        name="Prison Gangs — Level 1",
    )

# Run with:  solara run server.py
