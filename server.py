from __future__ import annotations

from typing import Any, Dict

import altair as alt
import solara
from solara.components import figure_altair as _figure_altair
from matplotlib.figure import Figure
from mesa.visualization.components.altair_components import make_altair_space
from mesa.visualization.components.matplotlib_components import (
    make_mpl_plot_component,
)
from mesa.visualization.user_param import Slider
from mesa.visualization import solara_viz as mesa_solara_viz
from mesa.experimental.devs.simulator import Simulator
from mesa.visualization.utils import update_counter

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
MAX_GANG_SERIES = 6


def _compact_layout(num_components: int):
    return [
        {
            "i": i,
            "w": 5,
            "h": 11,
            "moved": False,
            "x": 5 * (i % 2),
            "y": 18 * (i - i % 2),
        }
        for i in range(num_components)
    ]


mesa_solara_viz.make_initial_grid_layout = _compact_layout


def _speed_to_intervals(speed: int) -> tuple[int, int]:
    """Map a speed knob (1-100) to play/render intervals."""
    speed = max(1, min(100, int(speed)))
    play_interval = int(round(500 - (speed - 1) * 4.8))  # 500 -> 20 ms
    render_interval = max(1, int(round(5 - (speed - 1) * 0.05)))  # 5 -> 1 steps
    return play_interval, render_interval


def _speed_from_play(play_interval: int) -> int:
    """Inverse of _speed_to_intervals for initial slider value."""
    return max(1, min(100, int(round(1 + (500 - play_interval) / 4.8))))


@solara.component
def SolaraVizSpeed(
    model,
    renderer=None,
    components=[],
    *,
    play_interval: int = 100,
    render_interval: int = 1,
    simulator=None,
    model_params=None,
    name=None,
    use_threads: bool = False,
    **console_kwargs,
):
    """SolaraViz variant with a single speed slider for play+render intervals."""
    if components == "default":
        components = [
            (
                mesa_solara_viz.components_altair.make_altair_space(
                    agent_portrayal=None,
                    propertylayer_portrayal=None,
                    post_process=None,
                ),
                0,
            )
        ]
    if model_params is None:
        model_params = {}

    if not isinstance(model, solara.Reactive):
        model = solara.use_reactive(model)  # noqa: RUF100  # noqa: SH102

    reactive_model_parameters = solara.use_reactive({})
    reactive_play_interval = solara.use_reactive(play_interval)
    reactive_render_interval = solara.use_reactive(render_interval)
    reactive_use_threads = solara.use_reactive(use_threads)
    reactive_speed = solara.use_reactive(_speed_from_play(play_interval))

    def set_speed(value: int):
        speed = max(1, min(100, int(value)))
        p, r = _speed_to_intervals(speed)
        reactive_speed.set(speed)
        reactive_play_interval.set(p)
        reactive_render_interval.set(r)

    # Keep intervals aligned with the current speed on first render
    solara.use_effect(lambda: set_speed(reactive_speed.value), [])

    display_components = list(components)
    if renderer is not None:
        if isinstance(renderer, mesa_solara_viz.SpaceRenderer):
            renderer = solara.use_reactive(renderer)  # noqa: RUF100  # noqa: SH102
        display_components.insert(0, (mesa_solara_viz.create_space_component(renderer.value), 0))

    with solara.AppBar():
        solara.AppBarTitle(name if name else model.value.__class__.__name__)
        solara.lab.ThemeToggle()

    with solara.Sidebar(), solara.Column():
        with solara.Card("Controls"):
            solara.SliderInt(
                label="Speed",
                value=reactive_speed,
                on_value=set_speed,
                min=1,
                max=100,
                step=1,
            )
            solara.Text("Higher = faster (lower delay + more frequent renders)")

            if reactive_use_threads.value:
                solara.Text("Increase delay if plots are skipped")

            solara.Checkbox(
                label="Use Threads",
                value=reactive_use_threads,
                on_value=lambda v: reactive_use_threads.set(v),
            )

            if not isinstance(simulator, Simulator):
                mesa_solara_viz.ModelController(
                    model,
                    renderer=renderer,
                    model_parameters=reactive_model_parameters,
                    play_interval=reactive_play_interval,
                    render_interval=reactive_render_interval,
                    use_threads=reactive_use_threads,
                )
            else:
                mesa_solara_viz.SimulatorController(
                    model,
                    simulator,
                    renderer=renderer,
                    model_parameters=reactive_model_parameters,
                    play_interval=reactive_play_interval,
                    render_interval=reactive_render_interval,
                    use_threads=reactive_use_threads,
                )
        with solara.Card("Model Parameters"):
            mesa_solara_viz.ModelCreator(
                model, model_params, model_parameters=reactive_model_parameters
            )
        with solara.Card("Information"):
            mesa_solara_viz.ShowSteps(model.value)
        if (
            mesa_solara_viz.CommandConsole in display_components
        ):  # If command console in components show it in sidebar
            display_components.remove(mesa_solara_viz.CommandConsole)
            additional_imports = console_kwargs.get("additional_imports", {})
            with solara.Card("Command Console"):
                mesa_solara_viz.CommandConsole(
                    model.value, additional_imports=additional_imports
                )

    mesa_solara_viz.ComponentsView(display_components, model.value)


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
            moore=True,
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
        chart = chart.properties(width=300, height=300)
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
        counts = {}
        for entry in data_values:
            label = entry.get("legend_label")
            color = entry.get("color")
            if not label or not color or label in label_to_color:
                continue
            label_to_color[label] = color
            counts[label] = counts.get(label, 0) + 1
        if not label_to_color:
            return chart
        isolation_label = "Isolation Cell"
        domain_no_iso = [l for l in label_to_color.keys() if l != isolation_label]
        total_non_iso = sum(counts.get(l, 0) for l in domain_no_iso)
        if total_non_iso <= 0:
            domain_no_iso.sort()
        else:
            domain_no_iso.sort(
                key=lambda l: counts.get(l, 0) / total_non_iso, reverse=True
            )
        domain = domain_no_iso + ([isolation_label] if isolation_label in label_to_color else [])
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
                legend=alt.Legend(
                    title="Affiliation",
                    labelFontSize=12,
                    titleFontSize=13,
                    symbolSize=90,
                ),
            ),
            tooltip=tooltip_fields,
        )

    space_component = make_altair_space(
        agent_portrayal,
        post_process=_style_space_chart,
    )

    @solara.component
    def GangSharePlot(model):
        update_counter.get()
        fig = Figure(figsize=(4.5, 3.2))
        ax = fig.subplots()
        df = model.datacollector.get_model_vars_dataframe()
        n = getattr(getattr(model, "params", None), "n_initial_gangs", 1)
        n = max(1, min(MAX_GANG_SERIES, int(n)))
        for gid in range(1, n + 1):
            col = f"pct_gang{gid}"
            if col in df.columns:
                gang_name = getattr(getattr(model, "gangs", {}).get(gid), "name", f"Gang {gid}")
                ax.plot(
                    df.index,
                    df[col],
                    label=gang_name,
                    color=GANG_COLORS[(gid - 1) % len(GANG_COLORS)],
                )
        if "pct_unaffiliated" in df.columns:
            ax.plot(df.index, df["pct_unaffiliated"], label="pct_unaffiliated", color="#888888")
        if "pct_isolated" in df.columns:
            ax.plot(df.index, df["pct_isolated"], label="pct_isolated", color=ISOLATION_COLOR)
        ax.set_xlabel("Step")
        ax.set_ylabel("Share")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=9)
        solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")

    components = [
        (space_component, 0),
        (GangSharePlot, 0),
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

    return SolaraVizSpeed(
        default_model_level1(),
        components=components,
        model_params=model_params_level1,
        name="Prison Gangs — Level 1",
    )

# Run with:  solara run server.py
