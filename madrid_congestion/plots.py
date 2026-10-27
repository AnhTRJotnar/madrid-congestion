from typing import Literal

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from madrid_congestion.config import FIGURES_DIR

# Figure style shared by all report figures, taken from the authors' Ms Pac-Man report notebook:
# matplotlib defaults, named colours, a light grid, a framed legend, a "Topic: scope" title,
# reference lines grey and dashed, tight layout and 160 dpi.
LINE_WIDTH = 2.5
REFERENCE_WIDTH = 1.5
GRID_ALPHA = 0.25
DPI = 160

# Blue always means flow and orange always occupancy, in every figure.
FLOW = "blue"
OCCUPANCY = "orange"

# Descriptive groups (sensor types, districts) never borrow the flow or occupancy colour.
CATEGORY_COLORS = ["teal", "brown", "black"]
REFERENCE = "gray"  # thresholds, cut-offs and tolerances
EVENT = "red"  # high-occupancy slots and onsets


def style_axes(ax: Axes, grid_axis: Literal["both", "x", "y"] = "both") -> None:  # the values ax.grid accepts
    ax.grid(axis=grid_axis, alpha=GRID_ALPHA)


def save_figure(fig: Figure, file_name: str) -> None:
    # Call fig.tight_layout() first; Figure 3.5 needs it with room for its figure legend.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / file_name, dpi=DPI)
