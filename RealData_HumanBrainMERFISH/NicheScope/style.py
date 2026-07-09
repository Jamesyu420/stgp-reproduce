from __future__ import annotations

from dataclasses import dataclass

import matplotlib as mpl


@dataclass(frozen=True)
class VarPartColors:
    age: str = "#E64B35"
    region: str = "#4DBBD5"
    both: str = "#3C5488"
    residuals: str = "#BFBFBF"


# Method palette (mirrors the mouse pipeline; PCA/NMF kept for back-compat).
METHOD_COLORS = {
    "stGP":       "#E64B35",
    "PCA":        "#91D1C2",
    "NMF":        "#F39B7F",
    "SpatialPCA": "#4DBBD5",
    "MEFISTO":    "#8491B4",
    "STAMP":      "#B09C85",
    "Popari":     "#00A087",
}


def set_nature_style(*, font: str | None = None) -> None:
    font_stack = [f for f in [font, "Arial", "Helvetica", "DejaVu Sans"] if f]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_stack,
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 300,
            "savefig.dpi": 400,
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.linewidth": 1.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 14,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "lines.linewidth": 1.5,
        }
    )
