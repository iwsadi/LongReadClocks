"""
plot_style.py
~~~~~~~~~~~~~
Shared matplotlib style and color palette for all LongReadClock figures.

Apply once at the top of any notebook or script:
    from longreadclock.plot_style import apply_style, COLORS, PLATFORM_ORDER

This ensures every figure uses the same font, line weights, and color palette
as the published paper (Nature/Science-compatible, PDF-embeddable fonts).
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict

# ─────────────────────────────────────────────────────────────────────────────
# Platform / cohort color palette
# ─────────────────────────────────────────────────────────────────────────────

COLORS: Dict[str, str] = {
    # Sequencing platforms
    "ONT":       "#185FA5",   # deep blue
    "Revio":     "#0F6E56",   # deep green
    "Sequel2e":  "#993C1D",   # terracotta
    "V7_mixed":  "#6B6B6B",   # grey (older mixed batch)

    # Cell types
    "bulk":        "#444444",
    "Myeloid":     "#D4813A",
    "Lymphoid":    "#4A90D9",
    "T_Cell":      "#2196F3",
    "B_Cell":      "#9C27B0",
    "NK_Cell":     "#00BCD4",
    "Monocyte":    "#FF9800",
    "Granulocyte": "#8BC34A",
}

# Canonical ordering for legends and axes
PLATFORM_ORDER = ["ONT", "Revio", "Sequel2e", "V7_mixed"]

CELL_TYPE_ORDER = [
    "bulk", "Myeloid", "Lymphoid",
    "T_Cell", "B_Cell", "NK_Cell", "Monocyte", "Granulocyte",
]

SOURCE_ORDER = [
    "JHU_ONT", "BCM_ONT",
    "BI_Sequel2e", "UW_Sequel2e", "BCM_Sequel2e",
    "Broad_Revio", "BCM_Revio", "UW_Revio", "HA_Revio",
    "V7_Base",
]

SOURCE_LABELS = {
    "JHU_ONT":      "JHU · ONT",
    "BCM_ONT":      "BCM · ONT",
    "BI_Sequel2e":  "Broad · Sequel2e",
    "UW_Sequel2e":  "UW · Sequel2e",
    "BCM_Sequel2e": "BCM · Sequel2e",
    "Broad_Revio":  "Broad · Revio",
    "BCM_Revio":    "BCM · Revio",
    "UW_Revio":     "UW · Revio",
    "HA_Revio":     "HA · Revio",
    "V7_Base":      "V7 base · mixed PacBio",
}

SOURCE_MARKERS = {
    "JHU_ONT":     "o",
    "BCM_ONT":     "s",
    "BI_Sequel2e": "^",
    "UW_Sequel2e": "v",
    "BCM_Sequel2e":"P",
    "Broad_Revio": "o",
    "BCM_Revio":   "s",
    "UW_Revio":    "^",
    "HA_Revio":    "D",
    "V7_Base":     "X",
}

# ─────────────────────────────────────────────────────────────────────────────
# rcParams preset
# ─────────────────────────────────────────────────────────────────────────────

_RCPARAMS = {
    # Font — Arial is the standard for nature-style figures;
    # falls back gracefully to Helvetica then DejaVu Sans.
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            9,
    "axes.titlesize":       9,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "figure.titlesize":     10,

    # Lines and axes
    "axes.linewidth":       0.6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    0.6,
    "ytick.major.width":    0.6,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.2,

    # Scatter / patches
    "patch.linewidth":      0.5,

    # Legend
    "legend.frameon":       False,
    "legend.borderpad":     0.3,
    "legend.labelspacing":  0.3,

    # Output quality
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.facecolor":    "white",

    # PDF / PS — embeds fonts so figures render identically in Word / Illustrator
    "pdf.fonttype":         42,
    "ps.fonttype":          42,

    # White background
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
}


def apply_style() -> None:
    """Apply the LongReadClock figure style globally.

    Call this once at the top of each notebook or script:

        from longreadclock.plot_style import apply_style
        apply_style()
    """
    mpl.rcParams.update(_RCPARAMS)


def get_cell_type_color(cell_type: str) -> str:
    """Return the canonical color for a cell type (falls back to grey)."""
    return COLORS.get(cell_type, "#888888")


def get_platform_color(platform: str) -> str:
    """Return the canonical color for a sequencing platform."""
    return COLORS.get(platform, "#888888")
