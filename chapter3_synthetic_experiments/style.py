"""Figure style of the thesis: Computer Modern through LaTeX, Okabe-Ito palette, 15 cm text width.

If LaTeX is not available (it needs ``latex``, ``dvipng`` and the ``cm-super`` fonts), ``setup``
falls back to Matplotlib's own mathtext with Computer Modern glyphs: the layout and the data are the
same, only the typesetting of the text differs.
"""
import io
import logging
import shutil
import sys

import matplotlib
if "ipykernel" not in sys.modules:      # scripts: no display; notebooks keep the inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("fontTools").setLevel(logging.ERROR)   # silence warnings about font timestamps

OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#7F7F7F"}
TEXTWIDTH = 5.9   # inches, about 15 cm


def latex_available():
    """True only if a test figure renders with LaTeX in both PDF and PNG."""
    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        with plt.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(1, 1)); ax.set_title(r"$\bar c$ \% test")
            fig.savefig(io.BytesIO(), format="png"); fig.savefig(io.BytesIO(), format="pdf")
        return True
    except Exception:
        return False
    finally:
        plt.close("all")


def setup(usetex=None):
    """Set the style; returns True if LaTeX is used. ``usetex=None`` detects it."""
    usetex = latex_available() if usetex is None else usetex
    plt.rcParams.update({
        "text.usetex": usetex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if usetex else ["cmr10", "DejaVu Serif"],
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": not usetex,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4,
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })
    return usetex


def save(fig, path):
    """Write the figure as PDF and PNG (``path`` without extension)."""
    fig.savefig(f"{path}.pdf"); fig.savefig(f"{path}.png", dpi=200)
    plt.close(fig)
