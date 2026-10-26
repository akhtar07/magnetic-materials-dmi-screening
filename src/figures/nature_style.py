"""Shared figure style for the paper (Nature-family conventions).

Widths follow the Nature Communications artwork guide: 88 mm single column, 120 mm one-and-a-half, 180 mm double.
Text is a sans-serif at 6-7 pt (Liberation Sans, metric-compatible with Arial), panel letters are bold
lowercase 8 pt at the top left, axes are open (left/bottom spines only), ticks point outwards,
there are no grids and no in-panel titles (the caption carries the title).

Colours (validated with the dataviz palette checker, light surface):
  verdict   overturned / confirmed / degenerate -- red / blue / neutral gray (the gray is the
            'no sign' midpoint, deliberately achromatic); all-pairs CVD dE >= 8.7
  sources   Materials Project, JARVIS-DFT, AFLOW, other -- adjacent CVD dE >= 9.2
  sizes     one-hue ordinal blue ramp for the finite-size-scaling system sizes
"""
import matplotlib
import matplotlib.pyplot as plt

MM = 1 / 25.4
SINGLE = 88 * MM
DOUBLE = 180 * MM
ONEHALF = 120 * MM

OVERTURNED, CONFIRMED, DEGENERATE = "#e34948", "#2a78d6", "#8c8b85"
INK, INK2 = "#1a1a1a", "#5f5e5a"
SOURCE_COLOR = {"MaterialsProject": "#2a78d6", "JARVIS": "#eb6834", "AFLOW": "#1baf7a", "other": "#5f5e5a"}
SIZE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans:italic",
    "mathtext.sf": "Liberation Sans",
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 6.5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.5,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "errorbar.capsize": 1.5,
    "axes.grid": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply():
    matplotlib.use("Agg")
    plt.rcParams.update(STYLE)


def label_panels(fig, axes, letters="abcdefgh", align_rows=True, tol=0.02, x=None):
    """Bold lowercase panel letters, left-aligned with the outermost extent of each panel (tick and axis
    labels included) and level with the top of its axes. Call after the layout is final. With
    align_rows, panels whose axes tops agree to `tol` (figure fraction) share the highest letter height;
    raise tol for a row whose axes differ in height (an aspect-locked image beside a bar chart). x maps a panel
    index to a fixed figure-fraction x (a schematic whose content does not start at the figure edge)."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    pos = []
    for ax in axes:
        tb = inv.transform(ax.get_tightbbox(fig.canvas.get_renderer()))
        pos.append([tb[0][0], ax.get_position().y1])
    if align_rows:
        for p in pos:
            p[1] = max(q[1] for q in pos if abs(q[1] - p[1]) < tol)
    for i, p in enumerate(pos):
        if x and i in x:
            p[0] = x[i]
    for (px, y), letter in zip(pos, letters):
        fig.text(max(px, 0.0), y + 0.01, letter, fontsize=8, fontweight="bold", ha="left", va="bottom")
