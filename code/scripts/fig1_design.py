"""Figure 1: study design schematic (no results).

One contrast: labelers used zero-shot vs supervised models trained at one hospital,
all tested across 3 hospitals. Drawn in inch coordinates on a 183 mm (7.2 in) canvas.
Output: output/figures/Fig1_study_design.{pdf,png}
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, Circle, FancyArrowPatch, Arc

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Palette (matches code/eegrl/figures.py FAMILY_COLOR)
DEC, LLM, NLI, SUP, RULE = "#00468B", "#D97848", "#7E6BAF", "#ADB6B6", "#1F2A44"
LABEL, MUTED, BORDER, LINE = "#2B2B2B", "#6B7280", "#B5BDC8", "#9CA3AF"
PANEL, SOFT = "#FAFBFD", "#F1F3F5"

W, H = 7.2, 4.1
HDR, BODY, SMALL = 8.5, 7.0, 6.3

fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(-0.3, H); ax.axis("off")


def box(x, y, w, h, fc="white", ec=BORDER, lw=0.8, r=0.06, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def t(x, y, s, size=BODY, color=LABEL, weight="normal", ha="left", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=z, **kw)


def header(x, y, n, s):
    ax.add_patch(Circle((x + 0.1, y), 0.085, fc=RULE, ec="none", zorder=5))
    t(x + 0.1, y - 0.004, str(n), size=7, color="white", weight="bold", ha="center")
    t(x + 0.25, y, s, size=HDR, weight="bold")


def hosp_tag(x, y, n, s=0.13, fc="white", ec=RULE, tc=RULE):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=fc, ec=ec, lw=0.7, zorder=5))
    t(x, y - 0.003, str(n), size=5.8, color=tc, weight="bold", ha="center", z=6)


def arrow(pts, head=True):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] if head else xs, ys[:-1] if head else ys, color=LINE, lw=0.9,
            solid_capstyle="butt", zorder=2)
    if head:
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>,head_length=4,head_width=2.4",
                                     color=LINE, lw=0.9, shrinkA=0, shrinkB=0, zorder=2))


# ---------------------------------------------------------------- layout
TOP, BOT = H - 0.06, -0.24
C1x, C1w = 0.06, 1.62
C2x, C2w = 1.98, 2.86
C3x, C3w = 5.14, 2.0
REFy0 = 2.88                       # reference block bottom
LABy1 = 2.72                       # labelers block top

# ---------------------------------------------------------------- (1) data source
box(C1x, BOT, C1w, TOP - BOT, fc=PANEL)
header(C1x + 0.08, TOP - 0.2, 1, "Data source")
cx = C1x + C1w / 2
# database glyph
cy, ew, eh, ch = 3.28, 0.46, 0.13, 0.30
ax.add_patch(Rectangle((cx - ew / 2, cy - ch / 2), ew, ch, fc="#E3E8EF", ec="none", zorder=3))
ax.plot([cx - ew / 2] * 2, [cy - ch / 2, cy + ch / 2], color=RULE, lw=0.8, zorder=4)
ax.plot([cx + ew / 2] * 2, [cy - ch / 2, cy + ch / 2], color=RULE, lw=0.8, zorder=4)
ax.add_patch(Ellipse((cx, cy - ch / 2), ew, eh, fc="#E3E8EF", ec=RULE, lw=0.8, zorder=3))
ax.add_patch(Ellipse((cx, cy + ch / 2), ew, eh, fc="#F7F9FB", ec=RULE, lw=0.8, zorder=4))
ax.add_patch(Arc((cx, cy), ew, eh, ec=RULE, lw=0.5, zorder=4, theta1=180, theta2=360))
t(cx, 2.98, "Harvard EEG Database", size=BODY, ha="center")
t(cx, 2.83, "(HEEDB)", size=SMALL, color=MUTED, ha="center")

t(C1x + 0.12, 2.56, "3 academic hospitals", size=BODY, weight="bold")
for i, name in enumerate(["I0001-S0001", "I0001-S0002", "I0003"]):
    yy = 2.36 - i * 0.19
    hosp_tag(C1x + 0.2, yy, i + 1)
    t(C1x + 0.34, yy, name, size=BODY)
yy = 2.36 - 3 * 0.19
ax.add_patch(Rectangle((C1x + 0.2 - 0.065, yy - 0.065), 0.13, 0.13, fc="white", ec=LINE,
                       lw=0.7, ls=(0, (2, 1.2)), zorder=5))
t(C1x + 0.34, yy, "I0002 excluded", size=SMALL, color=MUTED)
t(C1x + 0.34, yy - 0.14, "(empty report files)", size=SMALL, color=MUTED)

t(cx, 1.22, "99,791 reports", size=BODY + 0.5, weight="bold", ha="center")
t(cx, 1.06, "64,976 abnormal", size=SMALL, color=MUTED, ha="center")

# split bar
bx, bw, by, bh = C1x + 0.1, C1w - 0.2, 0.5, 0.17
segs = [("Train 70%", 0.70, "#E3E8EF", LABEL), ("", 0.10, "#C9D1DC", LABEL), ("", 0.20, RULE, "white")]
xx = bx
for lab, f, fc, tc in segs:
    ax.add_patch(Rectangle((xx, by), bw * f, bh, fc=fc, ec="white", lw=0.8, zorder=4))
    if lab:
        t(xx + bw * f / 2, by + bh / 2, lab, size=5.8, color=tc, ha="center", z=6)
    xx += bw * f
t(bx + bw * 0.75, by + bh + 0.08, "Val 10%", size=5.8, color=MUTED, ha="center")
t(bx, by + bh + 0.08, "Patient-level split", size=SMALL, color=MUTED)
t(bx + bw * 0.9, by - 0.08, "Test 20%", size=5.8, color=LABEL, weight="bold", ha="center")
t(cx, -0.02, "Test set: 20,093 reports", size=BODY, weight="bold", ha="center")

# ---------------------------------------------------------------- (2) reference labels
box(C2x, REFy0, C2w, TOP - REFy0, fc=PANEL)
header(C2x + 0.08, TOP - 0.2, 2, "Reference labels")
t(C2x + 0.14, 3.58, "HEEDB report-derived labels", size=BODY, weight="bold")
t(C2x + 0.14, 3.43, "Normal vs abnormal; 10 findings", size=BODY)
t(C2x + 0.14, 3.11, "(epileptiform discharges, seizure, status epilepticus, LPD, GPD,", size=SMALL, color=MUTED)
t(C2x + 0.14, 2.99, "LRDA, GRDA, burst suppression, focal/generalized slowing)", size=SMALL, color=MUTED)
# other references, right of divider? keep below as a single line
t(C2x + C2w - 0.12, 3.58, "Other references", size=BODY, weight="bold", ha="right")
t(C2x + C2w - 0.12, 3.43, "Recording annotations", size=BODY, ha="right")
t(C2x + C2w - 0.12, 3.28, "HEEDB verified tier", size=BODY, ha="right")

# ---------------------------------------------------------------- (3) labelers
box(C2x, BOT, C2w, LABy1 - BOT, fc=PANEL)
header(C2x + 0.08, LABy1 - 0.2, 3, "Labelers")

# (a) zero-shot labelers
ax_x, ax_w = C2x + 0.1, C2w - 0.2
a_top, a_bot = LABy1 - 0.38, 0.84
box(ax_x, a_bot, ax_w, a_top - a_bot, fc="white", ec=BORDER, lw=0.7, r=0.05)
t(ax_x + 0.1, a_top - 0.14, "No local training", size=BODY + 0.5, weight="bold")
t(ax_x + ax_w - 0.1, a_top - 0.14, "used as released", size=SMALL, color=MUTED, ha="right")
fam_x, mod_x = ax_x + 0.26, ax_x + 1.18
rows = [
    (DEC, "Decision models", ["Kev, Laya"]),
    (LLM, "Open LLMs", ["Gemma-4, Qwen3.8"]),
    (NLI, "Zero-shot NLI", ["DeBERTa-v3"]),
    (RULE, "Rule", ["Verdict regex"]),
]
yy = a_top - 0.36
for col, fam, mods in rows:
    ax.add_patch(Rectangle((ax_x + 0.1, yy - 0.05), 0.1, 0.1, fc=col, ec="none", zorder=5))
    t(fam_x, yy, fam, size=BODY, weight="bold", color=col if col != RULE else LABEL)
    for k, m in enumerate(mods):
        t(mod_x, yy - k * 0.14, m, size=BODY)
    yy -= 0.14 * len(mods) + 0.08

# (b) trained
b_top, b_bot = 0.74, BOT + 0.1
box(ax_x, b_bot, ax_w, b_top - b_bot, fc=SOFT, ec=BORDER, lw=0.7, r=0.05)
t(ax_x + 0.1, b_top - 0.14, "Trained on labeled reports", size=BODY + 0.5, weight="bold")
yy = b_top - 0.36
ax.add_patch(Rectangle((ax_x + 0.1, yy - 0.05), 0.1, 0.1, fc=SUP, ec="none", zorder=5))
t(fam_x, yy, "Supervised", size=BODY, weight="bold", color="#5E6A6A")
t(fam_x, yy - 0.14, "TF-IDF + LR", size=BODY)
t(fam_x, yy - 0.28, "Bio_ClinicalBERT", size=BODY)
# training scheme glyphs: pooled + one hospital at a time
gx = mod_x + 0.04
t(gx, yy, "Trained 4 times", size=SMALL, color=MUTED)
sets = [(1, 2, 3), (1,), (2,), (3,)]
gy = yy - 0.24
for j, s in enumerate(sets):
    x0 = gx + j * 0.36 + (0 if j == 0 else 0.1)
    for k in range(3):
        on = (k + 1) in s
        hosp_tag(x0 + k * 0.1, gy, k + 1, s=0.085,
                 fc=RULE if on else "white", ec=RULE if on else BORDER, tc="white" if on else BORDER)
t(gx + 0.1, gy - 0.14, "pooled", size=5.8, color=MUTED, ha="center")
t(gx + 0.46 + 0.36, gy - 0.14, "one hospital at a time", size=5.8, color=MUTED, ha="center")

# ---------------------------------------------------------------- (4) evaluation
box(C3x, BOT, C3w, TOP - BOT, fc=PANEL)
header(C3x + 0.08, TOP - 0.2, 4, "Evaluation")
t(C3x + 0.12, 3.55, "Primary: transfer across hospitals", size=BODY, weight="bold")

cs = 0.2                                   # matrix cell size
mx0, my_top = C3x + 0.46, 3.28              # left of matrix, top edge
t(mx0 + 1.5 * cs, my_top + 0.06, "Tested at", size=SMALL, color=MUTED, ha="center", va="bottom")
col_tag_y = my_top - 0.07
for j in range(3):
    hosp_tag(mx0 + (j + 0.5) * cs, col_tag_y, j + 1, s=0.11)
grid_top = col_tag_y - 0.1
# trained rows
for i in range(3):
    ry = grid_top - (i + 1) * cs
    hosp_tag(mx0 - 0.1, ry + cs / 2, i + 1, s=0.11)
    for j in range(3):
        same = i == j
        ax.add_patch(Rectangle((mx0 + j * cs, ry), cs, cs, fc=SUP if same else "white",
                               ec="#8A9494", lw=0.6, zorder=4))
t(mx0 - 0.25, grid_top - 1.5 * cs, "Trained at", size=SMALL, color=MUTED, ha="center", rotation=90)
# zero-shot row
zy = grid_top - 3 * cs - 0.1 - cs
for j in range(3):
    ax.add_patch(Rectangle((mx0 + j * cs, zy), cs, cs, fc="white", ec=LABEL, lw=0.9, zorder=4))
t(mx0 + 3 * cs + 0.14, zy + cs / 2, "Zero-shot", size=SMALL, color=LABEL)
# legend right of matrix
lx = mx0 + 3 * cs + 0.11
ax.add_patch(Rectangle((lx, grid_top - 0.62 * cs - 0.045), 0.09, 0.09, fc=SUP, ec="#8A9494", lw=0.6, zorder=4))
t(lx + 0.14, grid_top - 0.62 * cs, "Same hospital", size=SMALL)
ax.add_patch(Rectangle((lx, grid_top - 1.62 * cs - 0.045), 0.09, 0.09, fc="white", ec="#8A9494", lw=0.6, zorder=4))
t(lx + 0.14, grid_top - 1.62 * cs, "Other hospital", size=SMALL)

gy0 = zy - 0.2
t(C3x + 0.12, gy0, "Generalization gap (AUROC)", size=BODY, weight="bold")
t(C3x + 0.12, gy0 - 0.15, "Trained: same minus other hospitals", size=SMALL)
t(C3x + 0.12, gy0 - 0.28, "Zero-shot: best minus worst hospital", size=SMALL)

ax.plot([C3x + 0.12, C3x + C3w - 0.12], [gy0 - 0.43] * 2, color="#D1D5DB", lw=0.8, zorder=3)
sy = gy0 - 0.6
t(C3x + 0.12, sy, "Secondary", size=BODY, weight="bold")
for k, s in enumerate(["Verdict-masked input", "10-finding extraction",
                       "Threshold transport (post hoc)", "Reference-label audit",
                       "Speed and cost benchmark"]):
    y_ = sy - 0.2 - k * 0.19
    ax.add_patch(Circle((C3x + 0.17, y_), 0.022, fc=RULE, ec="none", zorder=5))
    t(C3x + 0.26, y_, s, size=BODY)

# ---------------------------------------------------------------- connectors
ref_mid = (REFy0 + TOP) / 2
lab_mid = (BOT + LABy1) / 2
jx = C1x + C1w + (C2x - C1x - C1w) / 2
arrow([(C1x + C1w, lab_mid), (C2x, lab_mid)])
ax.plot([jx, jx], [lab_mid, ref_mid], color=LINE, lw=0.9, zorder=2)
arrow([(jx, ref_mid), (C2x, ref_mid)])
kx = C2x + C2w + (C3x - C2x - C2w) / 2
ax.plot([C2x + C2w, kx], [ref_mid] * 2, color=LINE, lw=0.9, zorder=2)
ax.plot([C2x + C2w, kx], [lab_mid] * 2, color=LINE, lw=0.9, zorder=2)
ax.plot([kx, kx], [lab_mid, ref_mid], color=LINE, lw=0.9, zorder=2)
mid = (ref_mid + lab_mid) / 2
arrow([(kx, mid), (C3x, mid)])

out = Path(__file__).resolve().parents[2] / "output" / "figures"
out.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(out / f"Fig1_study_design.{ext}", dpi=600, bbox_inches="tight", pad_inches=0.05)
print(out / "Fig1_study_design.png")
