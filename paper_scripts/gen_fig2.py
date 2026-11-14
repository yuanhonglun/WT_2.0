#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate manuscript Fig. 2 - the ASFAMProcessor data-processing pipeline schematic.

Revision 261019 edition of manuscript/fig 2/gen_fig2.py:
  * palette: Okabe-Ito subset (paper_palette) - blue = MS1 evidence / MS1-detected,
    orange = MS2 product-ion evidence, purple = MS2-only / inferred precursor;
    box headers are neutral so the hues keep a single meaning each.
  * "Precursor m/z assignment" card now shows the v1.4 source order:
    strict MS1 peak -> relaxed MS1 peak -> neutral-loss consensus ->
    window-centroid fallback (low confidence).  No accuracy claim for inferred
    precursors.
  * "Alignment & export" card describes the v1.4 identity rule in one sentence.

Layout (unchanged): three columns on top (Data loading / Feature detection /
Precursor m/z assignment) -> a full-width Deduplication band with four
sub-cards -> two columns at the bottom (Annotation / Alignment & export).
Canvas: A4 portrait width x 8.8 in, Arial 8 pt titles / 6-7 pt notes, 1 pt lines.

Outputs "Fig 2.svg / .pdf / .png" (300 dpi) and legend_fig2.txt into --out-dir
(default revision_261019/figures/fig2).
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402

# ---- typography / output config --------------------------------------------
pp.apply_rcparams({
    "svg.fonttype": "none",   # keep text as text in the SVG
    "pdf.fonttype": 42,       # embed TrueType in PDF (editable text)
})

# ---- palette -----------------------------------------------------------------
INK      = pp.INK          # body text
NEUTRAL  = pp.NEUTRAL      # box stroke / arrows / partner marks
RULE     = pp.RULE         # axes / grid
SUBIN    = "#6B6258"       # muted italic note
HDR      = "#E4EAF0"       # neutral header band for every stage box
C_MS1,  C_MS1_F  = pp.BLUE,   pp.light(pp.BLUE)      # MS1 evidence / MS1-detected
C_MS2,  C_MS2_F  = pp.ORANGE, pp.light(pp.ORANGE)    # MS2 product-ion evidence
C_MS2_D, C_MS2_L = pp.dark(pp.ORANGE), pp.mix(pp.ORANGE, 0.55)   # co-eluting traces
C_INF,  C_INF_F  = pp.PURPLE, pp.light(pp.PURPLE)    # MS2-only / inferred precursor

# ---- canvas: A4 portrait width, ~2/3 height ----------------------------------
FW, FH = 8.27, 8.8
fig = plt.figure(figsize=(FW, FH), dpi=300)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ASPECT = FH / FW
ROUND = 0.010

# =============================================================================
# primitive helpers
# =============================================================================
def draw_box(x0, y0, w, h, title):
    """White rounded box + neutral rounded-top header band. Returns the inner
    content rect (ix, iy, iw, ih) with iy the BOTTOM edge of the content area."""
    hh = 0.030
    body = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={ROUND}",
                          mutation_aspect=1 / ASPECT, fc="white", ec="none", zorder=1)
    ax.add_patch(body)
    hdr = Rectangle((x0, y0 + h - hh), w, hh, fc=HDR, ec="none", zorder=2)
    ax.add_patch(hdr); hdr.set_clip_path(body)
    ax.add_line(Line2D([x0, x0 + w], [y0 + h - hh, y0 + h - hh], color=NEUTRAL,
                       lw=0.75, zorder=3))
    edge = FancyBboxPatch((x0, y0), w, h,
                          boxstyle=f"round,pad=0,rounding_size={ROUND}",
                          mutation_aspect=1 / ASPECT, fc="none", ec=NEUTRAL,
                          lw=1.0, zorder=4)
    ax.add_patch(edge)
    ax.text(x0 + w / 2, y0 + h - hh / 2, title, fontsize=8, fontweight="bold",
            color=INK, ha="center", va="center", zorder=6)
    padx, pady = 0.010, 0.013
    return (x0 + padx, y0 + pady, w - 2 * padx, h - hh - 2 * pady)

def T(x, y, s, size=7.5, weight="normal", color=INK, style="normal",
      ha="left", va="baseline", z=6):
    ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va,
            style=style, zorder=z)

def axes_frame(x, y, w, h, yaxis=True):
    ax.add_line(Line2D([x, x + w], [y, y], color=RULE, lw=0.55, zorder=4))
    if yaxis:
        ax.add_line(Line2D([x, x], [y, y + h], color=RULE, lw=0.55, zorder=4))

def eic(x, y, w, h, center, sigma, amp, color, lw=1.0, fill=False, z=5, n=64):
    """Gaussian chromatographic peak across [x, x+w], apex height amp*h."""
    t = np.linspace(0, 1, n)
    g = amp * np.exp(-0.5 * ((t - center) / sigma) ** 2)
    X, Y = x + t * w, y + g * h
    if fill:
        ax.fill_between(X, y, Y, color=color, alpha=0.10, lw=0, zorder=z - 1)
    ax.plot(X, Y, color=color, lw=lw, solid_capstyle="round", zorder=z)

def sticks(x, y, w, h, peaks, color, lw=1.0, z=5):
    """Stick spectrum; peaks = [(frac_x, frac_h), ...]. Returns x-positions."""
    xs = []
    for fx, fh in peaks:
        X = x + fx * w
        ax.add_line(Line2D([X, X], [y, y + fh * h], color=color, lw=lw,
                           solid_capstyle="butt", zorder=z))
        xs.append(X)
    return xs

def mirror(x, y, w, h, top, bot, ctop, cbot, matched=(), ax_color=RULE,
           ax_lw=0.6, ax_z=4):
    """Mirror plot: top sticks up / bottom sticks down about a mid baseline."""
    ymid = y + h / 2
    ax.add_line(Line2D([x, x + w], [ymid, ymid], color=ax_color, lw=ax_lw, zorder=ax_z))
    ax.add_line(Line2D([x, x], [y, y + h], color=ax_color, lw=ax_lw, zorder=ax_z))
    for fx in matched:                                   # faint match guides
        X = x + fx * w
        ax.add_line(Line2D([X, X], [y + 0.02 * h, y + 0.98 * h], color=NEUTRAL,
                           lw=0.5, ls=(0, (1.4, 1.4)), alpha=0.8, zorder=4))
    for fx, fh in top:
        X = x + fx * w
        ax.add_line(Line2D([X, X], [ymid, ymid + fh * (h / 2)], color=ctop,
                           lw=1.0, zorder=6))
    for fx, fh in bot:
        X = x + fx * w
        ax.add_line(Line2D([X, X], [ymid, ymid - fh * (h / 2)], color=cbot,
                           lw=1.0, zorder=6))

def file_icon(x, y, w, h, label, fill, stroke, z=6):
    fx, fy = w * 0.30, h * 0.30                           # folded corner
    body = [(x, y), (x, y + h), (x + w - fx, y + h),
            (x + w, y + h - fy), (x + w, y), (x, y)]
    ax.add_patch(Polygon(body, closed=True, fc=fill, ec=stroke, lw=0.8, zorder=z))
    ax.add_patch(Polygon([(x + w - fx, y + h), (x + w - fx, y + h - fy),
                          (x + w, y + h - fy)], closed=True, fc="white",
                         ec=stroke, lw=0.6, zorder=z + 1))
    T(x + w / 2, y - 0.011, label, size=7, color=stroke, ha="center", va="top")

def arrow(x0, y0, x1, y1, color=NEUTRAL, lw=1.0, scale=10, z=3):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0,
                                shrinkB=0, mutation_scale=scale), zorder=z)

def elbow(pts, color=NEUTRAL, lw=1.0, scale=10):
    for i in range(len(pts) - 2):
        (a, b), (c, d) = pts[i], pts[i + 1]
        ax.add_line(Line2D([a, c], [b, d], color=color, lw=lw, zorder=3))
    arrow(pts[-2][0], pts[-2][1], pts[-1][0], pts[-1][1], color=color, lw=lw, scale=scale)

# =============================================================================
# geometry
# =============================================================================
ML = 0.022
GX = 0.020
CW = 1 - 2 * ML                                   # content width
W1 = (CW - 2 * GX) / 3                            # row-1 column width
COL = [ML, ML + W1 + GX, ML + 2 * (W1 + GX)]
CXc = [c + W1 / 2 for c in COL]

R1_Y, R1_H = 0.690, 0.295                         # row 1 (bottom-y, height)
# Compact deduplication / annotation / alignment blocks. The canvas is unchanged; the empty
# band left below row 3 is cropped when the figure is saved (CROP_Y).
DE_Y, DE_H = 0.456, 0.196                         # dedup band
R3_Y, R3_H = 0.220, 0.198                         # row 3
CROP_Y = R3_Y - 0.022                             # figure fraction cut off at the bottom
W3 = (CW - GX) / 2
COL3 = [ML, ML + W3 + GX]
CX3 = [c + W3 / 2 for c in COL3]

# =============================================================================
# STAGE 1 - Data loading
# =============================================================================
ix, iy, iw, ih = draw_box(COL[0], R1_Y, W1, R1_H, "Data loading")
top = iy + ih
fic_w, fic_h = 0.030, 0.034
file_icon(ix + 0.004, top - fic_h, fic_w, fic_h, ".mzML", C_MS1_F, C_MS1)
arrow(ix + 0.004 + fic_w + 0.006, top - fic_h / 2 - 0.002,
      ix + 0.004 + fic_w + 0.028, top - fic_h / 2 - 0.002, color=C_MS1, scale=8)
T(ix + iw, top - 0.006, "parse + centroid", size=7.5, color=C_MS1,
  weight="bold", ha="right")
scy = top - fic_h - 0.052
T(ix, scy + 0.014, "scan cycle", size=8, weight="bold", color=INK)
sv_b = scy - 0.046
axes_frame(ix + 0.004, sv_b, iw - 0.010, 0.040, yaxis=True)
sticks(ix + 0.004, sv_b, iw - 0.010, 0.040,
       [(0.08, 0.5), (0.2, 0.85), (0.33, 0.4), (0.46, 0.95), (0.58, 0.55),
        (0.7, 0.75), (0.82, 0.35), (0.93, 0.6)], C_MS1, lw=0.9)
T(ix + iw - 0.006, sv_b - 0.002, "MS1 survey", size=7, color=C_MS1, ha="right",
  va="top")
st_b = scy - 0.120
axes_frame(ix + 0.004, st_b, iw - 0.010, 0.044, yaxis=True)
nwin = 8
for i in range(nwin):
    wx = ix + 0.012 + i * (iw - 0.030) / nwin
    wy = st_b + 0.002 + i * (0.040 / (nwin - 1))
    ax.add_patch(Rectangle((wx, wy), (iw - 0.030) / nwin * 0.62, 0.008,
                           fc=C_MS2_F, ec=C_MS2, lw=0.7, zorder=6))
T(ix + 0.006, st_b - 0.006, "HR-MRM MS2  ·  1 Da stepped windows", size=7,
  color=C_MS2_D, va="top")

# =============================================================================
# STAGE 2 - Feature detection (MS1/MS2 dual-driven)
# =============================================================================
ix, iy, iw, ih = draw_box(COL[1], R1_Y, W1, R1_H, "Feature detection")
top = iy + ih
EW, EH = 0.072, 0.052
SPX = ix + 0.114
SPW = iw - 0.118
T(ix, top - 0.004, "MS2-driven", size=8, weight="bold", color=C_MS2_D, va="top")
ex, ey = ix + 0.004, top - 0.086
axes_frame(ex, ey, EW, EH)
for col, amp, sig in [(C_MS2, 0.94, 0.15), (C_MS2_D, 0.72, 0.17), (C_MS2_L, 0.85, 0.14)]:
    eic(ex, ey, EW, EH, 0.5, sig, amp, col, lw=0.9)
T(ix + 0.002, ey - 0.006, "co-eluting product ions", size=6, color=C_MS2_D,
  ha="left", va="top")
arrow(ex + EW + 0.006, ey + EH / 2, ex + EW + 0.030, ey + EH / 2, color=C_MS2, scale=8)
axes_frame(SPX, ey, SPW, EH)
sticks(SPX, ey, SPW, EH,
       [(0.14, 0.55), (0.3, 0.95), (0.46, 0.35), (0.62, 0.72), (0.8, 0.48)],
       C_MS2, lw=1.0)
T(SPX + 0.002, ey - 0.006, "MS2 spectrum", size=6, color=C_MS2_D, va="top")
my = top - 0.150
T(ix, my + 0.004, "MS1-driven", size=8, weight="bold", color=C_MS1, va="top")
ey2 = my - 0.054
axes_frame(ix + 0.004, ey2, EW, EH)
eic(ix + 0.004, ey2, EW, EH, 0.5, 0.17, 0.9, C_MS1, lw=1.0, fill=True)
T(ix + 0.004 + EW / 2, ey2 - 0.006, "survey peak", size=6, color=C_MS1,
  ha="center", va="top")
axm = ix + 0.004 + EW
arrow(axm + 0.004, ey2 + EH / 2, axm + 0.030, ey2 + EH / 2, color=C_MS1, scale=8)
T(axm + 0.017, ey2 + EH / 2 + 0.014, "MS2", size=6, color=SUBIN, ha="center", va="bottom")
T(axm + 0.017, ey2 + EH / 2 - 0.020, "deconv.", size=6, color=SUBIN, ha="center", va="bottom")
axes_frame(SPX, ey2, SPW, EH)
sticks(SPX, ey2, SPW, EH,
       [(0.18, 0.6), (0.36, 0.92), (0.56, 0.4), (0.8, 0.66)], C_MS1, lw=1.0)
T(SPX + 0.002, ey2 - 0.006, "assigned MS2", size=6, color=C_MS1, va="top")
T(ix + iw / 2, iy + 0.004, "union → candidate features", size=8, weight="bold",
  color=INK, ha="center", va="bottom")

# =============================================================================
# STAGE 3 - Precursor m/z assignment: first source that succeeds (v1.4 order)
# =============================================================================
ix, iy, iw, ih = draw_box(COL[2], R1_Y, W1, R1_H, "Precursor m/z assignment")
top = iy + ih
T(ix, top - 0.004, "First source that succeeds, in order:", size=7.5,
  weight="bold", color=INK, va="top")

# steps 1-2: an MS1 peak is matched to the product-ion peak group
GW, GH = 0.044, 0.036
g_y = top - 0.072
axes_frame(ix + 0.002, g_y, GW, GH)
for col, amp, sig in [(C_MS2, 0.9, 0.14), (C_MS2_D, 0.7, 0.16), (C_MS2_L, 0.82, 0.13)]:
    eic(ix + 0.002, g_y, GW, GH, 0.5, sig, amp, col, lw=0.8)
T(ix + 0.002 + GW / 2, g_y - 0.005, "peak group", size=6, color=C_MS2_D, ha="center", va="top")
mx1 = ix + 0.002 + GW + 0.024
axes_frame(mx1, g_y, GW, GH)
eic(mx1, g_y, GW, GH, 0.5, 0.16, 0.92, C_MS1, lw=1.0, fill=True)
T(mx1 + GW / 2, g_y - 0.005, "MS1 peak", size=6, color=C_MS1, ha="center", va="top")
arrow(mx1 - 0.003, g_y + GH * 0.55, ix + 0.002 + GW + 0.003, g_y + GH * 0.55, color=C_MS1, scale=8)
tx = mx1 + GW + 0.010
T(tx, top - 0.038, "1  strict MS1 peak", size=7, weight="bold", color=C_MS1, va="top")
T(tx, top - 0.058, "2  relaxed MS1 peak", size=7, weight="bold", color=C_MS1, va="top")

# step 3: neutral-loss consensus (MS2-only)
n_y = top - 0.138
nx, nw, nh = ix + 0.002, 0.112, 0.040
axes_frame(nx, n_y, nw, nh, yaxis=False)
prec_x = nx + 0.90 * nw
prec_top = n_y + 0.74 * nh
for fxp, fhp in [(0.10, 0.78), (0.22, 0.50), (0.34, 0.90), (0.48, 0.62)]:
    fxx = nx + fxp * nw
    ax.add_line(Line2D([fxx, fxx], [n_y, n_y + fhp * nh], color=C_MS2, lw=1.2, zorder=6))
    ax.add_line(Line2D([fxx, prec_x], [n_y + fhp * nh, prec_top], color=NEUTRAL,
                       lw=0.5, ls=(0, (1.4, 1.4)), zorder=5))
ax.add_line(Line2D([prec_x, prec_x], [n_y, prec_top], color=C_INF, lw=1.6, zorder=6))
T(nx + 0.28 * nw, n_y - 0.005, "product ions", size=6, color=C_MS2_D, ha="center", va="top")
T(prec_x, n_y - 0.005, "inferred", size=6, color=C_INF, ha="center", va="top")
T(tx, top - 0.106, "3  neutral-loss consensus", size=7, weight="bold", color=C_INF, va="top")
T(tx + 0.013, top - 0.119, "vote across product ions", size=6, color=SUBIN, va="top")

# step 4: isolation-window centroid fallback (low confidence)
w_y = top - 0.192
ww, wh = 0.112, 0.026
axes_frame(nx, w_y, ww, wh, yaxis=False)
ax.add_patch(Rectangle((nx + 0.18 * ww, w_y + 0.002), 0.64 * ww, wh * 0.55,
                       fc=C_INF_F, ec=C_INF, lw=0.8, zorder=5))
cx_w = nx + 0.18 * ww + 0.64 * ww * 0.56
ax.add_line(Line2D([cx_w, cx_w], [w_y, w_y + wh * 0.95], color=C_INF, lw=1.6, zorder=6))
T(nx, w_y - 0.005, "1 Da isolation window", size=6,
  color=C_INF, ha="left", va="top")
T(tx, top - 0.166, "4  window centroid", size=7, weight="bold", color=C_INF, va="top")
T(tx + 0.013, top - 0.179, "low-confidence fallback", size=6, color=SUBIN, va="top")

# =============================================================================
# STAGE 4 - Deduplication (full-width band, four sub-cards)
# =============================================================================
ix, iy, iw, ih = draw_box(ML, DE_Y, CW, DE_H, "Deduplication")
top = iy + ih
T(ML + CW / 2, top - 0.002,
  "four sequential steps  ·  relations are marked, nothing is deleted",
  size=7.5, color=SUBIN, style="italic", ha="center", va="top")
cpad = 0.008
cgap = 0.012
cx0 = ML + cpad
cardW = (CW - 2 * cpad - 3 * cgap) / 4
cardTop = top - 0.024
cardBot = iy + 0.004
cardH = cardTop - cardBot
cardX = [cx0 + i * (cardW + cgap) for i in range(4)]
# inside a card: main graphic on the left, co-elution evidence on the right (side by side)
G_X, G_W = 0.014, 0.108          # main graphic: x offset in the card, width
G_Y, G_H = 0.020, 0.048          # baseline above cardBot, height
M_X, M_W = 0.146, 0.060          # co-elution mini: x offset in the card, width
M_Y, M_H = 0.034, 0.034          # baseline above cardBot, height

def card(i, num, name):
    x = cardX[i]
    ax.add_patch(FancyBboxPatch((x, cardBot), cardW, cardH,
                                boxstyle="round,pad=0,rounding_size=0.006",
                                mutation_aspect=1 / ASPECT, fc="white", ec=RULE,
                                lw=0.8, zorder=5))
    T(x + 0.008, cardTop - 0.013, str(num), size=8, weight="bold", color=C_MS1,
      va="center")
    T(x + 0.019, cardTop - 0.013, name, size=7.5, weight="bold", color=INK, va="center")
    return x

def coelution_mini(x, ca, cb):
    """Two overlapping MS1 EIC peaks (shared RT apex) - the co-elution evidence
    common to isotope / adduct / ISF. x = left edge of the card."""
    gx, gy = x + M_X, cardBot + M_Y
    axes_frame(gx, gy, M_W, M_H, yaxis=False)
    eic(gx, gy, M_W, M_H, 0.5, 0.17, 0.95, ca, lw=0.9)
    eic(gx, gy, M_W, M_H, 0.5, 0.17, 0.62, cb, lw=0.9)
    T(gx + M_W / 2, gy - 0.005, "co-elute", size=6, color=SUBIN, ha="center", va="top")
    T(gx + M_W / 2, gy - 0.016, "EIC r ≥ 0.9", size=6, color=SUBIN, ha="center", va="top")

# 1 Isotope
x = card(0, 1, "Isotope")
gx, gy, gw, gh = x + G_X, cardBot + G_Y, G_W, G_H
axes_frame(gx, gy, gw, gh)
ax.add_line(Line2D([gx + 0.20 * gw] * 2, [gy, gy + 0.95 * gh], color=C_MS1, lw=1.5, zorder=6))
for fxp, fhp in [(0.5, 0.42), (0.8, 0.17)]:
    ax.add_line(Line2D([gx + fxp * gw] * 2, [gy, gy + fhp * gh], color=NEUTRAL, lw=1.2, zorder=6))
for lab, fxp, col in [("M", 0.20, C_MS1), ("M+1", 0.5, NEUTRAL), ("M+2", 0.8, NEUTRAL)]:
    T(gx + fxp * gw, gy - 0.005, lab, size=6, color=col, ha="center", va="top")
ax.annotate("", xy=(gx + 0.5 * gw, gy + gh * 1.05), xytext=(gx + 0.2 * gw, gy + gh * 1.05),
            arrowprops=dict(arrowstyle="<->", color=SUBIN, lw=0.6, shrinkA=0, shrinkB=0,
                            mutation_scale=5), zorder=6)
T(gx + 0.35 * gw, gy + gh * 1.10, "+1.003 (13C)", size=6, color=SUBIN, ha="center", va="bottom")
coelution_mini(x, C_MS1, NEUTRAL)

# 2 Adduct
x = card(1, 2, "Adduct")
gx, gy, gw, gh = x + G_X, cardBot + G_Y, G_W, G_H
axes_frame(gx, gy, gw, gh, yaxis=False)
xh, xna = gx + 0.24 * gw, gx + 0.80 * gw
ax.add_line(Line2D([xh, xh], [gy, gy + 0.90 * gh], color=C_MS1, lw=1.5, zorder=6))
ax.add_line(Line2D([xna, xna], [gy, gy + 0.60 * gh], color=NEUTRAL, lw=1.5, zorder=6))
ax.annotate("", xy=(xna, gy + gh * 1.05), xytext=(xh, gy + gh * 1.05),
            arrowprops=dict(arrowstyle="<->", color=SUBIN, lw=0.6, shrinkA=0, shrinkB=0,
                            mutation_scale=5), zorder=6)
T((xh + xna) / 2, gy + gh * 1.10, "Δm/z 21.98", size=6, color=SUBIN, ha="center", va="bottom")
T(xh, gy - 0.005, "[M+H]+", size=6, color=C_MS1, ha="center", va="top")
T(xna, gy - 0.005, "[M+Na]+", size=6, color=NEUTRAL, ha="center", va="top")
coelution_mini(x, C_MS1, NEUTRAL)

# 3 Spectral duplicate
x = card(2, 3, "Spectral duplicate")
gx, gy, gw, gh = x + G_X, cardBot + 0.010, G_W, 0.068
dpk = [(0.16, 0.9), (0.36, 0.5), (0.56, 0.72), (0.8, 0.34)]
mirror(gx, gy, gw, gh, dpk, [(a, b * 0.95) for a, b in dpk], C_MS1, NEUTRAL,
       ax_color=NEUTRAL, ax_lw=0.9, ax_z=7)
T(x + M_X + M_W / 2, gy + gh * 0.60, "MS2 cosine", size=6, color=SUBIN, ha="center", va="center")
T(x + M_X + M_W / 2, gy + gh * 0.40, "≥ 0.85", size=6, color=SUBIN, ha="center", va="center")

# 4 In-source fragmentation
x = card(3, 4, "In-source fragmentation")
gx, gy, gw, gh = x + G_X, cardBot + G_Y, G_W, 0.056
ymid = gy + gh / 2
ax.add_line(Line2D([gx, gx + gw], [ymid, ymid], color=NEUTRAL, lw=0.9, zorder=7))
ax.add_line(Line2D([gx, gx], [gy, gy + gh], color=NEUTRAL, lw=0.9, zorder=7))
shared = 0.68
for fxp, fhp in [(0.42, 0.5), (0.54, 0.66), (shared, 0.9), (0.84, 0.34), (0.95, 0.56)]:
    col = INK if fxp == shared else C_MS1
    ax.add_line(Line2D([gx + fxp * gw] * 2, [ymid, ymid + fhp * (gh / 2)], color=col, lw=1.1, zorder=6))
for fxp, fhp in [(0.42, 0.46), (0.54, 0.60), (shared, 0.86)]:
    col = INK if fxp == shared else NEUTRAL
    ax.add_line(Line2D([gx + fxp * gw] * 2, [ymid, ymid - fhp * (gh / 2)], color=col, lw=1.1, zorder=6))
ax.add_line(Line2D([gx + shared * gw] * 2, [gy, gy + gh], color=INK, lw=0.5,
                   ls=(0, (1.4, 1.4)), alpha=0.7, zorder=5))
T(gx + 0.004, ymid + 0.52 * (gh / 2), "parent", size=6, color=C_MS1, va="center")
T(gx + 0.004, ymid - 0.52 * (gh / 2), "child", size=6, color=NEUTRAL, va="center")
T(gx + shared * gw, gy - 0.005, "child precursor", size=6, color=INK,
  ha="center", va="top")
coelution_mini(x, C_MS1, NEUTRAL)

# =============================================================================
# STAGE 5 - Annotation (mirror plot + two-tier confidence)
# =============================================================================
ix, iy, iw, ih = draw_box(COL3[0], R3_Y, W3, R3_H, "Annotation")
top = iy + ih
T(ix, top - 0.004, "Spectral-library matching", size=8,
  weight="bold", color=INK, va="top")
mx, my, mw, mh = ix + 0.006, iy + 0.016, iw * 0.50, ih - 0.040
# sticks start right of the "query" / "reference" labels so that no stick crosses them
q = [(0.30, 0.5), (0.42, 0.95), (0.54, 0.42), (0.66, 0.8), (0.78, 0.35), (0.92, 0.6)]
r = [(0.30, 0.45), (0.42, 0.9), (0.54, 0.3), (0.66, 0.75), (0.78, 0.5), (0.92, 0.55)]
mirror(mx, my, mw, mh, q, r, C_MS1, C_MS2, matched=[0.30, 0.42, 0.66, 0.92])
T(mx + 0.004, my + mh - 0.004, "query", size=7, color=C_MS1, va="top")
T(mx + 0.004, my + 0.004, "reference", size=7, color=C_MS2_D, va="bottom")
T(mx + mw / 2, my - 0.004, "m/z", size=6, color=SUBIN, ha="center", va="top")
tx = mx + mw + 0.024
T(tx, top - 0.024, "composite score =", size=8, weight="bold", color=INK, va="top")
T(tx + 0.006, top - 0.042, "(√WDP + √SDP + √RDP)/3", size=7.5, color=INK, va="top")
T(tx + 0.028, top - 0.058, "+ Matched%", size=7.5, color=INK, va="top")
by = iy + 0.004
ax.add_patch(FancyBboxPatch((tx, by), iw - (tx - ix) - 0.006, 0.052,
                            boxstyle="round,pad=0,rounding_size=0.005",
                            mutation_aspect=1 / ASPECT, fc=C_MS1_F, ec=C_MS1,
                            lw=0.8, zorder=5))
T(tx + 0.008, by + 0.040, "high-scoring:", size=7.5, weight="bold",
  color=C_MS1, va="center")
T(tx + 0.008, by + 0.026, "Total score ≥ 1.4 and", size=7, color=INK, va="center")
T(tx + 0.008, by + 0.012, "≥ 3 matched peaks", size=7, color=INK, va="center")

# =============================================================================
# STAGE 6 - Alignment & export (v1.4 identity rule)
# =============================================================================
ix, iy, iw, ih = draw_box(COL3[1], R3_Y, W3, R3_H, "Alignment & export")
top = iy + ih
T(ix, top - 0.004, "Cross-sample alignment", size=8, weight="bold",
  color=INK, va="top")
lat_x = ix + 0.056
lat_w = iw * 0.60
cols = [0.10, 0.30, 0.52, 0.74, 0.92]
present = [[1, 1, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 1, 1]]
jit = [[0.0, 0.01, -0.01, 0, 0.008], [0.006, -0.008, 0.004, -0.006, 0.0],
       [-0.006, 0, 0.008, 0.006, -0.004]]
ry = [top - 0.032, top - 0.049, top - 0.066]
for ci, c in enumerate(cols):
    X = lat_x + c * lat_w
    ax.add_line(Line2D([X, X], [ry[2] - 0.006, ry[0] + 0.006], color=RULE,
                       lw=0.5, ls=(0, (1.4, 1.4)), zorder=4))
for i, y in enumerate(ry):
    ax.add_line(Line2D([lat_x, lat_x + lat_w], [y, y], color=RULE, lw=0.55, zorder=4))
    T(lat_x - 0.008, y, f"rep {i + 1}", size=7, color=SUBIN, ha="right", va="center")
    for ci, c in enumerate(cols):
        X = lat_x + (c + jit[i][ci]) * lat_w
        if present[i][ci]:
            ax.scatter([X], [y], s=15, c=C_MS1, edgecolors="white",
                       linewidths=0.4, zorder=7)
        else:                                          # gap-fill target
            ax.scatter([X], [y], s=16, facecolors="none", edgecolors=INK,
                       linewidths=0.7, zorder=7)
T(lat_x + lat_w + 0.010, ry[1], "aligned\nfeature\ntable", size=7, color=C_MS1,
  ha="left", va="center")
T(ix, ry[2] - 0.012, "identity decided pairwise from precursor and spectrum evidence",
  size=6, color=SUBIN, style="italic", va="top")
ey = iy + 0.016
fw2, fh2, fgap = 0.046, 0.028, 0.030
files = [("features.csv", C_MS1_F, C_MS1), (".mgf", C_MS2_F, C_MS2_D), (".msp", C_MS2_F, C_MS2_D)]
fx0 = ix + 0.020
for k, (nm, ff, fs) in enumerate(files):
    file_icon(fx0 + k * (fw2 + fgap), ey, fw2, fh2, nm, ff, fs)
T(fx0 + 3 * (fw2 + fgap) + 0.004, ey + fh2 / 2, "+ report", size=7, color=SUBIN,
  va="center")

# =============================================================================
# inter-stage flow arrows
# =============================================================================
yc1 = R1_Y + R1_H / 2
arrow(COL[0] + W1, yc1, COL[1], yc1)                   # 1 -> 2
arrow(COL[1] + W1, yc1, COL[2], yc1)                   # 2 -> 3
busA = (R1_Y + DE_Y + DE_H) / 2
elbow([(CXc[2], R1_Y), (CXc[2], busA), (0.5, busA), (0.5, DE_Y + DE_H)])
busB = (DE_Y + R3_Y + R3_H) / 2
elbow([(0.5, DE_Y), (0.5, busB), (CX3[0], busB), (CX3[0], R3_Y + R3_H)])
yc3 = R3_Y + R3_H / 2
arrow(COL3[0] + W3, yc3, COL3[1], yc3)

# =============================================================================
# save
# =============================================================================
LEGEND = """Fig. 2 legend sentences (revision 261019; paste into the manuscript)

Precursor m/z assignment takes the first source that succeeds, in order: a strict co-eluting MS1 peak (exclusive assignment), a relaxed MS1 peak match, a neutral-loss consensus vote across product ions, and, when both fail, the centroid of the 1 Da isolation window as a low-confidence fallback. MS1-driven features carry the m/z of their MS1 peak directly, and the source and confidence tier of every inferred precursor are exported with the feature.

Cross-sample alignment decides identity per pair from the evidence each side carries (an exact precursor whose spectrum must not contradict, a spectrum only, or a cross-route match with window containment and a chemical anchor), clusters by complete linkage, and marks redundancy without deleting any feature.

Colours: blue, MS1 evidence (survey scan, MS1-driven detection, MS1-detected features); orange, MS2 product-ion evidence (stepped HR-MRM windows, MS2-driven detection, library reference spectra); purple, MS2-only features and inferred precursors.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(HERE.parent / "figures" / "fig2"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from matplotlib.transforms import Bbox
    crop = Bbox.from_extents(0, CROP_Y * FH, FW, FH)      # inches; drops the empty band below row 3
    for ext in ("svg", "pdf", "png"):
        fig.savefig(out / f"Fig 2.{ext}", dpi=300, facecolor="white", bbox_inches=crop)
    (out / "legend_fig2.txt").write_text(LEGEND, encoding="utf-8")
    print("wrote Fig 2.svg / .pdf / .png and legend_fig2.txt to", out)


if __name__ == "__main__":
    main()
