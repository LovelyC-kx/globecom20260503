"""OrbitVLIF fancy architecture diagram (replaces the flat fig1.tex).

Generates `fig1.pdf` — a 3D-style block-diagram of the OrbitVLIF
single-satellite architecture + the 5x10 Walker-Star federated topology.

Style notes
-----------
* IEEE 2-column full-width target (renders well at \\textwidth ~ 7.0 in).
* Wong-2011 colour-blind safe palette, identical to fig1.tex / fig2_modules.tex.
* Feature maps drawn as parallelogram-perspective 3-D slabs (front + top
  + right faces) so resolution / channel count are visually obvious.
* Module blocks drawn as rounded rectangles with a soft drop shadow so
  the figure has 立体感 even when printed in greyscale.

Usage
-----
    python -m cloud_removal_v2.plot_arch_diagram --out_dir ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MPath


# ---------------------------------------------------------------------------
# Wong-2011 colour-blind-safe palette  (identical hex codes to fig1.tex)
# ---------------------------------------------------------------------------
C_BLUE   = "#0173B2"   # encoder / OrbitVLIF
C_ORANGE = "#DE8F05"   # decoder / ANN
C_GREEN  = "#029E73"   # SHAM / accent 1
C_RED    = "#D55E00"   # DSP bottleneck / accent 2
C_PURPLE = "#CC78BC"   # 5QS / AGFM / skip
C_YELLOW = "#ECE133"   # LIF (matches fig2_modules.tex)
C_GRAY   = "#4D4D4D"   # outline / text
C_DGRAY  = "#2D2D2D"   # darker outline
C_LIGHT  = "#F4F4F4"   # subtle background
C_WHITE  = "#FFFFFF"


def _hex_to_rgb(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def lighten(color: str, amount: float = 0.55) -> Tuple[float, float, float]:
    """Mix `color` with white.  amount=0 → original, amount=1 → white."""
    r, g, b = _hex_to_rgb(color)
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)


def darken(color: str, amount: float = 0.30) -> Tuple[float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return (r * (1 - amount), g * (1 - amount), b * (1 - amount))


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def shadow_rect(ax, x: float, y: float, w: float, h: float,
                radius: float = 0.05, dx: float = 0.04, dy: float = -0.04,
                zorder: int = 1) -> None:
    """Soft drop shadow under a rounded rectangle."""
    s = FancyBboxPatch(
        (x + dx, y + dy), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor="#9A9A9A", edgecolor="none", alpha=0.32, zorder=zorder,
    )
    ax.add_patch(s)


def fancy_block(ax, x: float, y: float, w: float, h: float,
                title: str, subtitle: Optional[str] = None,
                color: str = C_BLUE, *,
                radius: float = 0.06, fontsize: float = 7.0,
                subfontsize: float = 5.2, zorder: int = 3,
                bold: bool = True, shadow: bool = True) -> None:
    """Rounded module block with light fill + colored outline + drop shadow."""
    if shadow:
        shadow_rect(ax, x, y, w, h, radius=radius, zorder=zorder - 1)

    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=lighten(color, 0.62), edgecolor=color, linewidth=1.0,
        zorder=zorder,
    )
    ax.add_patch(rect)

    weight = "bold" if bold else "normal"
    cy_title = y + h / 2 + (0.07 if subtitle else 0)
    ax.text(x + w / 2, cy_title, title,
            ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            color=darken(color, 0.30), zorder=zorder + 1)
    if subtitle:
        ax.text(x + w / 2, y + h / 2 - 0.10, subtitle,
                ha="center", va="center",
                fontsize=subfontsize, color=darken(color, 0.20),
                alpha=0.92, zorder=zorder + 1)


def tensor_3d(ax, cx: float, cy: float, fw: float, fh: float, depth: float,
              color: str = C_BLUE, *,
              label: Optional[str] = None, dim_label: Optional[str] = None,
              zorder: int = 2) -> None:
    """Parallelogram-perspective 3-D feature-map slab.

    (cx, cy) is the centre of the FRONT face.  fw,fh = front-face size,
    `depth` = isometric depth (visually proportional to channel count).
    """
    dx = depth * 0.65
    dy = depth * 0.40

    x0, y0 = cx - fw / 2, cy - fh / 2
    x1, y1 = cx + fw / 2, cy + fh / 2

    fc_front = lighten(color, 0.45)
    fc_top   = lighten(color, 0.20)
    fc_right = lighten(color, 0.65)
    ec       = darken(color, 0.20)

    # Right face (drawn first; sits behind)
    right = Polygon(
        [[x1, y0], [x1 + dx, y0 + dy], [x1 + dx, y1 + dy], [x1, y1]],
        closed=True, facecolor=fc_right, edgecolor=ec, linewidth=0.5,
        zorder=zorder,
    )
    ax.add_patch(right)

    # Top face
    top = Polygon(
        [[x0, y1], [x1, y1], [x1 + dx, y1 + dy], [x0 + dx, y1 + dy]],
        closed=True, facecolor=fc_top, edgecolor=ec, linewidth=0.5,
        zorder=zorder,
    )
    ax.add_patch(top)

    # Front face (on top of side faces)
    front = FancyBboxPatch(
        (x0, y0), fw, fh,
        boxstyle="round,pad=0,rounding_size=0.018",
        facecolor=fc_front, edgecolor=ec, linewidth=0.6, zorder=zorder + 1,
    )
    ax.add_patch(front)

    if label:
        ax.text(cx, cy + (0.04 if dim_label else 0.0), label,
                ha="center", va="center",
                fontsize=4.6, fontweight="bold",
                color=darken(color, 0.40), zorder=zorder + 2)
    if dim_label:
        ax.text(cx, cy - 0.10, dim_label,
                ha="center", va="center",
                fontsize=4.0, color=darken(color, 0.30),
                alpha=0.92, zorder=zorder + 2)


def arrow_h(ax, x1: float, x2: float, y: float, *,
            label: Optional[str] = None, color: str = C_GRAY,
            lw: float = 0.9, zorder: int = 5) -> None:
    """Horizontal arrow x1 → x2 at height y, with optional label above."""
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=7),
                zorder=zorder)
    if label:
        ax.text((x1 + x2) / 2, y + 0.07, label,
                ha="center", va="bottom",
                fontsize=4.5, color=color, zorder=zorder + 1)


def arrow_v(ax, x: float, y1: float, y2: float, *,
            label: Optional[str] = None, color: str = C_GRAY,
            lw: float = 0.9, label_side: str = "right",
            zorder: int = 5) -> None:
    """Vertical arrow y1 → y2 at x.  label_side ∈ {'left','right'}."""
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=7),
                zorder=zorder)
    if label:
        ox = 0.07 if label_side == "right" else -0.07
        ha = "left" if label_side == "right" else "right"
        ax.text(x + ox, (y1 + y2) / 2, label,
                ha=ha, va="center",
                fontsize=4.5, color=color, zorder=zorder + 1)


def arrow_L(ax, x1: float, y1: float, x2: float, y2: float, *,
            color: str = C_GRAY, lw: float = 0.9, zorder: int = 4,
            via: str = "vh") -> None:
    """L-shaped (Manhattan) arrow.

    via='vh': go vertical first, then horizontal (good for top-to-side).
    via='hv': go horizontal first, then vertical.
    """
    if via == "vh":
        verts = [(x1, y1), (x1, y2), (x2, y2)]
    else:
        verts = [(x1, y1), (x2, y1), (x2, y2)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.LINETO]
    pp = PathPatch(MPath(verts, codes), fc="none", ec=color, lw=lw, zorder=zorder)
    ax.add_patch(pp)
    # Arrow head at the very end
    head_dx = 0 if (via == "vh" or x1 == x2) else (0.001 if x2 > x1 else -0.001)
    head_dy = 0
    ax.annotate("", xy=(x2, y2), xytext=(x2 - head_dx, y2 - head_dy + 0.08),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=6),
                zorder=zorder + 1)


def skip_with_gate(ax, x: float, y_top: float, y_bot: float,
                   gate_label: str = "AGFM", color: str = C_PURPLE,
                   lw: float = 0.9, zorder: int = 4) -> None:
    """Vertical dashed skip connection with a diamond gate at the midpoint."""
    # Dashed line top→bottom (we'll re-draw a small gap around the gate)
    ymid = (y_top + y_bot) / 2
    ax.plot([x, x], [y_top, ymid + 0.13], color=color, lw=lw,
            ls=(0, (3, 1.5)), zorder=zorder, alpha=0.85)
    ax.plot([x, x], [ymid - 0.13, y_bot], color=color, lw=lw,
            ls=(0, (3, 1.5)), zorder=zorder, alpha=0.85)

    # Diamond gate
    diamond = Polygon(
        [[x, ymid + 0.11], [x + 0.13, ymid], [x, ymid - 0.11], [x - 0.13, ymid]],
        closed=True, facecolor=lighten(color, 0.45), edgecolor=color,
        linewidth=0.8, zorder=zorder + 1,
    )
    ax.add_patch(diamond)
    ax.text(x, ymid, gate_label, ha="center", va="center",
            fontsize=4.4, fontweight="bold",
            color=darken(color, 0.35), zorder=zorder + 2)

    # Arrow head at bottom
    ax.annotate("", xy=(x, y_bot + 0.005), xytext=(x, y_bot + 0.10),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=6),
                zorder=zorder + 1)


# ---------------------------------------------------------------------------
# Architecture-pipeline coordinates  (centre of each module along x)
# ---------------------------------------------------------------------------
# These are chosen so the encoder and decoder share x-coordinates per
# resolution level, making skip connections strictly vertical.
X_IN, X_PE, X_E1, X_E2, X_E3, X_DSP = 0.50, 1.30, 2.40, 3.50, 4.60, 5.70
Y_ENC, Y_DEC = 4.55, 2.75

# Module block dimensions (w, h) at each stage
DIM_IN  = (0.55, 0.78)
DIM_PE  = (0.55, 0.62)
DIM_E1  = (0.80, 1.05)
DIM_E2  = (0.74, 0.92)
DIM_E3  = (0.62, 0.78)
DIM_DSP = (0.78, 0.78)

# Feature-map slab dimensions (front-face w, h, depth) between successive stages
FM_IN_PE  = (0.16, 0.55, 0.04)   # 3 ch
FM_PE_E1  = (0.16, 0.42, 0.10)   # 24 ch
FM_E1_E2  = (0.13, 0.34, 0.13)   # 48 ch (after E1 down-2)
FM_E2_E3  = (0.10, 0.26, 0.16)   # 96 ch (after E2 down-2)
FM_E3_DSP = (0.08, 0.20, 0.18)   # 96 ch (after E3 down-2)
FM_BOT    = (0.08, 0.20, 0.20)   # bottleneck output
# Decoder mirrors encoder (same shapes, orange tint)
FM_D3_D2 = FM_E2_E3
FM_D2_D1 = FM_E1_E2
FM_D1_HD = FM_PE_E1
FM_HD_OUT = FM_IN_PE


def _draw_image_icon(ax, x_c: float, y_c: float, w: float, h: float,
                     color: str = C_GRAY, label: str = "Cloudy") -> None:
    """Tiny 'image' icon (rounded rectangle with diagonal gradient hint)."""
    shadow_rect(ax, x_c - w/2, y_c - h/2, w, h, radius=0.04, zorder=2)
    rect = FancyBboxPatch(
        (x_c - w/2, y_c - h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.04",
        facecolor=lighten(color, 0.85), edgecolor=color, linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(rect)
    # Hint of "image content" — two diagonal stripes
    for off in (-0.18, 0.0, 0.18):
        ax.plot([x_c - w/2 + 0.06, x_c + w/2 - 0.06],
                [y_c + off + 0.02, y_c + off - 0.10],
                color=color, lw=0.4, alpha=0.35, zorder=4)
    ax.text(x_c, y_c - h/2 - 0.16, label,
            ha="center", va="top", fontsize=5.2,
            color=darken(color, 0.10), zorder=5)


def _panel_architecture(ax) -> None:
    """Panel (a): OrbitVLIF single-satellite U-Net with 3-D feature maps."""
    # ---- Panel title (left-aligned over the encoder row) -------------------
    ax.text(0.25, 5.30,
            r"(a) OrbitVLIF — single-satellite cloud-removal backbone",
            ha="left", va="center",
            fontsize=8.5, fontweight="bold", color="black")

    # ---- Soft background panel for the architecture region -----------------
    bg = FancyBboxPatch(
        (0.18, 1.95), 6.10, 3.18,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0,
    )
    ax.add_patch(bg)

    # ====================================================================
    # ENCODER row (left → right):  Input → PatchEmbed → E1 → E2 → E3 → DSP
    # ====================================================================

    # Input image icon
    _draw_image_icon(ax, X_IN, Y_ENC, *DIM_IN,
                     color=C_GRAY, label=r"$\mathbf{x}$  (cloudy)")

    # Slab: input
    tensor_3d(ax,
              cx=(X_IN + X_PE) / 2 + 0.02, cy=Y_ENC,
              fw=FM_IN_PE[0], fh=FM_IN_PE[1], depth=FM_IN_PE[2],
              color=C_GRAY,
              dim_label=r"$3{\times}H{\times}W$")

    # Patch-Embed
    fancy_block(ax,
                X_PE - DIM_PE[0]/2, Y_ENC - DIM_PE[1]/2,
                DIM_PE[0], DIM_PE[1],
                title="Patch", subtitle="Embed", color=C_GRAY,
                fontsize=6.5, subfontsize=5.2)

    # Slab: after Patch-Embed (×4 downsample, 24 ch)
    tensor_3d(ax,
              cx=(X_PE + X_E1) / 2, cy=Y_ENC,
              fw=FM_PE_E1[0], fh=FM_PE_E1[1], depth=FM_PE_E1[2],
              color=C_BLUE,
              dim_label=r"$24{\times}\tfrac{H}{4}{\times}\tfrac{W}{4}$")

    # Encoder L1
    fancy_block(ax,
                X_E1 - DIM_E1[0]/2, Y_ENC - DIM_E1[1]/2,
                DIM_E1[0], DIM_E1[1],
                title="Encoder-1",
                subtitle="MFRB×2 + SHAM\n$C{=}24$",
                color=C_BLUE, fontsize=6.6, subfontsize=5.0)

    # Slab: after E1 (resolution unchanged ; before downsample)
    tensor_3d(ax,
              cx=(X_E1 + X_E2) / 2, cy=Y_ENC,
              fw=FM_E1_E2[0], fh=FM_E1_E2[1], depth=FM_E1_E2[2],
              color=C_BLUE,
              dim_label=r"$48{\times}\tfrac{H}{8}{\times}\tfrac{W}{8}$")
    # Down-sample annotation
    ax.text((X_E1 + X_E2) / 2, Y_ENC + 0.55, r"$\downarrow\!2$",
            ha="center", va="center", fontsize=5.2,
            color=darken(C_BLUE, 0.20))

    # Encoder L2
    fancy_block(ax,
                X_E2 - DIM_E2[0]/2, Y_ENC - DIM_E2[1]/2,
                DIM_E2[0], DIM_E2[1],
                title="Encoder-2",
                subtitle="MFRB×2 + SHAM\n$C{=}48$",
                color=C_BLUE, fontsize=6.4, subfontsize=4.9)

    # Slab: after E2
    tensor_3d(ax,
              cx=(X_E2 + X_E3) / 2, cy=Y_ENC,
              fw=FM_E2_E3[0], fh=FM_E2_E3[1], depth=FM_E2_E3[2],
              color=C_BLUE,
              dim_label=r"$96{\times}\tfrac{H}{16}{\times}\tfrac{W}{16}$")
    ax.text((X_E2 + X_E3) / 2, Y_ENC + 0.50, r"$\downarrow\!2$",
            ha="center", va="center", fontsize=5.2,
            color=darken(C_BLUE, 0.20))

    # Encoder L3 (deepest, MFRB×4)
    fancy_block(ax,
                X_E3 - DIM_E3[0]/2, Y_ENC - DIM_E3[1]/2,
                DIM_E3[0], DIM_E3[1],
                title="Encoder-3",
                subtitle="MFRB×4\n$C{=}96$",
                color=C_BLUE, fontsize=6.3, subfontsize=4.9)

    # Slab: E3 → DSP (same resolution)
    tensor_3d(ax,
              cx=(X_E3 + X_DSP) / 2, cy=Y_ENC,
              fw=FM_E3_DSP[0], fh=FM_E3_DSP[1], depth=FM_E3_DSP[2],
              color=C_RED,
              dim_label="")

    # DSP bottleneck (red — distinctive)
    fancy_block(ax,
                X_DSP - DIM_DSP[0]/2, Y_ENC - DIM_DSP[1]/2,
                DIM_DSP[0], DIM_DSP[1],
                title="DSP",
                subtitle="5QS + CTM\n$C{=}96$",
                color=C_RED, fontsize=7.0, subfontsize=5.0)

    # Soft halo behind DSP to draw the eye
    halo = FancyBboxPatch(
        (X_DSP - DIM_DSP[0]/2 - 0.07, Y_ENC - DIM_DSP[1]/2 - 0.07),
        DIM_DSP[0] + 0.14, DIM_DSP[1] + 0.14,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="none", edgecolor=lighten(C_RED, 0.55), linewidth=0.6,
        linestyle=(0, (2, 1.2)), zorder=2,
    )
    ax.add_patch(halo)

    # ====================================================================
    # DECODER row (right → left): D3 ← D2 ← D1 ← Head ← Output
    # ====================================================================
    fancy_block(ax,
                X_E3 - DIM_E3[0]/2, Y_DEC - DIM_E3[1]/2,
                DIM_E3[0], DIM_E3[1],
                title="Decoder-3",
                subtitle="MFRB×2\n$C{=}96$",
                color=C_ORANGE, fontsize=6.4, subfontsize=4.9)

    tensor_3d(ax,
              cx=(X_E3 + X_E2) / 2, cy=Y_DEC,
              fw=FM_D3_D2[0], fh=FM_D3_D2[1], depth=FM_D3_D2[2],
              color=C_ORANGE,
              dim_label="")
    ax.text((X_E3 + X_E2) / 2, Y_DEC - 0.50, r"$\uparrow\!2$",
            ha="center", va="center", fontsize=5.2,
            color=darken(C_ORANGE, 0.25))

    fancy_block(ax,
                X_E2 - DIM_E2[0]/2, Y_DEC - DIM_E2[1]/2,
                DIM_E2[0], DIM_E2[1],
                title="Decoder-2",
                subtitle="MFRB×2\n$C{=}48$",
                color=C_ORANGE, fontsize=6.4, subfontsize=4.9)

    tensor_3d(ax,
              cx=(X_E2 + X_E1) / 2, cy=Y_DEC,
              fw=FM_D2_D1[0], fh=FM_D2_D1[1], depth=FM_D2_D1[2],
              color=C_ORANGE,
              dim_label="")
    ax.text((X_E2 + X_E1) / 2, Y_DEC - 0.50, r"$\uparrow\!2$",
            ha="center", va="center", fontsize=5.2,
            color=darken(C_ORANGE, 0.25))

    fancy_block(ax,
                X_E1 - DIM_E1[0]/2, Y_DEC - DIM_E1[1]/2,
                DIM_E1[0], DIM_E1[1],
                title="Decoder-1",
                subtitle="MFRB×2\n$C{=}24$",
                color=C_ORANGE, fontsize=6.6, subfontsize=5.0)

    tensor_3d(ax,
              cx=(X_E1 + X_PE) / 2, cy=Y_DEC,
              fw=FM_D1_HD[0], fh=FM_D1_HD[1], depth=FM_D1_HD[2],
              color=C_ORANGE,
              dim_label="")

    fancy_block(ax,
                X_PE - DIM_PE[0]/2, Y_DEC - DIM_PE[1]/2,
                DIM_PE[0], DIM_PE[1],
                title="Output", subtitle="Head", color=C_GRAY,
                fontsize=6.5, subfontsize=5.2)

    tensor_3d(ax,
              cx=(X_PE + X_IN) / 2 - 0.02, cy=Y_DEC,
              fw=FM_HD_OUT[0], fh=FM_HD_OUT[1], depth=FM_HD_OUT[2],
              color=C_GRAY,
              dim_label="")

    _draw_image_icon(ax, X_IN, Y_DEC, *DIM_IN,
                     color=C_GRAY, label=r"$\hat{\mathbf{y}}$  (restored)")

    # ====================================================================
    # Connecting arrows  (encoder L→R, decoder R→L)
    # ====================================================================
    # Encoder right-pointing arrows between block edges (pass under slabs)
    for (xa, wa), (xb, wb) in [
        ((X_IN,  DIM_IN[0]),  (X_PE,  DIM_PE[0])),
        ((X_PE,  DIM_PE[0]),  (X_E1,  DIM_E1[0])),
        ((X_E1,  DIM_E1[0]),  (X_E2,  DIM_E2[0])),
        ((X_E2,  DIM_E2[0]),  (X_E3,  DIM_E3[0])),
        ((X_E3,  DIM_E3[0]),  (X_DSP, DIM_DSP[0])),
    ]:
        arrow_h(ax, xa + wa/2 + 0.01, xb - wb/2 - 0.01, Y_ENC,
                color=C_GRAY, lw=0.7, zorder=2)

    # Decoder left-pointing arrows
    for (xa, wa), (xb, wb) in [
        ((X_E3, DIM_E3[0]), (X_E2, DIM_E2[0])),
        ((X_E2, DIM_E2[0]), (X_E1, DIM_E1[0])),
        ((X_E1, DIM_E1[0]), (X_PE, DIM_PE[0])),
        ((X_PE, DIM_PE[0]), (X_IN, DIM_IN[0])),
    ]:
        # Arrow from xa-w/2 (left edge of right block) to xb+w/2 (right edge of left block)
        ax.annotate("",
                    xy=(xb + wb/2 + 0.01, Y_DEC),
                    xytext=(xa - wa/2 - 0.01, Y_DEC),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                    lw=0.7, mutation_scale=7),
                    zorder=2)

    # DSP → Decoder-3  (L-shaped, on the right side)
    arrow_L(ax,
            x1=X_DSP, y1=Y_ENC - DIM_DSP[1]/2 - 0.02,
            x2=X_E3 + DIM_E3[0]/2 + 0.02, y2=Y_DEC,
            color=C_RED, lw=1.0, via="vh")

    # ====================================================================
    # Skip connections (encoder → decoder via AGFM)
    # ====================================================================
    for x_skip in (X_E1, X_E2, X_E3):
        skip_with_gate(ax,
                       x=x_skip,
                       y_top=Y_ENC - {X_E1: DIM_E1[1], X_E2: DIM_E2[1],
                                      X_E3: DIM_E3[1]}[x_skip] / 2 - 0.02,
                       y_bot=Y_DEC + {X_E1: DIM_E1[1], X_E2: DIM_E2[1],
                                      X_E3: DIM_E3[1]}[x_skip] / 2 + 0.02,
                       gate_label="AGFM", color=C_PURPLE, lw=0.8)

    # Global residual (curve from input to output, dashed)
    ax.annotate("",
                xy=(X_IN - 0.04, Y_DEC + DIM_IN[1]/2 - 0.05),
                xytext=(X_IN - 0.04, Y_ENC - DIM_IN[1]/2 + 0.05),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                lw=0.6, ls=(0, (3, 1.5)), mutation_scale=6),
                zorder=2)
    ax.text(X_IN - 0.18, (Y_ENC + Y_DEC) / 2,
            "global\nresidual", ha="right", va="center",
            fontsize=4.2, color=C_GRAY, zorder=2)


# ---------------------------------------------------------------------------
# Panel (b): Federated topology  (5 orbital planes × 10 sats, isometric)
# ---------------------------------------------------------------------------

def _panel_constellation(ax) -> None:
    """Right-side panel: 5×10 Walker-Star with intra-/inter-plane edges."""
    # Title
    ax.text(7.45, 5.30, r"(b) 5×10 Walker-Star federation",
            ha="center", va="center",
            fontsize=8.0, fontweight="bold", color="black")

    # Background panel
    bg = FancyBboxPatch(
        (6.40, 1.95), 1.95, 3.18,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0,
    )
    ax.add_patch(bg)

    # ---- 5 planes drawn as tilted ellipses, stacked vertically -------------
    n_planes = 5
    n_sat    = 10
    ring_rx, ring_ry = 0.65, 0.18    # ellipse semi-axes
    cx_const = 7.40                  # x centre of all rings
    plane_y0, plane_y1 = 4.78, 2.30  # top → bottom y centres
    plane_ys = [plane_y0 + (plane_y1 - plane_y0) * i / (n_planes - 1)
                for i in range(n_planes)]

    sat_positions = []   # [[(x, y) per sat] per plane]
    plane_colors  = [C_BLUE, C_GREEN, C_PURPLE, C_ORANGE, C_RED]

    for p, (cy, pc) in enumerate(zip(plane_ys, plane_colors)):
        # Ring (tilted ellipse, fill very light, dashed for "behind" arc)
        ring = mpatches.Ellipse(
            (cx_const, cy), 2 * ring_rx, 2 * ring_ry,
            facecolor=lighten(pc, 0.85), edgecolor=pc, linewidth=0.7,
            zorder=2,
        )
        ax.add_patch(ring)

        # Plane label (left of ring)
        ax.text(cx_const - ring_rx - 0.08, cy,
                rf"$\mathcal{{P}}_{{{p}}}$",
                ha="right", va="center", fontsize=5.2,
                color=darken(pc, 0.20), fontweight="bold")

        # 10 satellite dots on the ellipse, evenly spaced in angle
        import math
        sats_this_plane = []
        for i in range(n_sat):
            ang = 2 * math.pi * i / n_sat
            sx = cx_const + ring_rx * math.cos(ang)
            sy = cy + ring_ry * math.sin(ang)
            sats_this_plane.append((sx, sy))
            # Highlight one "plane head" per ring (sat 0)
            if i == 0:
                circ = mpatches.Circle((sx, sy), 0.05,
                                       facecolor=lighten(pc, 0.55),
                                       edgecolor=pc, linewidth=0.8, zorder=4)
            else:
                circ = mpatches.Circle((sx, sy), 0.035,
                                       facecolor="white",
                                       edgecolor=darken(pc, 0.10),
                                       linewidth=0.5, zorder=4)
            ax.add_patch(circ)
        sat_positions.append(sats_this_plane)

        # Intra-plane edges: ring-cycle connecting adjacent sats
        for i in range(n_sat):
            j = (i + 1) % n_sat
            x1, y1 = sats_this_plane[i]
            x2, y2 = sats_this_plane[j]
            ax.plot([x1, x2], [y1, y2], color=lighten(pc, 0.30),
                    lw=0.4, alpha=0.55, zorder=3)

    # ---- Inter-plane edges (Gossip chain): plane-head to plane-head ----------
    for p in range(n_planes - 1):
        x1, y1 = sat_positions[p][0]      # head of plane p
        x2, y2 = sat_positions[p + 1][0]  # head of plane p+1
        # Slight bezier-ish bend by going through midpoint shifted right
        mid_x, mid_y = (x1 + x2) / 2 + 0.18, (y1 + y2) / 2
        ax.plot([x1, mid_x, x2], [y1, mid_y, y2],
                color=C_RED, lw=0.7, alpha=0.85, zorder=5)
        # Tiny arrow at the destination
        ax.annotate("", xy=(x2 + 0.005, y2), xytext=(mid_x, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color=C_RED,
                                    lw=0.5, mutation_scale=4),
                    zorder=5)

    # Optional inter-plane shortcut (P0 ↔ P4) to suggest non-chain Gossip
    x0, y0 = sat_positions[0][0]
    x4, y4 = sat_positions[-1][0]
    ax.plot([x0 + 0.35, x4 + 0.35], [y0, y4],
            color=lighten(C_RED, 0.30), lw=0.5, ls=(0, (2, 1.5)),
            alpha=0.7, zorder=4)

    # ---- Mini legend at the bottom of the panel ------------------------------
    leg_y = 2.10
    # Plane head dot
    ax.add_patch(mpatches.Circle((6.55, leg_y), 0.045,
                                 facecolor=lighten(C_BLUE, 0.55),
                                 edgecolor=C_BLUE, linewidth=0.7, zorder=3))
    ax.text(6.62, leg_y, "head", ha="left", va="center",
            fontsize=4.5, color=C_GRAY)
    # Worker sat dot
    ax.add_patch(mpatches.Circle((6.92, leg_y), 0.030,
                                 facecolor="white", edgecolor=C_GRAY,
                                 linewidth=0.5, zorder=3))
    ax.text(6.97, leg_y, "sat", ha="left", va="center",
            fontsize=4.5, color=C_GRAY)
    # Intra ring edge
    ax.plot([7.18, 7.32], [leg_y, leg_y], color=lighten(C_BLUE, 0.30),
            lw=0.7, alpha=0.85)
    ax.text(7.36, leg_y, "intra", ha="left", va="center",
            fontsize=4.5, color=C_GRAY)
    # Inter Gossip edge
    ax.plot([7.65, 7.79], [leg_y, leg_y], color=C_RED, lw=0.7)
    ax.text(7.83, leg_y, "Gossip", ha="left", va="center",
            fontsize=4.5, color=C_GRAY)

    # FedBN+Dirichlet caption under the legend
    ax.text(7.40, 2.05 - 0.20,
            r"FedBN + Dir$(\alpha{=}0.1)$",
            ha="center", va="center",
            fontsize=4.8, fontstyle="italic",
            color=darken(C_GRAY, 0.10))


# ---------------------------------------------------------------------------
# Panel (c): Module zoom-ins  (MFRB / 5QS / SHAM)
# ---------------------------------------------------------------------------

def _panel_module_mfrb(ax, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (c1): MFRB dual-frequency residual block."""
    # Background
    bg = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0,
    )
    ax.add_patch(bg)
    # Title
    ax.text(x0 + 0.10, y0 + h - 0.13, "(c) MFRB",
            ha="left", va="center",
            fontsize=7.0, fontweight="bold", color="black")

    # Sub-block coordinates relative to panel
    cx_in   = x0 + 0.30
    cx_lif  = x0 + 0.85
    cx_conv = x0 + 1.40
    cx_bn   = x0 + 1.90
    cx_gate = x0 + 2.30
    cx_out  = x0 + 2.65

    y_top = y0 + h - 0.55
    y_bot = y0 + 0.40
    y_mid = (y_top + y_bot) / 2

    # Input
    fancy_block(ax, cx_in - 0.18, y_mid - 0.16, 0.36, 0.32,
                title=r"$\mathbf{X}^{(t)}$", color=C_GRAY,
                fontsize=6.0, shadow=False)

    # Group 1: LIF1 → Conv1 → BN  (top path, temporal)
    fancy_block(ax, cx_lif - 0.20, y_top - 0.13, 0.40, 0.26,
                title=r"LIF$_1$", color=C_YELLOW, fontsize=5.5,
                radius=0.04)
    fancy_block(ax, cx_conv - 0.22, y_top - 0.13, 0.44, 0.26,
                title=r"Conv$_1$", color=C_BLUE, fontsize=5.5,
                radius=0.04)
    fancy_block(ax, cx_bn - 0.18, y_top - 0.13, 0.36, 0.26,
                title=r"BN", color=C_GREEN, fontsize=5.5,
                radius=0.04)

    # Group 2: PS-LIF2 → Conv2 → BN  (bottom path, spatial / pixel-shuffle)
    fancy_block(ax, cx_lif - 0.22, y_bot - 0.13, 0.44, 0.26,
                title=r"PS-LIF$_2$", color=C_YELLOW, fontsize=5.3,
                radius=0.04)
    fancy_block(ax, cx_conv - 0.22, y_bot - 0.13, 0.44, 0.26,
                title=r"Conv$_2$", color=C_BLUE, fontsize=5.5,
                radius=0.04)
    fancy_block(ax, cx_bn - 0.18, y_bot - 0.13, 0.36, 0.26,
                title=r"BN", color=C_GREEN, fontsize=5.5,
                radius=0.04)

    # Gate σ(g)
    gate = mpatches.Circle((cx_gate, y_mid), 0.16,
                           facecolor=lighten(C_PURPLE, 0.45),
                           edgecolor=C_PURPLE, linewidth=0.8, zorder=4)
    ax.add_patch(gate)
    ax.text(cx_gate, y_mid, r"$\sigma(g)$", ha="center", va="center",
            fontsize=5.0, color=darken(C_PURPLE, 0.30), zorder=5)

    # Output
    fancy_block(ax, cx_out - 0.18, y_mid - 0.16, 0.36, 0.32,
                title=r"$\mathbf{Y}^{(t)}$", color=C_GRAY,
                fontsize=6.0, shadow=False)

    # Connections — input splits up + down
    ax.plot([cx_in + 0.18, cx_lif - 0.40], [y_mid + 0.05, y_mid + 0.05],
            color=C_GRAY, lw=0.5)
    ax.plot([cx_lif - 0.40, cx_lif - 0.40], [y_mid + 0.05, y_top - 0.05],
            color=C_GRAY, lw=0.5)
    ax.annotate("", xy=(cx_lif - 0.20, y_top), xytext=(cx_lif - 0.40, y_top),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))

    ax.plot([cx_in + 0.18, cx_lif - 0.40], [y_mid - 0.05, y_mid - 0.05],
            color=C_GRAY, lw=0.5)
    ax.plot([cx_lif - 0.40, cx_lif - 0.40], [y_mid - 0.05, y_bot + 0.05],
            color=C_GRAY, lw=0.5)
    ax.annotate("", xy=(cx_lif - 0.22, y_bot), xytext=(cx_lif - 0.40, y_bot),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))

    # LIF -> Conv -> BN sequencing (each row)
    for y in (y_top, y_bot):
        ax.annotate("", xy=(cx_conv - 0.22, y), xytext=(cx_lif + 0.20, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                    mutation_scale=4))
        ax.annotate("", xy=(cx_bn - 0.18, y), xytext=(cx_conv + 0.22, y),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                    mutation_scale=4))

    # Top BN -> gate (top); bottom BN -> gate (bottom)
    ax.plot([cx_bn + 0.18, cx_gate - 0.05], [y_top, y_mid + 0.10],
            color=C_PURPLE, lw=0.5)
    ax.plot([cx_bn + 0.18, cx_gate - 0.05], [y_bot, y_mid - 0.10],
            color=C_PURPLE, lw=0.5)
    ax.annotate("", xy=(cx_out - 0.18, y_mid), xytext=(cx_gate + 0.16, y_mid),
                arrowprops=dict(arrowstyle="-|>", color=C_PURPLE, lw=0.6,
                                mutation_scale=5))

    # Residual shortcut (dashed) — input → output bypass
    ax.annotate("", xy=(cx_out - 0.20, y0 + 0.20),
                xytext=(cx_in + 0.16, y0 + 0.20),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.4,
                                ls=(0, (2.5, 1.5)), mutation_scale=4))
    ax.text((cx_in + cx_out) / 2, y0 + 0.10, "shortcut",
            ha="center", va="center", fontsize=4.2,
            color=C_GRAY, fontstyle="italic")

    # Path labels
    ax.text(cx_lif - 0.26, y_top + 0.18, "temporal",
            ha="center", va="center", fontsize=4.4,
            color=darken(C_GRAY, 0.10), fontstyle="italic")
    ax.text(cx_lif - 0.26, y_bot - 0.18, "spatial",
            ha="center", va="center", fontsize=4.4,
            color=darken(C_GRAY, 0.10), fontstyle="italic")


def _panel_module_5qs(ax, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (c2): 5QS step function (staircase) + surrogate gradient window."""
    bg = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0,
    )
    ax.add_patch(bg)
    ax.text(x0 + 0.10, y0 + h - 0.13, "(d) 5QS quantizer",
            ha="left", va="center",
            fontsize=7.0, fontweight="bold", color="black")

    # Drawing area inside the panel
    x_ax0, x_ax1 = x0 + 0.30, x0 + w - 0.20
    y_ax0, y_ax1 = y0 + 0.30, y0 + h - 0.40

    # Axes
    ax.annotate("", xy=(x_ax1, y_ax0), xytext=(x_ax0, y_ax0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                                mutation_scale=5))
    ax.annotate("", xy=(x_ax0, y_ax1), xytext=(x_ax0, y_ax0),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=0.7,
                                mutation_scale=5))
    ax.text(x_ax1 + 0.04, y_ax0, r"$u$", ha="left", va="center",
            fontsize=5.0, color=C_GRAY)
    ax.text(x_ax0, y_ax1 + 0.06, r"$s$", ha="center", va="bottom",
            fontsize=5.0, color=C_GRAY)

    # Map u-range [0, 4] → x-axis, s-range [0, 1] → y-axis
    u_range = (0, 4)
    s_range = (0, 1)
    def ux(u): return x_ax0 + (u - u_range[0]) / (u_range[1] - u_range[0]) * (x_ax1 - x_ax0)
    def sy(s): return y_ax0 + (s - s_range[0]) / (s_range[1] - s_range[0]) * (y_ax1 - y_ax0)

    # Surrogate gradient window (rectangle in u∈[0,4], s∈[0,1])
    surr = mpatches.Rectangle(
        (ux(0), sy(0)), ux(4) - ux(0), sy(1) - sy(0),
        facecolor=lighten(C_PURPLE, 0.70), edgecolor="none", alpha=0.45,
        zorder=1,
    )
    ax.add_patch(surr)

    # 5QS step function (5 steps at s = 0, 0.25, 0.50, 0.75, 1.0)
    step_u = [0.5, 1.5, 2.5, 3.5]
    step_s = [0.25, 0.50, 0.75, 1.0]
    # Pre-step: from u=0 to u=0.5, s=0
    pts = [(ux(0), sy(0)), (ux(0.5), sy(0))]
    last_s = 0
    for u_jump, s_new in zip(step_u, step_s):
        pts.append((ux(u_jump), sy(s_new)))    # vertical jump
        # horizontal until next jump (or to u=4 at end)
        if u_jump == step_u[-1]:
            pts.append((ux(4), sy(s_new)))
        else:
            next_u = step_u[step_u.index(u_jump) + 1]
            pts.append((ux(next_u), sy(s_new)))
        last_s = s_new
    xs_step, ys_step = zip(*pts)
    ax.plot(xs_step, ys_step, color=C_BLUE, lw=1.2, zorder=4)

    # Open circles at step transitions
    for u_j, s_n in zip(step_u, step_s):
        ax.add_patch(mpatches.Circle(
            (ux(u_j), sy(s_n)), 0.012, facecolor="white",
            edgecolor=C_BLUE, linewidth=0.6, zorder=5))

    # Binary spike comparison (dashed orange)
    ax.plot([ux(0), ux(0.5), ux(0.5), ux(4)],
            [sy(0), sy(0), sy(1), sy(1)],
            color=C_ORANGE, lw=0.8, ls=(0, (2.5, 1.5)),
            zorder=3, alpha=0.9)

    # Legend
    ax.text(x_ax1 - 0.25, sy(0.30), "5QS (ours)",
            ha="right", va="center",
            fontsize=4.7, color=darken(C_BLUE, 0.20))
    ax.text(x_ax1 - 0.25, sy(0.18), "binary spike",
            ha="right", va="center",
            fontsize=4.7, color=darken(C_ORANGE, 0.20))

    # Surrogate-window caption
    ax.text((x_ax0 + x_ax1) / 2, y_ax0 - 0.16,
            r"surrogate window $[0,4]$, grad $\tfrac{1}{4}$",
            ha="center", va="center", fontsize=4.5,
            color=darken(C_PURPLE, 0.30), fontstyle="italic")

    # Tick labels (just at extremes)
    for u, lbl in [(0, "0"), (4, "4")]:
        ax.text(ux(u), y_ax0 - 0.05, lbl, ha="center", va="top",
                fontsize=4.4, color=C_GRAY)
    for s, lbl in [(0, "0"), (1, "1")]:
        ax.text(x_ax0 - 0.03, sy(s), lbl, ha="right", va="center",
                fontsize=4.4, color=C_GRAY)


def _panel_module_sham(ax, x0: float, y0: float, w: float, h: float) -> None:
    """Panel (c3): SHAM — spectral-hybrid attention (temporal + DCT spatial)."""
    bg = FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=C_LIGHT, edgecolor="#D8D8D8", linewidth=0.5, zorder=0,
    )
    ax.add_patch(bg)
    ax.text(x0 + 0.10, y0 + h - 0.13, "(e) SHAM attention",
            ha="left", va="center",
            fontsize=7.0, fontweight="bold", color="black")

    # Two sub-rows: (top) temporal amplitude branch, (bottom) DCT spatial
    y_t_in   = y0 + h - 0.30
    y_t_w    = y0 + h - 0.55
    y_t_out  = y0 + h - 0.78

    y_s_in   = y0 + 0.78
    y_s_w    = y0 + 0.50
    y_s_out  = y0 + 0.22

    cx0      = x0 + 0.45
    dx_cube  = 0.16
    n_cube   = 4

    # ---- Temporal amplitude branch (top) ----
    ax.text(x0 + 0.10, y_t_in + 0.12, "Temporal amp.",
            fontsize=4.7, fontstyle="italic",
            color=darken(C_GRAY, 0.10))

    # T=4 input cubes
    for t in range(n_cube):
        cx = cx0 + t * dx_cube
        tensor_3d(ax, cx, y_t_in, fw=0.10, fh=0.10, depth=0.05,
                  color=C_BLUE, zorder=2)
    ax.text(cx0 + n_cube * dx_cube + 0.05, y_t_in,
            rf"$T{{=}}{n_cube}$",
            ha="left", va="center", fontsize=4.5,
            color=darken(C_GRAY, 0.10))

    # 1-D temporal weight strip (4 squares with diff colors)
    weight_colors = [C_ORANGE, C_BLUE, C_GREEN, C_PURPLE]
    for t, wc in enumerate(weight_colors):
        cx = cx0 + t * dx_cube
        rect = mpatches.Rectangle(
            (cx - 0.05, y_t_w - 0.05), 0.10, 0.10,
            facecolor=lighten(wc, 0.30), edgecolor=darken(wc, 0.20),
            linewidth=0.4, zorder=3)
        ax.add_patch(rect)
    # Down-arrow (Gen → weights)
    ax.annotate("", xy=(cx0 + 1.5 * dx_cube, y_t_w + 0.07),
                xytext=(cx0 + 1.5 * dx_cube, y_t_in - 0.07),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))
    ax.text(cx0 + 1.5 * dx_cube + 0.04, (y_t_w + y_t_in) / 2,
            "Gen", fontsize=4.2, color=C_GRAY,
            ha="left", va="center")

    # Output cubes (tinted with weight color)
    for t, wc in enumerate(weight_colors):
        cx = cx0 + t * dx_cube
        tensor_3d(ax, cx, y_t_out, fw=0.10, fh=0.10, depth=0.05,
                  color=wc, zorder=2)
    # Broadcast arrow
    ax.annotate("", xy=(cx0 + 1.5 * dx_cube, y_t_out + 0.07),
                xytext=(cx0 + 1.5 * dx_cube, y_t_w - 0.07),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))
    ax.text(cx0 + 1.5 * dx_cube + 0.04, (y_t_w + y_t_out) / 2,
            "Bcast", fontsize=4.2, color=C_GRAY,
            ha="left", va="center")

    # Divider line between top and bottom rows
    ax.plot([x0 + 0.10, x0 + w - 0.10],
            [(y_t_out + y_s_in) / 2, (y_t_out + y_s_in) / 2],
            color="#D0D0D0", lw=0.4, ls=(0, (1, 1.2)))

    # ---- DCT spatial branch (bottom) ----
    ax.text(x0 + 0.10, y_s_in + 0.12, "DCT spatial",
            fontsize=4.7, fontstyle="italic",
            color=darken(C_GRAY, 0.10))

    # Single input cube (bigger)
    tensor_3d(ax, cx0 + 0.6, y_s_in, fw=0.30, fh=0.18, depth=0.10,
              color=C_BLUE, zorder=2)
    ax.text(cx0 + 0.85, y_s_in, r"$\mathbf{U}$",
            ha="left", va="center", fontsize=5.0, color=darken(C_BLUE, 0.30))

    # 2-D spatial weight grid (3×3)
    grid_cx, grid_cy = cx0 + 0.6, y_s_w
    for i in range(3):
        for j in range(3):
            shade = 0.30 + 0.20 * ((i + j) % 3) / 2
            rect = mpatches.Rectangle(
                (grid_cx - 0.10 + j * 0.07, grid_cy - 0.10 + i * 0.07),
                0.06, 0.06,
                facecolor=lighten(C_BLUE, 0.50 + 0.20 * ((i + j) % 3) / 2),
                edgecolor=darken(C_BLUE, 0.20),
                linewidth=0.3, zorder=3)
            ax.add_patch(rect)

    # DCT arrow
    ax.annotate("", xy=(grid_cx, grid_cy + 0.10),
                xytext=(grid_cx, y_s_in - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))
    ax.text(grid_cx + 0.04, (grid_cy + y_s_in) / 2,
            "DCT", fontsize=4.2, color=C_GRAY,
            ha="left", va="center")

    # Output cube (hatched, tinted)
    out_cx = cx0 + 0.6
    rect = mpatches.Rectangle(
        (out_cx - 0.18, y_s_out - 0.07), 0.36, 0.14,
        facecolor=lighten(C_BLUE, 0.30), edgecolor=darken(C_BLUE, 0.30),
        linewidth=0.5, hatch="//", zorder=3, alpha=0.85)
    ax.add_patch(rect)

    # Broadcast
    ax.annotate("", xy=(out_cx, y_s_out + 0.08),
                xytext=(out_cx, grid_cy - 0.10),
                arrowprops=dict(arrowstyle="-|>", color=C_GRAY, lw=0.5,
                                mutation_scale=4))
    ax.text(out_cx + 0.04, (grid_cy + y_s_out) / 2,
            "Bcast", fontsize=4.2, color=C_GRAY,
            ha="left", va="center")


def _panel_module_zoom(ax) -> None:
    """Bottom strip: three module zoom-in panels arranged horizontally."""
    # Layout: 3 panels at the bottom of the figure
    bottom_y = 0.05
    panel_h  = 1.80

    # Compute widths so that gaps + panels fill the 8.5 width with margins.
    # margin_left = margin_right = 0.20, gap_between = 0.12
    margin = 0.20
    gap    = 0.12
    total_w = 8.5 - 2 * margin - 2 * gap
    # Allocate widths: MFRB gets 36%, 5QS 30%, SHAM 34%
    w_mfrb  = total_w * 0.36
    w_5qs   = total_w * 0.30
    w_sham  = total_w * 0.34

    x_mfrb = margin
    x_5qs  = x_mfrb + w_mfrb + gap
    x_sham = x_5qs  + w_5qs  + gap

    _panel_module_mfrb(ax, x_mfrb, bottom_y, w_mfrb, panel_h)
    _panel_module_5qs (ax, x_5qs,  bottom_y, w_5qs,  panel_h)
    _panel_module_sham(ax, x_sham, bottom_y, w_sham, panel_h)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, default="./figures")
    p.add_argument("--out_name", type=str, default="fig1.pdf")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 5.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 5.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    _panel_architecture(ax)
    _panel_constellation(ax)
    _panel_module_zoom(ax)

    out_path = out_dir / args.out_name
    fig.savefig(out_path, dpi=300, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
