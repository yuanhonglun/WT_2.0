"""Colour-vision-deficiency check for the produced figure PNGs.

For every PNG given (files or directories), simulate deuteranopia and
protanopia with colorspacious (sRGB1+CVD, severity 100) and save a
side-by-side  original | deuteranopia | protanopia  image to
<out-dir>/<name>_cvd.png.  The palette colours (paper_palette) are also run
through both simulations and the minimum pairwise CIEDE2000 distance is
printed for the edge colours and for the light fills, together with
colorspacious' own CAM02-UCS deltaE.

Usage:
    python cvd_check.py <png or dir> [...] [--out-dir DIR] [--palette main|full]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
from colorspacious import cspace_convert, deltaE
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402

SIMULATIONS = {
    "deuteranopia": {"name": "sRGB1+CVD", "cvd_type": "deuteranomaly", "severity": 100},
    "protanopia": {"name": "sRGB1+CVD", "cvd_type": "protanomaly", "severity": 100},
}


# --------------------------------------------------------------------------- #
# colour metrics
# --------------------------------------------------------------------------- #
def simulate(rgb01: np.ndarray, space: dict) -> np.ndarray:
    out = cspace_convert(rgb01, space, "sRGB1")
    return np.clip(out, 0.0, 1.0)


def ciede2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> float:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005)."""
    L1, a1, b1 = (float(v) for v in lab1)
    L2, a2, b2 = (float(v) for v in lab2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    Cbar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - math.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0 if (a2p or b2) else 0.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))
    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hbp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hbp = (h1p + h2p + 360.0) / 2.0
    else:
        hbp = (h1p + h2p - 360.0) / 2.0
    T = (1.0 - 0.17 * math.cos(math.radians(hbp - 30.0)) + 0.24 * math.cos(math.radians(2.0 * hbp))
         + 0.32 * math.cos(math.radians(3.0 * hbp + 6.0)) - 0.20 * math.cos(math.radians(4.0 * hbp - 63.0)))
    dtheta = 30.0 * math.exp(-(((hbp - 275.0) / 25.0) ** 2))
    RC = 2.0 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    SL = 1.0 + 0.015 * (Lbp - 50.0) ** 2 / math.sqrt(20.0 + (Lbp - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2.0 * dtheta)) * RC
    tL, tC, tH = dLp / (kL * SL), dCp / (kC * SC), dHp / (kH * SH)
    return math.sqrt(tL * tL + tC * tC + tH * tH + RT * tC * tH)


def _selftest_ciede2000() -> None:
    """Three reference pairs from Sharma et al. (2005), Table 1."""
    cases = [
        ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
        ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
        ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ]
    for lab1, lab2, expected in cases:
        got = ciede2000(lab1, lab2)
        if abs(got - expected) > 1e-3:
            print(f"[warn] CIEDE2000 self-test mismatch: {got:.4f} vs {expected:.4f}")


def to_lab(rgb01: np.ndarray) -> np.ndarray:
    return cspace_convert(rgb01, "sRGB1", "CIELab")


def pairwise_min(colours: dict[str, str], space: dict | None):
    rgb = np.array([pp.hex_to_rgb01(c) for c in colours.values()])
    if space is not None:
        rgb = simulate(rgb, space)
    names = list(colours)
    lab = to_lab(rgb)
    best = None
    best_cam = None
    for i, j in itertools.combinations(range(len(names)), 2):
        d = ciede2000(lab[i], lab[j])
        d_cam = float(deltaE(rgb[i], rgb[j], input_space="sRGB1", uniform_space="CAM02-UCS"))
        if best is None or d < best[0]:
            best = (d, names[i], names[j])
        if best_cam is None or d_cam < best_cam[0]:
            best_cam = (d_cam, names[i], names[j])
    return best, best_cam, {n: pp.rgb01_to_hex(c) for n, c in zip(names, rgb)}


def palette_report(colours: dict[str, str], label: str) -> list[str]:
    lines = [f"palette ({label}): " + ", ".join(f"{n}={c}" for n, c in colours.items())]
    fills = {n: pp.light(c) for n, c in colours.items()}
    for cond, space in [("original", None)] + list(SIMULATIONS.items()):
        (d, a, b), (dc, ac, bc), simulated = pairwise_min(colours, space)
        (fd, fa, fb), _, _ = pairwise_min(fills, space)
        lines.append(
            f"  {cond:13s} min CIEDE2000 edges {d:5.1f} ({a}-{b}) | fills {fd:5.1f} ({fa}-{fb})"
            f" | colorspacious CAM02-UCS dE edges {dc:5.1f} ({ac}-{bc})"
        )
        if space is not None:
            lines.append("                simulated edges: " + ", ".join(f"{n}={c}" for n, c in simulated.items()))
    return lines


# --------------------------------------------------------------------------- #
# image simulation
# --------------------------------------------------------------------------- #
def _label_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_manager.findfont(pp.font_family()), size)
    except Exception:
        return ImageFont.load_default()


def check_png(path: Path, out_dir: Path, max_width: int) -> Path:
    image = Image.open(path).convert("RGB")
    if image.width > max_width:
        scale = max_width / image.width
        image = image.resize((max_width, max(1, int(round(image.height * scale)))), Image.LANCZOS)
    arr = np.asarray(image, dtype=float) / 255.0
    panels = [("original", image)]
    for name, space in SIMULATIONS.items():
        sim = simulate(arr, space)
        panels.append((name, Image.fromarray((sim * 255.0 + 0.5).astype(np.uint8))))
    gap = max(8, image.width // 100)
    header = max(24, image.width // 30)
    canvas = Image.new("RGB", (3 * image.width + 2 * gap, image.height + header), "white")
    draw = ImageDraw.Draw(canvas)
    font = _label_font(max(12, header - 8))
    for k, (name, panel) in enumerate(panels):
        x = k * (image.width + gap)
        canvas.paste(panel, (x, header))
        draw.text((x + 4, 2), name, fill=(35, 24, 21), font=font)
        if k:
            draw.line([(x - gap // 2, 0), (x - gap // 2, canvas.height)], fill=(200, 200, 200), width=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}_cvd.png"
    canvas.save(out)
    return out


def collect(paths: list[str]) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found.extend(sorted(q for q in p.glob("*.png") if not q.stem.endswith("_cvd")))
        elif p.suffix.lower() == ".png":
            found.append(p)
        else:
            print(f"[skip] not a PNG: {p}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="PNG files or directories containing PNGs")
    parser.add_argument("--out-dir", default=str(HERE.parent / "figures" / "cvd_check"))
    parser.add_argument("--palette", default="full", choices=["main", "full", "cohort", "method", "recovery"])
    parser.add_argument("--max-width", type=int, default=1600,
                        help="downscale wider images before simulation (speed; default 1600 px)")
    args = parser.parse_args()

    _selftest_ciede2000()
    palette = {
        "main": pp.PALETTE_MAIN, "full": pp.PALETTE_FULL, "cohort": pp.COHORT,
        "method": {"DDA": pp.ORANGE, "DIA": pp.PURPLE, "ASFAM": pp.BLUE},
        "recovery": pp.RECOVERY_CLASS,
    }[args.palette]
    for line in palette_report(palette, args.palette):
        print(line)

    out_dir = Path(args.out_dir)
    for png in collect(args.paths):
        out = check_png(png, out_dir, args.max_width)
        print(f"[cvd] {png} -> {out}")


if __name__ == "__main__":
    main()
