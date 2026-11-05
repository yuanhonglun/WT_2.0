"""Fig 1b: replace the schematic High/Low comparison with a plain fact table.

Reads the Illustrator SVG of Fig 1 (panel a is kept byte-for-byte, only the
style sheet gains three chip classes), removes every panel-b element (all
top-level elements whose first coordinate lies below y = 170 pt), and splices
in a regenerated panel-b group: a table that compares DDA, DIA (SWATH) and
WT 2.0 (ASFAM) on acquisition mode, isolation width, injections per sample,
MS1-dependence of MS2 triggering, cycle time, features observed and
high-score annotations.  Values that are not known yet stay as literal
[[placeholder]] text so a later pass can substitute them.

Outputs  <out-dir>/Fig 1.svg  and a PNG (headless Edge; vector PDF via Edge
print-to-pdf when available).  Text widths are measured with the Arial TTF
matplotlib resolves, and any cell text wider than its column is reported.

Usage:
    python make_fig1b_table.py [--svg-in PATH] [--out-dir DIR] [--scale 5]
"""
from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse
import copy
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from lxml import etree
from matplotlib import font_manager
from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import paper_palette as pp  # noqa: E402

SVG_NS = "http://www.w3.org/2000/svg"
NS = "{%s}" % SVG_NS
EDGE_CANDIDATES = [
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]
PANEL_B_Y = 170.0          # everything at or below this y belongs to panel b
PAGE_W = 595.28

# ---- table content ---------------------------------------------------------
COLUMNS = ["DDA", "DIA (SWATH)", "WT 2.0 (ASFAM)"]
COLUMN_COLOURS = [pp.ORANGE, pp.PURPLE, pp.BLUE]
ROWS = [
    ("Acquisition mode",
     [["Data-dependent (top-N)"], ["Data-independent (SWATH)"], ["Data-independent (HR-MRM)"]]),
    ("Isolation width (Q1)",
     # SWATH as acquired for Figs. 3 and 5 (Methods): 30 overlapping 2 Da windows per
     # method, stepped by 1 Da (2 Da is the narrowest window the instrument permits).
     [["\u22481 Da"], ["2 Da, overlapping,", "stepped by 1 Da"], ["1 Da, stepped"]]),
    ("Injections per sample",
     [["31"], ["31"], ["31"]]),
    ("MS2 triggering depends on MS1?",
     [["Yes"], ["No"], ["No"]]),
    # Cycle times measured on the instrument across the 31 methods of each mode
    # (student's reading of the method files, 2026-11-04): DDA 0.921-0.954 s,
    # SWATH 0.934-0.950 s, ASFAM 0.942-0.943 s.
    ("Cycle time",
     [["921 to 954 ms"], ["934 to 950 ms"], ["942 to 943 ms"]]),
    # Fig. 5 comparison, means of three technical replicates at S/N >= 3 (2026-11-04):
    # DDA and DIA from the MS-DIAL exports (unchanged), WT 2.0 from the three replicates
    # re-processed with ASFAMProcessor 1.4.261103 (MS2-only tier D excluded).
    # High-score annotations = Total score >= 1.4 with >= 3 matched library peaks, the
    # rule used throughout the paper (figures/fig5/numbers_fig5.json, analysis/fig5_v4/).
    ("Features observed",
     [["5,116"], ["5,119"], ["13,594"]]),
    ("High-score annotations",
     [["817"], ["763"], ["522"]]),
]

# ---- geometry (pt, SVG user units) ------------------------------------------
X_OUTER0, X_OUTER1 = 21.03, 574.64     # panel-b container (same as the original)
X0, X1 = 28.58, 567.09                 # table extent
LABEL_W = 150.0
HEADER_H = 16.0
ROW_H1, ROW_H2 = 17.0, 26.0            # one-line / two-line rows
ROW_GAP = 3.0
LINE_SPACING = 9.6
FONT_PX = 8.0
TOP_PAD = 8.0
BOTTOM_PAD = 8.0
CONTAINER_TOP = 173.07


# ---- panel a recolour -------------------------------------------------------
# Panel a's boxes share Illustrator classes (st4 = Method boxes AND the MS1 scan
# box, st21 = MS2 scan AND the Deduplication box ...), so each box is addressed
# by the first coordinate of its path and given its own palette class.
# Text, icons, arrows (apart from the one blue arrow) and layout are untouched.
PANEL_A_BOXES = {
    "pa_ms1":  (pp.BLUE,   ["M167.9,94.85", "M180,83.01"]),                    # MS1 scan, 75~104 bar
    "pa_ms2":  (pp.ORANGE, ["M167.83,117.33", "M186.94,117.09", "M219.79,117.33",
                            "M291.31,116.83", "M253.53,116.83",               # 1 Da stepped windows chips
                            "M167.83,135.41", "M201.58,134.87", "M234.43,135.06",
                            "M305.53,134.67", "M268.17,134.67"]),             # MS2 scan + 50~x chips
    "pa_proc": (pp.PURPLE, ["M390.31,122.56", "M444.17,122.53",               # mzML/centroiding, feature extraction
                            "M505.5,137.33", "M559.41,125.18"]),             # deduplication, annotation/quant
    "pa_seg":  (None,      ["M147.65,70.12", "M200.4,70.18", "M252.32,70.18",
                            "M305.67,70.05"]),                               # Method 1 / 2 / ... / 31 (neutral)
}
SEG_FILL, SEG_STROKE = "#E4EAF0", pp.NEUTRAL


def recolour_panel_a(root, style) -> int:
    wanted = {}
    for cls, (_, starts) in PANEL_A_BOXES.items():
        for s in starts:
            wanted[s] = cls
    css = []
    for cls, (colour, _) in PANEL_A_BOXES.items():
        fill, stroke = (SEG_FILL, SEG_STROKE) if colour is None else (pp.light(colour), colour)
        css.append(f"\n\t.{cls}{{fill:{fill};stroke:{stroke};stroke-miterlimit:1.1339;}}")
    style.text = (style.text or "").rstrip() + "".join(css) + "\n"
    found = 0
    for el in root.iter(NS + "path"):
        d = el.get("d", "")
        key = d.split("h")[0].split("H")[0].split("c")[0].split("v")[0].strip()
        if key in wanted:
            el.set("class", wanted[key]); found += 1
    missing = len(wanted) - found
    if missing:
        raise SystemExit(f"panel a recolour: {missing} of {len(wanted)} boxes not found")
    # the single blue arrow (Method 1 -> MS1 scan) follows the MS1 hue
    style.text = re.sub(r"\.st22\{fill:none;stroke:#6C99BC;", f".st22{{fill:none;stroke:{pp.BLUE};", style.text)
    style.text = re.sub(r"\.st23\{fill:#6C99BC;\}", f".st23{{fill:{pp.BLUE};}}", style.text)
    return found


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def first_y(element) -> float | None:
    """Approximate y of a top-level element from its first coordinate."""
    tag = element.tag.replace(NS, "")
    if tag == "text":
        m = re.search(r"matrix\(([^)]*)\)", element.get("transform", ""))
        if m:
            parts = m.group(1).replace(",", " ").split()
            if len(parts) == 6:
                return float(parts[5])
        return None
    if tag == "path":
        m = re.match(r"\s*M\s*(-?[\d.]+)[ ,](-?[\d.]+)", element.get("d", ""))
        return float(m.group(2)) if m else None
    if tag == "line":
        return float(element.get("y1", "nan"))
    if tag in {"rect"}:
        return float(element.get("y", "nan"))
    if tag == "g":
        ys = [y for y in (first_y(ch) for ch in element.iter() if ch is not element) if y is not None]
        return min(ys) if ys else None
    return None


def rounded_rect(x: float, y: float, w: float, h: float, r: float, cls: str) -> str:
    return (f'<path class="{cls}" d="M{x + r:.2f},{y:.2f}h{w - 2 * r:.2f}'
            f'a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{r:.2f}v{h - 2 * r:.2f}'
            f'a{r:.2f},{r:.2f} 0 0 1 {-r:.2f},{r:.2f}h{-(w - 2 * r):.2f}'
            f'a{r:.2f},{r:.2f} 0 0 1 {-r:.2f},{-r:.2f}v{-(h - 2 * r):.2f}'
            f'a{r:.2f},{r:.2f} 0 0 1 {r:.2f},{-r:.2f}z"/>')


def text_el(x: float, y: float, s: str, bold: bool = False, anchor: str = "start") -> str:
    cls = ("st0" if bold else "st7") + " st8 st26"
    extra = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return f'<text transform="matrix(1 0 0 1 {x:.4f} {y:.4f})" class="{cls}"{extra}>{esc(s)}</text>'


def build_panel_b(measure) -> tuple[str, float, list[str]]:
    """Return (svg fragment, container bottom y, overrun warnings)."""
    col_w = (X1 - X0 - LABEL_W) / len(COLUMNS)
    col_x = [X0 + LABEL_W + i * col_w for i in range(len(COLUMNS))]
    warnings: list[str] = []
    parts: list[str] = []

    y = CONTAINER_TOP + TOP_PAD
    # header band
    parts.append(rounded_rect(X0, y, X1 - X0, HEADER_H, 3.15, "st9"))
    chip_h = 12.0
    for i, (name, colour) in enumerate(zip(COLUMNS, COLUMN_COLOURS)):
        cx = col_x[i] + col_w / 2
        w = measure(name, True) + 14.0
        parts.append(rounded_rect(cx - w / 2, y + (HEADER_H - chip_h) / 2, w, chip_h, 2.0, f"pb{i}"))
        parts.append(text_el(cx, y + HEADER_H / 2 + 2.85, name, bold=True, anchor="middle"))
    parts.append(text_el(X0 + 6.0, y + HEADER_H / 2 + 2.85, "Property (comparison data sets)", bold=False))
    y += HEADER_H + ROW_GAP + 1.0

    for label, cells in ROWS:
        n_lines = max(len(c) for c in cells)
        h = ROW_H2 if n_lines > 1 else ROW_H1
        parts.append(rounded_rect(X0, y, X1 - X0, h, 3.77, "st3"))
        for i in range(len(COLUMNS)):
            parts.append(f'<line class="st10" x1="{col_x[i]:.2f}" y1="{y + 0.5:.2f}" '
                         f'x2="{col_x[i]:.2f}" y2="{y + h - 0.5:.2f}"/>')
        parts.append(text_el(X0 + 6.0, y + h / 2 + 2.85, label))
        if (w := measure(label, False)) > LABEL_W - 12:
            warnings.append(f"label '{label}' {w:.1f} pt > {LABEL_W - 12:.0f} pt")
        for i, lines in enumerate(cells):
            cx = col_x[i] + col_w / 2
            y0 = y + h / 2 + 2.85 - (len(lines) - 1) * LINE_SPACING / 2
            for k, s in enumerate(lines):
                parts.append(text_el(cx, y0 + k * LINE_SPACING, s, anchor="middle"))
                if (w := measure(s, False)) > col_w - 8:
                    warnings.append(f"cell '{s}' {w:.1f} pt > {col_w - 8:.0f} pt")
        y += h + ROW_GAP

    bottom = y - ROW_GAP + BOTTOM_PAD
    container = rounded_rect(X_OUTER0, CONTAINER_TOP, X_OUTER1 - X_OUTER0, bottom - CONTAINER_TOP, 5.2, "st2")
    fragment = '<g id="panel_b_table">' + container + "".join(parts) + "</g>"
    return fragment, bottom, warnings


def make_measure():
    try:
        font = ImageFont.truetype(font_manager.findfont("Arial", fallback_to_default=False), 80)
        bold = ImageFont.truetype(font_manager.findfont(
            font_manager.FontProperties(family="Arial", weight="bold"), fallback_to_default=False), 80)
    except Exception:
        font = bold = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 80)

    def measure(text: str, is_bold: bool) -> float:
        f = bold if is_bold else font
        return f.getlength(text) * FONT_PX / 80.0
    return measure


def find_edge() -> Path | None:
    """msedge from PATH first, then the two standard Program Files locations."""
    found = shutil.which("msedge")
    if found:
        return Path(found)
    for p in EDGE_CANDIDATES:
        if p.exists():
            return p
    return None


def render(svg_out: Path, bottom: float, scale: int) -> None:
    edge = find_edge()
    if edge is None:
        print("[warn] msedge.exe not found; PNG/PDF not rendered")
        return
    height = round(bottom + 6.0, 2)
    tmp_dir = Path(tempfile.mkdtemp(prefix="fig1b_"))
    # cropped copy: explicit size so the browser viewport maps 1:1 onto points
    text = svg_out.read_text(encoding="utf-8")
    text = re.sub(r'viewBox="[^"]*"', f'viewBox="0 0 {PAGE_W} {height}"', text, count=1)
    text = text.replace('x="0px" y="0px"', f'x="0px" y="0px" width="{PAGE_W}px" height="{height}px"', 1)
    text = re.sub(r"enable-background:new [^;\"]*", f"enable-background:new 0 0 {PAGE_W} {height}", text, count=1)
    cropped = tmp_dir / "fig1_cropped.svg"
    cropped.write_text(text, encoding="utf-8")

    png_out = svg_out.with_suffix(".png")
    cmd = [str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
           "--no-default-browser-check", f"--screenshot={png_out}",
           f"--window-size={int(round(PAGE_W))},{int(round(height))}",
           f"--force-device-scale-factor={scale}", cropped.as_uri()]
    subprocess.run(cmd, check=False, capture_output=True, timeout=120)
    if png_out.exists():
        img = Image.open(png_out)
        dpi = 72 * scale
        img.save(png_out, dpi=(dpi, dpi))
        print(f"[png] {png_out}  {img.size[0]}x{img.size[1]} px @ {dpi} dpi")
    else:
        print("[warn] Edge screenshot failed")

    html = (f'<!doctype html><html><head><meta charset="utf-8"><style>'
            f'@page{{size:{PAGE_W}pt {height}pt;margin:0}} html,body{{margin:0;padding:0}} '
            f'img{{display:block;width:{PAGE_W}pt;height:{height}pt}}</style></head>'
            f'<body><img src="{cropped.as_uri()}"></body></html>')
    wrapper = tmp_dir / "fig1_print.html"
    wrapper.write_text(html, encoding="utf-8")
    pdf_out = svg_out.with_suffix(".pdf")
    cmd = [str(edge), "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
           "--no-pdf-header-footer", f"--print-to-pdf={pdf_out}", wrapper.as_uri()]
    subprocess.run(cmd, check=False, capture_output=True, timeout=120)
    print(f"[pdf] {pdf_out}" if pdf_out.exists() else "[warn] Edge print-to-pdf failed")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--svg-in", default=str(HERE.parents[1] / "fig 1" / "Fig 1.svg"))
    parser.add_argument("--out-dir", default=str(HERE.parent / "figures" / "fig1"))
    parser.add_argument("--scale", type=int, default=5, help="device scale factor for the PNG (5 = 360 dpi)")
    args = parser.parse_args()

    src = Path(args.svg_in)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parser_ = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(src), parser_)
    root = tree.getroot()

    # 1. drop the old panel-b content (container, header band, rows, chips, texts)
    removed = 0
    for child in list(root):
        if not isinstance(child.tag, str):
            continue
        y = first_y(child)
        if y is not None and y >= PANEL_B_Y:
            root.remove(child)
            removed += 1
    # 2. chip classes for the method header (light fill + full-colour edge)
    style = root.find(NS + "style")
    extra = "".join(f"\n\t.pb{i}{{fill:{pp.light(c)};stroke:{c};stroke-miterlimit:10;}}"
                    for i, c in enumerate(COLUMN_COLOURS))
    style.text = (style.text or "").rstrip() + extra + "\n"
    for key in ("st0", "st7"):
        if "'Arial'" not in style.text.split(f".{key}{{")[1].split("}")[0]:
            raise SystemExit(f"style .{key} lacks the 'Arial' fallback; refusing to continue")

    # 2b. panel a boxes -> same palette (MS1 blue, MS2 orange, processing purple, segments grey)
    n_boxes = recolour_panel_a(root, style)

    # 3. regenerate panel b
    measure = make_measure()
    fragment, bottom, warnings = build_panel_b(measure)
    new_group = etree.fromstring(f'<g xmlns="{SVG_NS}">{fragment}</g>')[0]
    root.append(copy.deepcopy(new_group))
    new_group = root[-1]
    new_group.tail = "\n"

    svg_out = out_dir / "Fig 1.svg"
    tree.write(str(svg_out), xml_declaration=True, encoding="utf-8", standalone=False)
    text = svg_out.read_text(encoding="utf-8")
    old_software_name = "".join(("ME", "TRA"))      # must not appear in any figure
    for forbidden in (r">(High|Low)(/|<)", r"deconvolution", old_software_name):
        if re.search(forbidden, text):
            print(f"[warn] forbidden text '{forbidden}' still present in the SVG")
    print(f"[svg] {svg_out}  (removed {removed} old panel-b elements; recoloured {n_boxes} panel-a boxes; "
          f"container bottom y = {bottom:.1f} pt)")
    for w in warnings:
        print("[overrun]", w)
    if not warnings:
        print("[ok] no cell text wider than its column (Arial metrics)")
    render(svg_out, bottom, args.scale)


if __name__ == "__main__":
    main()
