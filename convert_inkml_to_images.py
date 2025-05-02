import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw

# ─── PARAMETERS ──────────────────────────────────────────────────────────────
DATA_ROOT     = "data"      # top-level folder containing train/, valid/, test/, …
OUTPUT_ROOT   = "data/images"    # where to write out images
CANVAS_SIZE   = 128         # output image size (pixels)
MARGIN        = 5           # whitespace border
LINE_WIDTH    = 2           # stroke thickness (pixels)
INKML_NS      = "{http://www.w3.org/2003/InkML}"
SPLITS        = ["train", "valid", "test", "synthetic", "symbols"]
# ────────────────────────────────────────────────────────────────────────────────

import re


import xml.etree.ElementTree as ET
import re

# match ints, decimals, or floats in e-notation, no capturing groups
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def parse_inkml(path):
    """Parse an InkML trace into a list of (x,y) points, dropping any extra dims."""
    tree    = ET.parse(path)
    root    = tree.getroot()
    strokes = []

    for trace in root.findall('.//{*}trace'):
        text = (trace.text or "").strip()
        if not text:
            continue

        nums   = FLOAT_RE.findall(text)
        coords = list(map(float, nums))

        # If there's a time‐stamp, coords per point = 3, else = 2
        stride = 3 if len(coords) % 3 == 0 else 2
        pts = []
        for i in range(0, len(coords), stride):
            x = coords[i]
            y = coords[i+1]
            pts.append((x, y))
        if pts:
            strokes.append(pts)

    return strokes




def normalize_strokes(strokes, canvas_size=CANVAS_SIZE, margin=MARGIN):
    """Shift & scale so that everything fits in a [margin, canvas_size−margin]^2 box."""
    # flatten
    all_pts = np.vstack(strokes)
    min_xy  = all_pts.min(axis=0)
    max_xy  = all_pts.max(axis=0)
    scale   = (canvas_size - 2*margin) / max((max_xy - min_xy).max(), 1e-6)
    normalized = []
    for stroke in strokes:
        pts = (np.array(stroke) - min_xy) * scale + margin
        normalized.append([tuple(p) for p in pts])
    return normalized

def strokes_to_image(strokes, canvas_size=CANVAS_SIZE, line_width=LINE_WIDTH):
    """Rasterize strokes onto a white background."""
    img  = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        if len(stroke) > 1:
            draw.line(stroke, fill=0, width=line_width)
    return img

def convert_split(split_name):
    src_dir = os.path.join(DATA_ROOT, split_name)
    out_dir = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)

    inkml_paths = glob.glob(os.path.join(src_dir, "*.inkml"))
    for inkml in inkml_paths:
        try:
            strokes = parse_inkml(inkml)
            norm    = normalize_strokes(strokes)
            img     = strokes_to_image(norm)
            fname   = os.path.splitext(os.path.basename(inkml))[0] + ".png"
            img.save(os.path.join(out_dir, fname))
        except Exception as e:
            print(f"❌ Failed {inkml}: {e}")

if __name__ == "__main__":
    for split in SPLITS:
        convert_split(split)
    print("✅ Conversion complete!")
