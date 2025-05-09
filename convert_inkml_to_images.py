import os
import glob
import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import re

# ─── PARAMETERS ──────────────────────────────────────────────────────────────
DATA_ROOT       = "data"               # top-level folder with train/, valid/, test/
OUTPUT_ROOT     = "data/images"         # output directory for PNGs
CANVAS_SIZE     = 128                   # output image size (pixels)
MARGIN          = 5                     # whitespace border
LINE_WIDTH      = 2                     # stroke thickness (pixels)
TRAIN_SAMPLE_N  = 10000                 # number of train files to convert
# Only process these splits
SPLITS          = ["train", "valid", "test"]
# ────────────────────────────────────────────────────────────────────────────────

# match ints, decimals, or floats in e-notation, no capturing groups
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_inkml(path):
    """Parse an InkML trace into a list of (x,y) points, dropping any extra dims."""
    tree = ET.parse(path)
    root = tree.getroot()
    strokes = []
    for trace in root.findall('.//{*}trace'):
        text = (trace.text or "").strip()
        if not text:
            continue
        nums = FLOAT_RE.findall(text)
        coords = list(map(float, nums))
        # If time‐stamp present (x,y,t) use stride=3, else stride=2
        stride = 3 if len(coords) % 3 == 0 else 2
        pts = []
        for i in range(0, len(coords), stride):
            x, y = coords[i], coords[i+1]
            pts.append((x, y))
        if pts:
            strokes.append(pts)
    return strokes


def normalize_strokes(strokes, canvas_size=CANVAS_SIZE, margin=MARGIN):
    """Normalize stroke coordinates to fit in canvas."""
    all_pts = np.vstack(strokes)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    scale = (canvas_size - 2*margin) / max((max_xy - min_xy).max(), 1e-6)
    normalized = []
    for stroke in strokes:
        pts = (np.array(stroke) - min_xy) * scale + margin
        normalized.append([tuple(p) for p in pts])
    return normalized


def strokes_to_image(strokes, canvas_size=CANVAS_SIZE, line_width=LINE_WIDTH):
    """Rasterize strokes onto a white background."""
    img = Image.new('L', (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(img)
    for stroke in strokes:
        if len(stroke) > 1:
            draw.line(stroke, fill=0, width=line_width)
    return img


def convert_split(split_name, sample_paths=None):
    """Convert InkMLs in a split to PNGs."""
    src_dir = os.path.join(DATA_ROOT, split_name)
    out_dir = os.path.join(OUTPUT_ROOT, split_name)
    os.makedirs(out_dir, exist_ok=True)
    # choose file list: sampled or full
    if sample_paths is None:
        inkml_paths = glob.glob(os.path.join(src_dir, "*.inkml"))
    else:
        inkml_paths = sample_paths
    for inkml in inkml_paths:
        try:
            strokes = parse_inkml(inkml)
            norm = normalize_strokes(strokes)
            img = strokes_to_image(norm)
            fname = os.path.splitext(os.path.basename(inkml))[0] + ".png"
            img.save(os.path.join(out_dir, fname))
        except Exception as e:
            print(f"❌ Failed {inkml}: {e}")


if __name__ == "__main__":
    # Prepare train sample
    train_files = glob.glob(os.path.join(DATA_ROOT, 'train', '*.inkml'))
    random.seed(42)
    train_sample = random.sample(train_files, k=min(TRAIN_SAMPLE_N, len(train_files)))

    # Convert each split
    for split in SPLITS:
        if split == 'train':
            convert_split(split, sample_paths=train_sample)
        else:
            convert_split(split)
    print("✅ Conversion complete!")
