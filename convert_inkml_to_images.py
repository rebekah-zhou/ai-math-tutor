import os
import glob
import random
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw
import re

# ─── PARAMETERS ────────────────────────────────────────
DATA_ROOT = "data"          # Root folder with train/, valid/, test/
OUTPUT_ROOT = "data/images"
CANVAS_SIZE = 256
MARGIN = 20
LINE_WIDTH = 3
TRAIN_SAMPLE_N = 10000
SPLITS = ["train", "valid", "test"]

# match ints, decimals, or floats in e-notation, no capturing groups
FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def parse_inkml(path):
    """Parse InkML trace into a list of (x,y) points."""
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
    """Normalize strokes to fit canvas while preserving aspect ratio."""
    all_pts = np.vstack(strokes)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    
    # Calculate aspect ratio preserving scale
    width = max_xy[0] - min_xy[0]
    height = max_xy[1] - min_xy[1]
    scale = min(
        (canvas_size - 2*margin) / max(width, 1e-6),
        (canvas_size - 2*margin) / max(height, 1e-6)
    )
    
    # Center the equation
    scaled_w = width * scale
    scaled_h = height * scale
    x_offset = (canvas_size - scaled_w) / 2
    y_offset = (canvas_size - scaled_h) / 2
    
    normalized = []
    for stroke in strokes:
        pts = (np.array(stroke) - min_xy) * scale
        pts[:, 0] += x_offset
        pts[:, 1] += y_offset
        normalized.append([tuple(p) for p in pts])
    return normalized


def strokes_to_image(strokes, canvas_size=CANVAS_SIZE, line_width=LINE_WIDTH):
    """Rasterize strokes with anti-aliasing and proper line joins."""
    scale = 2
    big_size = canvas_size * scale
    img = Image.new('L', (big_size, big_size), color=255)
    draw = ImageDraw.Draw(img)
    
    for stroke in strokes:
        if len(stroke) > 1:
            big_stroke = [(x * scale, y * scale) for x, y in stroke]
            draw.line(
                big_stroke, 
                fill=0, 
                width=line_width * scale,
                joint="curve"
            )
            
            for x, y in big_stroke:
                r = (line_width * scale) / 2
                draw.ellipse(
                    [(x-r, y-r), (x+r, y+r)],
                    fill=0
                )
    
    img = img.resize(
        (canvas_size, canvas_size),
        Image.Resampling.LANCZOS
    )
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
    n_samples = min(TRAIN_SAMPLE_N, len(train_files))
    train_sample = random.sample(train_files, k=n_samples)

    # Convert each split
    for split in SPLITS:
        if split == 'train':
            convert_split(split, sample_paths=train_sample)
        else:
            convert_split(split)
    print("✅ Conversion complete!")
