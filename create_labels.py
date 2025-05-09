import os
import glob
import csv
import xml.etree.ElementTree as ET

DATA_ROOT   = "data"           # where your .inkml split folders live
IMAGE_ROOT  = "data/images"    # where you wrote out the PNGs
OUT_CSV     = os.path.join(IMAGE_ROOT, "labels.csv")

def extract_label(inkml_path):
    tree = ET.parse(inkml_path)
    root = tree.getroot()
    # try normalizedLabel first, then label
    ann_norm = root.find('.//{*}annotation[@type="normalizedLabel"]')
    ann_raw  = root.find('.//{*}annotation[@type="label"]')
    if ann_norm is not None and ann_norm.text:
        return ann_norm.text.strip()
    if ann_raw is not None and ann_raw.text:
        return ann_raw.text.strip()
    return ""

with open(OUT_CSV, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["split","id","label","image_path"])

    # only split folders you actually converted
    for split in ["train","valid","test"]:
        img_dir = os.path.join(IMAGE_ROOT, split)
        ink_dir = os.path.join(DATA_ROOT,   split)
        # glob over PNGs, not InkMLs
        for img_path in glob.glob(os.path.join(img_dir, "*.png")):
            sid = os.path.splitext(os.path.basename(img_path))[0]
            inkml_path = os.path.join(ink_dir, sid + ".inkml")
            if not os.path.exists(inkml_path):
                # skip if for some reason the InkML is missing
                continue
            label = extract_label(inkml_path)
            # write a relative path so your Dataset can read it directly
            rel_img = os.path.relpath(img_path)
            writer.writerow([split, sid, label, rel_img])

print("✅ labels.csv regenerated for existing images.")
