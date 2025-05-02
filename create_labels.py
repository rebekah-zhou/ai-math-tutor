import os, glob, csv, xml.etree.ElementTree as ET

def extract_label(path):
    tree = ET.parse(path); root = tree.getroot()
    norm = root.find('.//{*}annotation[@type="normalizedLabel"]')
    lbl  = root.find('.//{*}annotation[@type="label"]')
    return (norm or lbl).text.strip()

with open('data/images/labels.csv','w',newline='') as f:
    w = csv.writer(f); w.writerow(['split','id','label','image_path'])
    for split in ['train','valid','test']:
      for inkml in glob.glob(f"data/{split}/*.inkml"):
        sid   = os.path.splitext(os.path.basename(inkml))[0]
        label = extract_label(inkml)
        img   = f"images/{split}/{sid}.png"
        w.writerow([split, sid, label, img])
