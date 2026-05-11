"""
VisDrone annotation -> YOLO format converter.
VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)

Usage: python convert_visdrone.py --src /workspace/tly/uav-detection/data/VisDrone
"""
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# VisDrone categories 1-10 -> YOLO 0-9
# 0 = ignored, 11 = others, both skipped
CAT_MAP = {i: i - 1 for i in range(1, 11)}


def convert_one(txt_path, img_path, out_label_path):
    img = Image.open(img_path)
    img_w, img_h = img.size
    lines = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue
            cat_id = int(parts[5])
            if cat_id not in CAT_MAP:
                continue
            bw, bh = int(parts[2]), int(parts[3])
            if bw <= 0 or bh <= 0:
                continue
            yolo_cls = CAT_MAP[cat_id]
            xc = (int(parts[0]) + bw / 2) / img_w
            yc = (int(parts[1]) + bh / 2) / img_h
            lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {bw/img_w:.6f} {bh/img_h:.6f}")
    with open(out_label_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    args = parser.parse_args()
    src = Path(args.src)

    for split in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']:
        split_dir = src / split
        if not split_dir.exists():
            print(f"Skip {split} (not found)")
            continue
        img_dir = split_dir / 'images'
        ann_dir = split_dir / 'annotations'
        lbl_dir = split_dir / 'labels'
        lbl_dir.mkdir(exist_ok=True)

        # Handle both possible structures: images directly in split_dir or in images/
        if not img_dir.exists():
            # Images might be directly in split_dir
            img_dir = split_dir
        if not ann_dir.exists():
            ann_dir = split_dir

        imgs = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')))
        print(f"Converting {split}: {len(imgs)} images")

        for img_path in tqdm(imgs):
            ann_path = ann_dir / f'{img_path.stem}.txt'
            lbl_path = lbl_dir / f'{img_path.stem}.txt'
            if not ann_path.exists():
                lbl_path.touch()
                continue
            convert_one(ann_path, img_path, lbl_path)
    print("Done!")


if __name__ == '__main__':
    main()
