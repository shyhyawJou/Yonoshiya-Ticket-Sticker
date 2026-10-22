# -*- coding: utf-8 -*-
"""
orient_probe.py
===============
把 90 度轉正判準的每一項輸入都攤開來看：每個 det 框的寬高、長短邊比、
是否被判為直式、面積佔比，以及最後的面積加權比例 vs 門檻。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA
import logic.orientation as orientation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="tasks/ocr/yoshi_det_v01_03.dla")
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--rec", default="tasks/ocr/yoshi_rec_v01_05.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    files = []
    for p in args.images:
        files += sorted(glob.glob(os.path.join(p, "*.jpg"))) if os.path.isdir(p) else [p]

    print("門檻:", {k: v for k, v in vars(orientation).items()
                  if k.isupper() and isinstance(v, (int, float))})

    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        boxes, _ = ts.run_det(img)
        print(f"\n--- {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}  框={len(boxes)} ---")

        vert_area = 0.0
        total_area = 0.0
        for i, b in enumerate(boxes):
            b = np.array(b, dtype=np.float32)
            w, h, ss, ls = ts._get_box_dims(b)
            aspect = ls / max(ss, 1e-6)
            is_vert = h > w * 1.4
            area = w * h
            total_area += area
            if is_vert:
                vert_area += area
            print(f"    框{i}: width(排列方向)={w:6.1f} height(字高)={h:6.1f} "
                  f"長短比={aspect:5.2f} 面積={area:9.0f} 直式={is_vert}")

        ratio = vert_area / total_area if total_area else 0.0
        print(f"    直式面積佔比 = {ratio:.3f}  (門檻 {orientation.AREA_RATIO_THRESHOLD})")
        print(f"    needs_quarter_turn -> {orientation.needs_quarter_turn(boxes)}")
        print(f"    _check_boxes_need_rotate_90 -> {ts._check_boxes_need_rotate_90(boxes)}")


if __name__ == "__main__":
    main()
