# -*- coding: utf-8 -*-
"""
rotate_test.py
==============
驗證「2 字直式貼紙讀不出來」是否純粹是 90 度轉正判斷失誤造成：
同一張圖分別以原圖 / 順時針 90 / 逆時針 90 餵進 production 的
TextSystemDLA.predict，比較讀出的文字與信心。
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA

VARIANTS = [
    ("原圖", None),
    ("順時針90", cv2.ROTATE_90_CLOCKWISE),
    ("逆時針90", cv2.ROTATE_90_COUNTERCLOCKWISE),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="tasks/ocr/yoshi_det_v01_03.dla")
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--rec", default="tasks/ocr/yoshi_rec_v01_05.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            continue
        print(f"\n=== {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]} ===")
        for label, code in VARIANTS:
            frame = img if code is None else cv2.rotate(img, code)
            try:
                rec_res, boxes, is_flip, is_rot90, _ = ts.predict(frame, obj_type="sticker")
            except Exception as exc:
                print(f"    {label:8} predict 失敗: {exc}")
                continue
            texts = [t for t, _ in rec_res] if rec_res else []
            confs = [c for _, c in rec_res] if rec_res else []
            best = f"{max(confs):.3f}" if confs else "-"
            nb = 0 if boxes is None else len(boxes)
            print(f"    {label:8} 框={nb} 內部rot90={is_rot90} flip={is_flip} "
                  f"conf={best}  「{' | '.join(texts)}」")


if __name__ == "__main__":
    main()
