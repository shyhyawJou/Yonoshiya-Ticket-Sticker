# -*- coding: utf-8 -*-
"""
ocr_batch.py
============
對一整批已裁切影像跑完整 det -> cls -> rec（走 production 的
TextSystemDLA.predict，跟 app 內 AsyncOCR 的呼叫方式一致），
把每張的框數與讀出的文字印出來。

用法:
    python3 ocr_batch.py --det ... --rec ... --type sticker <資料夾>
"""
import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--rec", required=True)
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("--type", default="sticker")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只印出文字包含這些字之一的結果(逗號分隔)")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    files = []
    for p in args.images:
        files += sorted(glob.glob(os.path.join(p, "*.jpg"))) if os.path.isdir(p) else [p]
    if args.limit:
        files = files[: args.limit]

    keys = [k for k in args.only.split(",") if k]

    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    print(f"\n### det={os.path.basename(args.det)} rec={os.path.basename(args.rec)} "
          f"type={args.type} 共 {len(files)} 張")

    zero_box = 0
    zero_text = 0
    shown = 0
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            rec_res, dt_boxes, is_flip, is_rot90, _ = ts.predict(img, obj_type=args.type)
        except Exception as exc:
            print(f"  {os.path.basename(path)}  !! predict 失敗: {exc}")
            continue

        nb = 0 if dt_boxes is None else len(dt_boxes)
        texts = [t for t, _ in rec_res] if rec_res else []
        confs = [c for _, c in rec_res] if rec_res else []

        if nb == 0:
            zero_box += 1
        if not texts:
            zero_text += 1

        joined = " | ".join(texts)
        if keys and not any(k in joined for k in keys):
            continue

        best = f"{max(confs):.3f}" if confs else "-"
        print(f"  {os.path.basename(path)}  {img.shape[1]}x{img.shape[0]}  "
              f"框={nb} flip={is_flip} rot90={is_rot90} conf_max={best}  「{joined}」")
        shown += 1

    print(f"\n--- 統計 ---")
    print(f"總數 {len(files)}，零框 {zero_box} 張，有框但讀不出文字 {zero_text - zero_box} 張")
    if keys:
        print(f"符合關鍵字 {keys} 的有 {shown} 張")


if __name__ == "__main__":
    main()
