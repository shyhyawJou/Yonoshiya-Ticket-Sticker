# -*- coding: utf-8 -*-
"""
det_batch.py
============
把一整批已裁切影像餵給 det，統計有多少張「框不出任何文字框」，
並把零框的檔名列出來，順便印出機率圖最大值以區分
「模型沒反應」與「有反應但被 box_thresh 擋掉」。

用法:
    python3 det_batch.py --det tasks/ocr/yoshi_det_v01_03.dla <資料夾或影像...>
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "*.jpg")))
        else:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--rec", default="tasks/ocr/yoshi_rec_v01_05.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    files = collect(args.images)
    if args.limit:
        files = files[: args.limit]

    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    zero = []          # 完全沒框
    blocked = []       # 有候選但被 box_thresh 擋掉
    counts = []

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        boxes, _ = ts.run_det(img)
        n = len(boxes)
        counts.append(n)

        if n == 0:
            # 再看一次原始機率圖，判斷是「模型沒反應」還是「被門檻擋掉」
            data, shape = ts._det_preprocess(img)
            ori_h, ori_w, ratio = shape
            cw, ch = int(round(ori_w * ratio)), int(round(ori_h * ratio))
            flat = ts.det_model.run(np.expand_dims(data, 0).astype(np.float16))
            p = flat.reshape(1, 640, 640, 1)[0, :ch, :cw, 0]

            best = 0.0
            mask = (p > ts.det_thresh).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours[: ts.det_max_candidates]:
                pts, sside = ts._get_mini_boxes(c)
                if sside < ts.min_size:
                    continue
                best = max(best, ts._box_score_fast(p, np.array(pts).reshape(-1, 2)))

            entry = (os.path.basename(path), float(p.max()), best, len(contours))
            if p.max() <= ts.det_thresh:
                zero.append(entry)
            else:
                blocked.append(entry)

    total = len(counts)
    nz = sum(1 for c in counts if c == 0)
    print(f"\n================ {os.path.basename(args.det)} ================")
    print(f"影像總數: {total}")
    print(f"有框出文字: {total - nz} 張 ({100.0 * (total - nz) / max(total, 1):.1f}%)")
    print(f"零框      : {nz} 張 ({100.0 * nz / max(total, 1):.1f}%)")
    if counts:
        print(f"每張框數: 平均 {np.mean(counts):.2f}, 中位數 {int(np.median(counts))}, "
              f"最大 {max(counts)}")

    if zero:
        print(f"\n-- 機率圖最大值就低於 det_thresh({ts.det_thresh}) = 模型對這張沒反應 ({len(zero)} 張) --")
        for name, pmax, best, nc in zero[:15]:
            print(f"    {name}  機率max={pmax:.4f}")

    if blocked:
        print(f"\n-- 有候選區域但被 box_thresh({ts.det_box_thresh}) 或 min_size 擋掉 ({len(blocked)} 張) --")
        for name, pmax, best, nc in blocked[:15]:
            print(f"    {name}  機率max={pmax:.4f} 最佳框分數={best:.4f} 輪廓數={nc}")

    if not zero and not blocked:
        print("\n沒有任何一張零框。")


if __name__ == "__main__":
    main()
