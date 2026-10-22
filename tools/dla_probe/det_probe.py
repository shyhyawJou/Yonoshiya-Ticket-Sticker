# -*- coding: utf-8 -*-
"""
det_probe.py
============
單獨診斷 OCR det 的 DLA 模型：用 production 程式自己的前後處理
(TextSystemDLA._det_preprocess / _boxes_from_bitmap)，把每一道閘門
的數量都印出來，區分「門檻擋掉」與「模型輸出本身是壞的」。

用法:
    python3 det_probe.py --det tasks/ocr/yoshi_det_v01_03.dla <影像...>
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--rec", default="tasks/ocr/yoshi_rec_v01_05.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    print(f"### det 模型: {args.det}")
    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    for path in args.images:
        img = cv2.imread(path)
        if img is None:
            print(f"  讀不到影像: {path}")
            continue

        print(f"\n--- {os.path.basename(path)}  原圖 {img.shape[1]}x{img.shape[0]} ---")

        data, shape = ts._det_preprocess(img)
        ori_h, ori_w, ratio = shape
        content_w = int(round(ori_w * ratio))
        content_h = int(round(ori_h * ratio))
        print(f"    letterbox: ratio={ratio:.4f} 內容區={content_w}x{content_h} / 640x640")
        print(f"    前處理後數值範圍: min={data.min():.3f} max={data.max():.3f} mean={data.mean():.3f}")

        input_data = np.expand_dims(data, axis=0).astype(np.float16)
        flat = ts.det_model.run(input_data)
        pred = flat.reshape(1, 640, 640, 1)[:, :, :, 0]

        p = pred[0][:content_h, :content_w]     # 只看內容區，padding 不算
        print(f"    === 原始機率圖 (內容區 {p.shape[1]}x{p.shape[0]}) ===")
        print(f"    min={p.min():.4f} max={p.max():.4f} mean={p.mean():.4f} "
              f"p99={np.percentile(p, 99):.4f} p999={np.percentile(p, 99.9):.4f}")

        over = (p > ts.det_thresh)
        print(f"    > det_thresh({ts.det_thresh}) 的像素: {int(over.sum())} / {p.size} "
              f"({100.0 * over.sum() / p.size:.2f}%)")

        if p.max() <= ts.det_thresh:
            print(f"    !! 整張機率圖最大值 {p.max():.4f} 還低於 det_thresh({ts.det_thresh})")
            print(f"    !! 這不是門檻太嚴，是模型根本沒有輸出文字區域 -> 轉檔有問題")
            continue

        # 逐一列出候選輪廓的分數（不套 box_thresh）
        mask = (p > ts.det_thresh).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(f"    findContours 找到 {len(contours)} 個輪廓")

        scores = []
        for c in contours[: ts.det_max_candidates]:
            points, sside = ts._get_mini_boxes(c)
            if sside < ts.min_size:
                scores.append((None, sside))
                continue
            pts = np.array(points)
            sc = ts._box_score_fast(p, pts.reshape(-1, 2))
            scores.append((sc, sside))

        valid = [(s, ss) for s, ss in scores if s is not None]
        too_small = len(scores) - len(valid)
        print(f"    其中 {too_small} 個因 min_size({ts.min_size}) 被丟掉，"
              f"{len(valid)} 個進入 box_thresh 評分")
        if valid:
            valid.sort(key=lambda x: -x[0])
            top = ", ".join(f"{s:.3f}(短邊{ss:.0f})" for s, ss in valid[:8])
            print(f"    分數由高到低: {top}")
            for th in (0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
                n = sum(1 for s, _ in valid if s >= th)
                mark = "  <== 目前設定" if abs(th - ts.det_box_thresh) < 1e-9 else ""
                print(f"      box_thresh={th:.1f} -> 通過 {n} 個{mark}")

        # 最後跑一次 production 的完整 run_det 對照
        boxes, elapsed = ts.run_det(img)
        print(f"    >>> production run_det 實際回傳 {len(boxes)} 個框 ({elapsed * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
