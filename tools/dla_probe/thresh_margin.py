# -*- coding: utf-8 -*-
"""
thresh_margin.py
================
量測 orient_retry_conf_thresh 的安全餘裕。

對每張裁切圖，直接呼叫 _recognize 分別在「原方向」與「轉 90 度」讀一次
（繞過 predict 的觸發判斷），比較兩邊的平均 rec 信心，然後把資料分成兩群：

    需要轉 90 度的 (轉正後信心較高)  -> 看它們「原方向」的信心最高到多少
    不需要轉的     (原方向信心較高)  -> 看它們「原方向」的信心最低到多少

這兩個數字之間就是門檻可以安全落下的區間，做法跟 logic/orientation.py
當初決定 0.35 的方式一致。
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ocr_engine import TextSystemDLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="tasks/ocr/yoshi_det_v01_03.dla")
    ap.add_argument("--cls", default="tasks/ocr/cls_v2.dla")
    ap.add_argument("--rec", default="tasks/ocr/yoshi_rec_v01_05.dla")
    ap.add_argument("--dict", default="tasks/ocr/yoshinoya/japan_dict.txt")
    ap.add_argument("--type", default="sticker")
    ap.add_argument("images", nargs="+")
    args = ap.parse_args()

    files = []
    for p in args.images:
        files += sorted(glob.glob(os.path.join(p, "*.jpg"))) if os.path.isdir(p) else [p]

    ts = TextSystemDLA(args.det, args.cls, args.rec, args.dict)

    need_turn = []      # (原方向 mean conf, 轉90 mean conf, 檔名, 原文字, 轉後文字)
    no_turn = []
    skipped_vertical = 0
    skipped_not_ambiguous = 0

    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue

        boxes, _ = ts.run_det(img)
        if len(boxes) == 0:
            continue

        # 幾何已經判定要轉的，不走這條補救路徑，不列入統計
        if ts._check_boxes_need_rotate_90(boxes):
            skipped_vertical += 1
            continue
        if not ts._orientation_is_ambiguous(boxes):
            skipped_not_ambiguous += 1
            continue

        rec0, _, _ = ts._recognize(img, boxes, args.type)
        c0 = ts._mean_rec_conf(rec0)

        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rboxes, _ = ts.run_det(rot)
        if len(rboxes) == 0:
            c90, rec90 = 0.0, []
        else:
            rec90, _, _ = ts._recognize(rot, rboxes, args.type)
            c90 = ts._mean_rec_conf(rec90)

        t0 = " ".join(t for t, _ in rec0)
        t90 = " ".join(t for t, _ in rec90)
        entry = (c0, c90, os.path.basename(path), t0, t90)
        (need_turn if c90 > c0 else no_turn).append(entry)

    print(f"\n=== 正方形框(方向不明)的樣本 {len(need_turn) + len(no_turn)} 張 ===")
    print(f"(幾何已判定要轉、不走補救的 {skipped_vertical} 張；框不是正方形的 {skipped_not_ambiguous} 張，均未列入)")

    def stats(name, rows, idx):
        if not rows:
            print(f"\n{name}: 0 張")
            return None
        vals = np.array([r[idx] for r in rows])
        print(f"\n{name}: {len(rows)} 張")
        print(f"    原方向 mean conf: min={vals.min():.3f} max={vals.max():.3f} "
              f"平均={vals.mean():.3f} 中位數={np.median(vals):.3f}")
        print(f"    p90={np.percentile(vals, 90):.3f} p95={np.percentile(vals, 95):.3f} "
              f"p99={np.percentile(vals, 99):.3f}")
        return vals

    v_need = stats("A) 轉 90 度後比較好 (= 真的需要補救)", need_turn, 0)
    v_no = stats("B) 原方向就是對的 (= 不該補救)", no_turn, 0)

    if v_need is not None and v_no is not None:
        print(f"\n--- 門檻安全區 ---")
        print(f"    A 群原方向信心最高 = {v_need.max():.3f}")
        print(f"    B 群原方向信心最低 = {v_no.min():.3f}")
        if v_no.min() > v_need.max():
            print(f"    -> 安全區 {v_need.max():.3f} ~ {v_no.min():.3f}"
                  f"，中點 {(v_need.max() + v_no.min()) / 2:.3f}")
        else:
            print(f"    -> 兩群重疊，無法用單一門檻完全分開")
            print(f"       (重疊區間 {v_no.min():.3f} ~ {v_need.max():.3f})")

        for thr in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01):
            missed = int((v_need >= thr).sum())        # 該補救卻沒觸發
            wasted = int((v_no < thr).sum())           # 白跑一次(但結果不會變差)
            print(f"    門檻={thr:<5} 漏補救={missed:<4} 多跑一次={wasted}")

    if need_turn:
        worst = sorted(need_turn, key=lambda r: -r[0])[:5]
        print(f"\n--- A 群裡原方向信心最高的 5 張(最接近漏掉的邊緣) ---")
        for c0, c90, name, t0, t90 in worst:
            print(f"    {name}  原={c0:.3f}「{t0}」 -> 轉90={c90:.3f}「{t90}」")


if __name__ == "__main__":
    main()
