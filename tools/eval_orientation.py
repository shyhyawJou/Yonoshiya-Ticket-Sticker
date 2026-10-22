# -*- coding: utf-8 -*-
"""
eval_orientation.py
===================
量測 TextSystemDLA._check_boxes_need_rotate_90 的判準，並在同一批資料上
比較候選替代判準。

現行判準
--------
    vertical = #{box : h/w >= 1.4}
    需要轉 90 度  <=>  vertical / #box >= 0.2

問題：票據下半部的「小計/深夜料金/合計」金額區有一整排直式的「円」字，
加上單字框（箸、1），在**正立**的票據上就足以讓 vertical 比例超過 0.2，
於是把正立的票據誤轉 90 度。文字被轉成直式後，任何依「同一列」做品名↔數量
配對的後處理都會失效。

Ground truth 用 images/tool/orient_final.json（Claude 對 99 張逐張看圖驗證過
的轉正角度）。needs_quarter_turn = (該角度為 90 或 270)。

用法
----
    cd DLA_on_720
    py -3.12 tools/eval_orientation.py
"""

from __future__ import annotations

import json
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DLA_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(DLA_DIR)
for p in ([DLA_DIR] if sys.prefix != sys.base_prefix else [DLA_DIR, os.path.join(PROJECT_ROOT, "annotation_deps")]):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from tools.ocr_backend import OcrBackend  # noqa: E402

IMAGE_DIR = os.path.join(PROJECT_ROOT, "images", "order_ticket")
ORIENT_JSON = os.path.join(PROJECT_ROOT, "images", "tool", "orient_final.json")
CACHE = os.path.join(THIS_DIR, "_eval_cache_suzumo_profiles_v2", "det_original.json")


def box_stats(boxes):
    """回傳 (count_ratio, area_weighted_ratio, long_axis_ratio)。

    count_ratio        : 現行判準用的「直式框數 / 總框數」
    area_weighted_ratio: 以框面積加權的直式比例（大字權重高，抗小字雜訊）
    long_axis_ratio    : 所有框「長邊為垂直」的面積加權比例
    """
    n = len(boxes)
    if n == 0:
        return 0.0, 0.0, 0.0
    vc = 0
    a_vert = a_all = 0.0
    for b in boxes:
        b = np.asarray(b, dtype=np.float64)
        w = np.linalg.norm(b[0] - b[1])
        h = np.linalg.norm(b[0] - b[3])
        if w <= 0 or h <= 0:
            continue
        area = w * h
        a_all += area
        if h / w >= 1.4:
            vc += 1
            a_vert += area
    return (vc / n,
            (a_vert / a_all) if a_all > 0 else 0.0,
            (a_vert / a_all) if a_all > 0 else 0.0)


def main():
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    with open(ORIENT_JSON, "r", encoding="utf-8") as fh:
        verified = json.load(fh)

    names = sorted(n for n in os.listdir(IMAGE_DIR) if n.endswith(".jpg"))
    names = [n for n in names if n in verified]

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    todo = [n for n in names if n not in cache]
    if todo:
        backend = OcrBackend(prefer="auto", profile="suzumo")
        print(f"backend={backend.kind}; 對 {len(todo)} 張原圖跑 det ...")
        for i, n in enumerate(todo, 1):
            img = cv2.imread(os.path.join(IMAGE_DIR, n))
            boxes = backend._det(img)
            cache[n] = [np.asarray(b).tolist() for b in (boxes if boxes is not None else [])]
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}", flush=True)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh)

    rows = []
    for n in names:
        need = verified[n] in (90, 270)
        cr, ar, _ = box_stats(cache.get(n, []))
        rows.append((n, need, cr, ar, len(cache.get(n, []))))

    pos = [r for r in rows if r[1]]
    neg = [r for r in rows if not r[1]]
    print(f"\n樣本 {len(rows)} 張：需要轉 1/4 圈 {len(pos)} 張，不需要 {len(neg)} 張\n")

    def dist(label, idx):
        pv = sorted(r[idx] for r in pos)
        nv = sorted(r[idx] for r in neg)
        def q(v, p):
            return v[min(len(v) - 1, int(p * (len(v) - 1)))] if v else float("nan")
        print(f"{label}")
        print(f"    需要轉  : min={q(pv,0):.3f} p25={q(pv,.25):.3f} "
              f"median={q(pv,.5):.3f} p75={q(pv,.75):.3f} max={q(pv,1):.3f}")
        print(f"    不需要轉: min={q(nv,0):.3f} p25={q(nv,.25):.3f} "
              f"median={q(nv,.5):.3f} p75={q(nv,.75):.3f} max={q(nv,1):.3f}")

    dist("直式框「數量」比例 (現行判準用的量)", 2)
    dist("直式框「面積加權」比例", 3)

    print("\n各門檻的準確率：")
    print(f"{'判準':<26}{'門檻':>8}{'正確':>8}{'誤轉':>8}{'漏轉':>8}{'準確率':>10}")
    best = None
    for idx, label in ((2, "count_ratio"), (3, "area_ratio")):
        for thr in [0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
            fp = sum(1 for r in neg if r[idx] >= thr)
            fn = sum(1 for r in pos if r[idx] < thr)
            ok = len(rows) - fp - fn
            acc = 100.0 * ok / len(rows)
            mark = ""
            if label == "count_ratio" and abs(thr - 0.20) < 1e-9:
                mark = "  <-- 現行"
            print(f"{label:<26}{thr:>8.2f}{ok:>8}{fp:>8}{fn:>8}{acc:>9.1f}%{mark}")
            if best is None or acc > best[0]:
                best = (acc, label, thr, ok, fp, fn)
    print(f"\n最佳: {best[1]} >= {best[2]:.2f} -> {best[0]:.1f}% "
          f"(正確 {best[3]}, 誤轉 {best[4]}, 漏轉 {best[5]})")


if __name__ == "__main__":
    main()
