# -*- coding: utf-8 -*-
"""
eval_parser_ideal.py
====================
把「後處理品質」與「辨識品質」分開量測。

做法
----
用 images/order_ticket/*.json 標註的**真實框位與真實文字**當作理想 OCR 輸入，
餵給舊 / 新 OrderParser，比較兩者的行列配對結果。

這樣量到的是：
    * 真實幾何（含 90/180/270 旋轉與紙張傾斜）下的分列是否正確
    * 「数量」欄閘門在真實座標下是否抓對
    * 票據編號擷取
    * 詞彙比對與拒絕邏輯

量不到的是：
    * 對 OCR 錯字的容忍度（由 unit_test/test_ticket_layout.py 以合成案例覆蓋）
    * 對「金額欄干擾」的抵抗力 —— 標註沒有標金額欄，所以這裡的輸入不含
      金額數字。這一項同樣由 unit_test 的合成案例覆蓋
      （test_amount_column_not_taken_as_quantity）。

座標處理
--------
標註是原圖座標，票據可能是 90/180/270。正式流程由
AsyncOCR._worker_loop 先把影像轉正再交給 parser，所以這裡也依
images/tool/orient_final.json 把標註框轉到同一個正立座標系，
以對齊正式流程的輸入條件。

用法
----
    cd DLA_on_720
    py -3.12 tools/eval_parser_ideal.py --split all
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from typing import List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DLA_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(DLA_DIR)
for p in ([DLA_DIR] if sys.prefix != sys.base_prefix else [DLA_DIR, os.path.join(PROJECT_ROOT, "annotation_deps")]):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from tools.eval_ticket_parsing import aggregate, load_ground_truth, score  # noqa: E402
from tools.ocr_backend import load_vocab  # noqa: E402

IMAGE_DIR = os.path.join(PROJECT_ROOT, "images", "order_ticket")
VOCAB_TICKET = os.path.join(PROJECT_ROOT, "images", "日文ticket.txt")
ORIENT_JSON = os.path.join(PROJECT_ROOT, "images", "tool", "orient_final.json")


def rotate_points(pts: np.ndarray, ccw_deg: int, w: int, h: int) -> np.ndarray:
    """原圖座標 -> 轉正後座標。ccw_deg 為 orient_final.json 的值。

    對應 cv2.rotate：
        90  = ROTATE_90_COUNTERCLOCKWISE  (x, y) -> (y,       W - x)
        180 = ROTATE_180                  (x, y) -> (W - x,   H - y)
        270 = ROTATE_90_CLOCKWISE         (x, y) -> (H - y,   x)
    """
    p = np.asarray(pts, dtype=float)
    if ccw_deg == 0:
        return p.copy()
    if ccw_deg == 90:
        return np.column_stack([p[:, 1], w - p[:, 0]])
    if ccw_deg == 180:
        return np.column_stack([w - p[:, 0], h - p[:, 1]])
    if ccw_deg == 270:
        return np.column_stack([h - p[:, 1], p[:, 0]])
    raise ValueError(ccw_deg)


def build_ideal_input(json_path: str, ccw_deg: int):
    """從標註產生 (boxes, rec, frame_shape)，座標已轉到正立系。"""
    with open(json_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    W, H = int(doc["imageWidth"]), int(doc["imageHeight"])
    boxes, rec = [], []
    for sh in doc["shapes"]:
        desc = sh.get("description")
        if not isinstance(desc, str) or desc == "":
            continue
        q = rotate_points(np.asarray(sh["points"], dtype=float), ccw_deg, W, H)
        boxes.append(q)
        rec.append((desc, 1.0))
    shape = (W, H) if ccw_deg in (0, 180) else (H, W)  # (w, h) after rotation
    return boxes, rec, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--show-failures", type=int, default=10)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    with open(ORIENT_JSON, "r", encoding="utf-8") as fh:
        orient = json.load(fh)

    all_names = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))
    dev = [n for n in all_names if n.startswith("20260903_")]
    names = {"dev": dev, "test": [n for n in all_names if n not in dev],
             "all": all_names}[args.split]

    from logic.order_parser import OrderParser
    from logic.order_parser_legacy_reference import LegacyOrderParser
    from logic.text_utils import FuzzyMatcher, normalize_text

    vocab = [normalize_text(w, ignore_space=True) for w in load_vocab(VOCAB_TICKET)]
    fm = FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2,
                      special_score_threshold=35, default_score_threshold=40)
    mm = open(os.path.join(DLA_DIR, "tasks", "ocr", "yoshinoya", "menus_mapping.yaml"),
              encoding="utf-8").read()
    legacy_menus = [normalize_text(x, ignore_space=True)
                    for x in re.findall(r'- name: "(.*?)"', mm)]

    legacy = LegacyOrderParser(
        menus_ticket=legacy_menus,
        menus_with_digits={m for m in legacy_menus if re.search(r"[1-9]", m)},
        fuzzy_matcher=fm, rec_pred=None)
    new = OrderParser(menus_ticket=legacy_menus, menus_with_digits=set(),
                      fuzzy_matcher=fm, rec_pred=None, vocab_ticket=vocab,
                      enable_qty_rec_fallback=False)

    res_l, res_n, fails = [], [], []
    t_l = t_n = 0.0
    used = 0
    for n in names:
        jp = os.path.join(IMAGE_DIR, n[:-4] + ".json")
        if not os.path.exists(jp):
            continue
        gt = load_ground_truth(jp)
        if gt is None or n not in orient:
            continue
        boxes, rec, (fw, fh) = build_ideal_input(jp, orient[n])
        frame = np.zeros((fh, fw, 3), dtype=np.uint8)  # parser 只在 fallback 用影像
        used += 1

        t0 = time.perf_counter()
        try:
            li, lno = legacy.parse_legacy(frame, boxes, rec, False)
        except Exception as exc:
            li, lno = [], None
            logger.error(f"legacy failed {n}: {exc}")
        t1 = time.perf_counter()
        ni, nno = new.parse(frame, boxes, rec, False)
        t2 = time.perf_counter()
        t_l += (t1 - t0) * 1000
        t_n += (t2 - t1) * 1000

        sl, sn = score(gt["items"], li), score(gt["items"], ni)
        sl["order_ok"] = (lno == gt["order_number"])
        sn["order_ok"] = (nno == gt["order_number"])
        res_l.append(sl)
        res_n.append(sn)
        if sn["pair_hit"] != sn["gt_n"] or not sn["order_ok"]:
            fails.append({"image": f"images/order_ticket/{n}",
                          "orient_ccw": orient[n],
                          "gt_order": gt["order_number"], "new_order": nno,
                          "gt": gt["items"], "new": ni, "legacy": li})

    A, B = aggregate(res_l), aggregate(res_n)
    print(f"理想 OCR 輸入（標註框位+標註文字），split={args.split}，樣本 {used} 張，"
          f"GT 品項 {A['gt_n']} 組\n")
    print(f"{'指標':<28}{'舊版':>12}{'新版':>12}")
    print("-" * 64)
    print(f"{'品名 recall (%)':<28}{A['name_recall_pct']:>12.1f}{B['name_recall_pct']:>12.1f}")
    print(f"{'品名+數量 完全正確 (%)':<28}{A['pair_exact_pct']:>12.1f}{B['pair_exact_pct']:>12.1f}")
    print(f"{'品名 命中組數':<28}{A['name_hit']:>12}{B['name_hit']:>12}")
    print(f"{'品名+數量 命中組數':<28}{A['pair_hit']:>12}{B['pair_hit']:>12}")
    print(f"{'數量錯誤組數':<28}{A['qty_wrong']:>12}{B['qty_wrong']:>12}")
    print(f"{'數量未確認組數':<28}{A['qty_unconfirmed']:>12}{B['qty_unconfirmed']:>12}")
    print(f"{'多出的品名(誤配)次數':<28}{A['spurious']:>12}{B['spurious']:>12}")
    lo = sum(1 for r in res_l if r["order_ok"])
    no_ = sum(1 for r in res_n if r["order_ok"])
    print(f"{'票號完全正確 (張)':<28}{lo:>12}{no_:>12}   / {used}")
    print(f"{'後處理耗時 (ms/張)':<28}{t_l/max(1,used):>12.2f}{t_n/max(1,used):>12.2f}")

    if fails and args.show_failures:
        print(f"\n=== 新版未完全正確 (顯示前 {args.show_failures} / 共 {len(fails)}) ===")
        for f in fails[:args.show_failures]:
            print(f"\n  {f['image']} (orient={f['orient_ccw']})")
            print(f"    GT   號={f['gt_order']}  {f['gt']}")
            print(f"    新版 號={f['new_order']}  {f['new']}")
    out = os.path.join(THIS_DIR, "_eval_cache", f"failures_ideal_{args.split}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(fails, fh, ensure_ascii=False, indent=1)
    print(f"\n失敗案例明細: {out}")


if __name__ == "__main__":
    main()
