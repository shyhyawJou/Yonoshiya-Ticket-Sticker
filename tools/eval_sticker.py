# -*- coding: utf-8 -*-
"""
eval_sticker.py
===============
order_sticker 的端到端評估（det + cls + rec + 詞彙比對），比較舊 / 新比對邏輯。

貼紙這條路徑在開發機可以端到端量測，因為：
  * 每張貼紙只有一個詞，不需要行列配對
  * 本機 ONNX rec 對這三個詞的辨識足夠好（見輸出）
但**權重仍是 ONNX，不是 edge 的 DLA**，數字不可直接當 edge 實測。

GT 來自 images/order_sticker/*.json 的 description。
其中 200 張是 Claude 自動產生後逐張看圖複核、1 張是使用者原有的人工標註。

用法
----
    cd DLA_on_720
    py -3.12 tools/eval_sticker.py
    py -3.12 tools/eval_sticker.py --refresh
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DLA_DIR = os.path.dirname(THIS_DIR)
PROJECT_ROOT = os.path.dirname(DLA_DIR)
for p in ([DLA_DIR] if sys.prefix != sys.base_prefix else [DLA_DIR, os.path.join(PROJECT_ROOT, "annotation_deps")]):
    if p not in sys.path:
        sys.path.insert(0, p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from tools.ocr_backend import OcrBackend, load_vocab  # noqa: E402

IMAGE_DIR = os.path.join(PROJECT_ROOT, "images", "order_sticker")
VOCAB = os.path.join(PROJECT_ROOT, "images", "日文sticker.txt")
CACHE = os.path.join(THIS_DIR, "_eval_cache_suzumo_profiles_v2", "ocr_sticker.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--show-failures", type=int, default=15)
    args = ap.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    vocab = load_vocab(VOCAB)
    names = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))
    gt = {}
    for n in names:
        jp = os.path.join(IMAGE_DIR, n[:-4] + ".json")
        if not os.path.exists(jp):
            continue
        with open(jp, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        shapes = doc.get("shapes") or []
        if len(shapes) != 1:
            continue
        d = shapes[0].get("description")
        if isinstance(d, str) and d:
            gt[n] = d
    names = [n for n in names if n in gt]
    print(f"詞彙 = {vocab}")
    print(f"有 GT 的貼紙 {len(names)} 張  (GT 分佈: {dict(Counter(gt.values()))})")

    backend = OcrBackend(prefer="auto", profile="suzumo")
    print(f"OCR backend = {backend.kind}"
          + ("   ← 開發機替身，非 edge DLA 權重" if backend.kind == "onnx" else ""))

    cache = {}
    if os.path.exists(CACHE) and not args.refresh:
        with open(CACHE, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    todo = [n for n in names if n not in cache]
    if todo:
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        for i, n in enumerate(todo, 1):
            img = cv2.imread(os.path.join(IMAGE_DIR, n))
            out = backend.predict(img, "sticker") if img is not None else None
            cache[n] = ({"ok": False} if out is None else
                        {"ok": True, "rec": [[t, s] for t, s in out.rec],
                         "boxes": [np.asarray(b).tolist() for b in out.boxes],
                         "rot": out.rot_deg, "ms": out.elapsed_ms})
            if i % 40 == 0 or i == len(todo):
                print(f"  OCR {i}/{len(todo)}", flush=True)
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False)

    from logic.readers.sticker_matcher import StickerMatcher
    from logic.text_utils import FuzzyMatcher, normalize_text

    fm = FuzzyMatcher()
    new = StickerMatcher(menus_sticker=vocab, fuzzy_matcher=fm, vocab_sticker=vocab)
    # 舊行為：候選 = menus_mapping.yaml 的 sub_items（341 筆），門檻 score>60
    import re
    mm = open(os.path.join(DLA_DIR, "tasks", "ocr", "yoshinoya", "menus_mapping.yaml"),
              encoding="utf-8").read()
    legacy_vocab = sorted({normalize_text(x, ignore_space=True)
                           for x in re.findall(r'^\s+- "(.*?)"', mm, re.M)})
    legacy = StickerMatcher(menus_sticker=legacy_vocab, fuzzy_matcher=fm)

    ok_n = ok_l = 0
    unconf_n = unconf_l = 0
    wrong_n, wrong_l = [], []
    no_ocr = 0
    t_n = t_l = 0.0
    for n in names:
        e = cache.get(n, {})
        if not e.get("ok"):
            no_ocr += 1
            continue
        rec = [(t, float(s)) for t, s in e["rec"]]
        boxes = [np.asarray(b, dtype=float) for b in e["boxes"]]

        t0 = time.perf_counter()
        # 舊版沒有 read_text，用它原本的比對核心：fuzzy_matcher + score>60
        raw = legacy._pick_primary(rec, boxes)
        lname = None
        if raw:
            r = fm.match(raw[0], legacy_vocab)
            if r and r[1] > 60:
                lname = r[0]
        t1 = time.perf_counter()
        nname, _d = new.read_text(rec, boxes)
        t2 = time.perf_counter()
        t_l += (t1 - t0) * 1000
        t_n += (t2 - t1) * 1000

        if lname == gt[n]:
            ok_l += 1
        elif lname is None:
            unconf_l += 1
        else:
            wrong_l.append((n, gt[n], lname))
        if nname == gt[n]:
            ok_n += 1
        elif nname is None:
            unconf_n += 1
        else:
            wrong_n.append((n, gt[n], nname))

    tot = len(names) - no_ocr
    print(f"\n可評估 {tot} 張（OCR 無輸出 {no_ocr} 張）\n")
    print(f"{'指標':<26}{'舊版':>12}{'新版':>12}")
    print("-" * 60)
    print(f"{'正確 (張)':<26}{ok_l:>12}{ok_n:>12}   / {tot}")
    print(f"{'正確率 (%)':<26}{100.0*ok_l/max(1,tot):>12.1f}{100.0*ok_n/max(1,tot):>12.1f}")
    print(f"{'錯字 (張)':<26}{len(wrong_l):>12}{len(wrong_n):>12}")
    print(f"{'未確認 (張)':<26}{unconf_l:>12}{unconf_n:>12}")
    print(f"{'比對耗時 (ms/張)':<26}{t_l/max(1,tot):>12.3f}{t_n/max(1,tot):>12.3f}")
    print(f"\n舊版候選數={len(legacy_vocab)}  新版候選數={len(vocab)}")

    if wrong_n and args.show_failures:
        print(f"\n=== 新版錯字案例 (前 {args.show_failures} / 共 {len(wrong_n)}) ===")
        for n, g, p in wrong_n[:args.show_failures]:
            print(f"  images/order_sticker/{n}  GT={g}  預測={p}")
    if wrong_l and args.show_failures:
        print(f"\n=== 舊版錯字案例 (前 {args.show_failures} / 共 {len(wrong_l)}) ===")
        for n, g, p in wrong_l[:args.show_failures]:
            print(f"  images/order_sticker/{n}  GT={g}  預測={p}")


if __name__ == "__main__":
    main()
