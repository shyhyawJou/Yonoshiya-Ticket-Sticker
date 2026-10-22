# -*- coding: utf-8 -*-
"""
eval_ticket_parsing.py
======================
在同一批 OCR 輸入上比較「舊 OrderParser」與「新 OrderParser」。

用法
----
    cd DLA_on_720
    py -3.12 tools/eval_ticket_parsing.py --split dev
    py -3.12 tools/eval_ticket_parsing.py --split test
    py -3.12 tools/eval_ticket_parsing.py --split all --refresh   # 重跑 OCR

OCR 輸出會快取到 tools/_eval_cache_suzumo_profiles_v2/ocr_<backend>.json，之後的比較不必重跑推論，
兩個 parser 因此拿到**位元相同**的輸入，差異純粹來自後處理。

環境與量測單位
--------------
* 後端：開發機只能用 ONNX（AI.*.so 是 aarch64，見 ocr_backend.py 說明）。
  因此「文字辨識率」不代表 edge 的 DLA 表現；可以代表的是同一輸入下
  **後處理**（詞彙比對、行列配對、拒絕邏輯）的差異。
* 耗時：每張影像的後處理毫秒數（不含 OCR 推論），x86 CPU 單執行緒。
* 樣本：images/order_ticket 的 99 張。dev = 20260903_* (15 張，調規則用)，
  test = 其餘 84 張（20260909_* 與 Image_*，最終評估用）。

Ground truth 的來源與限制
-------------------------
GT 取自 images/order_ticket/*.json。**這些標註是 Claude 自動產生的**，
其中：品名與數量由 Claude 逐格放大判讀後確認、使用者另手動修正過部分框位與
2 處數量（見 images/annotation_review.md）。它**不是完全獨立的人工標註**，
所以這裡的數字要當「相對比較」看，不是絕對正確率。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

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

IMAGE_DIR = os.path.join(PROJECT_ROOT, "images", "order_ticket")
VOCAB_TICKET = os.path.join(PROJECT_ROOT, "images", "日文ticket.txt")
CACHE_DIR = os.path.join(THIS_DIR, "_eval_cache_suzumo_profiles_v2")

# Claude 先前驗證過的每張影像轉正角度（CCW 度數），用來檢查上游方向判定。
ORIENT_JSON = os.path.join(PROJECT_ROOT, "images", "tool", "orient_final.json")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> Optional[dict]:
    """從標註 JSON 取出 (票號, [(品名, 數量), ...])。

    標註的 shape 排列為：[0]=伝票No, [1..N]=品名(上->下),
    [N+1]='数量' 表頭, [N+2..2N+1]=數量(上->下)。
    """
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    shapes = doc.get("shapes", [])
    if len(shapes) < 4 or (len(shapes) - 2) % 2 != 0:
        return None
    n = (len(shapes) - 2) // 2
    order_no = shapes[0].get("description")
    names = [shapes[1 + i].get("description") for i in range(n)]
    qty_hdr = shapes[n + 1].get("description")
    qtys = [shapes[n + 2 + i].get("description") for i in range(n)]
    if qty_hdr != "数量":
        return None
    items = []
    for nm, q in zip(names, qtys):
        if not isinstance(nm, str) or not nm:
            return None
        try:
            items.append((nm, int(q)))
        except (TypeError, ValueError):
            items.append((nm, None))
    return {"order_number": order_no, "items": items,
            "width": doc.get("imageWidth"), "height": doc.get("imageHeight")}


# ---------------------------------------------------------------------------
# OCR cache
# ---------------------------------------------------------------------------

def build_ocr_cache(names: List[str], backend: OcrBackend, cache_path: str,
                    refresh: bool) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    if os.path.exists(cache_path) and not refresh:
        with open(cache_path, "r", encoding="utf-8") as fh:
            cache = json.load(fh)
    todo = [n for n in names if n not in cache]
    if todo:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        for i, n in enumerate(todo, 1):
            img = cv2.imread(os.path.join(IMAGE_DIR, n))
            if img is None:
                cache[n] = {"ok": False, "why": "unreadable"}
                continue
            out = backend.predict(img, "ticket")
            if out is None:
                cache[n] = {"ok": False, "why": "no_text"}
            else:
                cache[n] = {
                    "ok": True,
                    "rot_deg": out.rot_deg,
                    "boxes": [np.asarray(b).tolist() for b in out.boxes],
                    "rec": [[t, s] for t, s in out.rec],
                    "frame_shape": list(out.frame.shape[:2]),
                    "ocr_ms": out.elapsed_ms,
                }
            if i % 10 == 0 or i == len(todo):
                print(f"  OCR {i}/{len(todo)}", flush=True)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, ensure_ascii=False)
    return cache


def frame_from_cache(name: str, entry: dict) -> np.ndarray:
    """依快取的 rot_deg 重建轉正後的影像（parser 的 fallback 需要真影像）。"""
    img = cv2.imread(os.path.join(IMAGE_DIR, name))
    rot = entry.get("rot_deg", 0)
    if rot == 90:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    elif rot == 270:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(gt_items, pred_items) -> dict:
    """比較兩個 (name, qty) 序列。

    * name_recall     : GT 品名出現次數中被正確產出的比例（multiset）
    * pair_exact      : 完整 (品名, 數量) 組合正確的比例（multiset）
    * qty_wrong       : 產出了數量但與 GT 不符的組數
    * qty_unconfirmed : 產出品名但數量為 None 的組數
    * spurious        : 產出了 GT 沒有的品名的次數
    """
    gt_names = Counter(n for n, _ in gt_items)
    pred_names = Counter(n for n, _ in pred_items)
    name_hit = sum((gt_names & pred_names).values())

    gt_pairs = Counter(gt_items)
    pred_pairs = Counter(pred_items)
    pair_hit = sum((gt_pairs & pred_pairs).values())

    unconf = sum(1 for _, q in pred_items if q is None)
    spurious = sum((pred_names - gt_names).values())

    qty_wrong = 0
    remaining = Counter(gt_items)
    for nm, q in pred_items:
        if q is None:
            continue
        if remaining.get((nm, q), 0) > 0:
            remaining[(nm, q)] -= 1
        elif gt_names.get(nm, 0) > 0:
            qty_wrong += 1

    return {"gt_n": sum(gt_names.values()), "name_hit": name_hit,
            "pair_hit": pair_hit, "qty_wrong": qty_wrong,
            "qty_unconfirmed": unconf, "spurious": spurious,
            "pred_n": sum(pred_names.values())}


def aggregate(rows: List[dict]) -> dict:
    tot = {k: 0 for k in ("gt_n", "name_hit", "pair_hit", "qty_wrong",
                          "qty_unconfirmed", "spurious", "pred_n")}
    for r in rows:
        for k in tot:
            tot[k] += r[k]
    gt = max(1, tot["gt_n"])
    tot["name_recall_pct"] = 100.0 * tot["name_hit"] / gt
    tot["pair_exact_pct"] = 100.0 * tot["pair_hit"] / gt
    return tot


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--backend", choices=["auto", "dla", "onnx"], default="auto")
    ap.add_argument("--refresh", action="store_true", help="重跑 OCR 推論")
    ap.add_argument("--show-failures", type=int, default=12)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="ERROR")

    all_names = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".jpg"))
    dev = [n for n in all_names if n.startswith("20260903_")]
    test = [n for n in all_names if n not in dev]
    names = {"dev": dev, "test": test, "all": all_names}[args.split]

    gts = {}
    for n in names:
        jp = os.path.join(IMAGE_DIR, n[:-4] + ".json")
        if os.path.exists(jp):
            gt = load_ground_truth(jp)
            if gt:
                gts[n] = gt
    names = [n for n in names if n in gts]

    print(f"split={args.split}  有 GT 的樣本數={len(names)} "
          f"(dev={len(dev)}, test={len(test)}, all={len(all_names)})")

    backend = OcrBackend(prefer=args.backend, profile="suzumo")
    print(f"OCR backend = {backend.kind}"
          + ("   ← 開發機替身，非 edge 的 DLA 權重" if backend.kind == "onnx" else ""))
    cache = build_ocr_cache(names, backend,
                            os.path.join(CACHE_DIR, f"ocr_{backend.kind}.json"),
                            args.refresh)

    # --- orientation check against Claude 驗證過的角度 --------------------
    orient_acc = None
    if os.path.exists(ORIENT_JSON):
        with open(ORIENT_JSON, "r", encoding="utf-8") as fh:
            verified_ccw = json.load(fh)
        ok = bad = 0
        bad_list = []
        for n in names:
            e = cache.get(n, {})
            if not e.get("ok") or n not in verified_ccw:
                continue
            want_cw = (360 - verified_ccw[n]) % 360
            got = e["rot_deg"] % 360
            if got == want_cw:
                ok += 1
            else:
                bad += 1
                bad_list.append((n, want_cw, got))
        if ok + bad:
            orient_acc = (ok, bad, bad_list)

    # --- run both parsers -------------------------------------------------
    from logic.szm.order_parser import OrderParser
    from logic.szm.order_parser_legacy_reference import LegacyOrderParser
    from logic.text_utils import FuzzyMatcher, normalize_text

    vocab = [normalize_text(w, ignore_space=True) for w in load_vocab(VOCAB_TICKET)]
    fm = FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2,
                      special_score_threshold=35, default_score_threshold=40)

    # 舊版：候選 = menus_mapping.yaml 的吉野家菜名（現行 config 的實際行為）
    import re as _re
    mm = open(os.path.join(DLA_DIR, "tasks", "ocr", "yoshinoya", "menus_mapping.yaml"),
              encoding="utf-8").read()
    legacy_menus = [normalize_text(x, ignore_space=True)
                    for x in _re.findall(r'- name: "(.*?)"', mm)]
    legacy = LegacyOrderParser(menus_ticket=legacy_menus,
                               menus_with_digits={m for m in legacy_menus
                                                  if _re.search(r"[1-9]", m)},
                               fuzzy_matcher=fm, rec_pred=None)
    new = OrderParser(menus_ticket=legacy_menus,
                      menus_with_digits=set(), fuzzy_matcher=fm,
                      rec_pred=None,                 # 本機不做裁圖 fallback
                      vocab_ticket=vocab)

    res_legacy, res_new, fails, t_legacy, t_new = [], [], [], 0.0, 0.0
    no_ocr = 0
    for n in names:
        e = cache.get(n, {})
        if not e.get("ok"):
            no_ocr += 1
            continue
        boxes = [np.asarray(b, dtype=float) for b in e["boxes"]]
        rec = [(t, float(s)) for t, s in e["rec"]]
        frame = frame_from_cache(n, e)
        gt = gts[n]

        t0 = time.perf_counter()
        try:
            li, _lno = legacy.parse_legacy(frame, boxes, rec, False)
        except Exception as exc:
            li, _lno = [], None
            logger.error(f"legacy failed on {n}: {exc}")
        t1 = time.perf_counter()
        ni, nno = new.parse(frame, boxes, rec, False)
        t2 = time.perf_counter()
        t_legacy += (t1 - t0) * 1000
        t_new += (t2 - t1) * 1000

        sl, sn = score(gt["items"], li), score(gt["items"], ni)
        sl["name"], sn["name"] = n, n
        sl["order_ok"] = (_lno == gt["order_number"])
        sn["order_ok"] = (nno == gt["order_number"])
        res_legacy.append(sl)
        res_new.append(sn)
        if sn["pair_hit"] != sn["gt_n"]:
            fails.append({"image": os.path.join("images", "order_ticket", n),
                          "gt": gt["items"], "new": ni, "legacy": li,
                          "gt_order": gt["order_number"], "new_order": nno})

    A, B = aggregate(res_legacy), aggregate(res_new)
    nimg = len(res_legacy)
    print(f"\n可評估影像 {nimg} 張（OCR 無輸出 {no_ocr} 張），"
          f"GT 品項總數 {A['gt_n']} 組\n")
    hdr = f"{'指標':<28}{'舊版':>12}{'新版':>12}"
    print(hdr); print("-" * len(hdr.encode('utf-8')) )
    def row(label, a, b, suffix=""):
        print(f"{label:<28}{a:>12}{b:>12}" if isinstance(a, str)
              else f"{label:<28}{a:>12.1f}{b:>12.1f}{suffix}")
    row("品名 recall (%)", A["name_recall_pct"], B["name_recall_pct"])
    row("品名+數量 完全正確 (%)", A["pair_exact_pct"], B["pair_exact_pct"])
    print(f"{'品名 命中組數':<28}{A['name_hit']:>12}{B['name_hit']:>12}")
    print(f"{'品名+數量 命中組數':<28}{A['pair_hit']:>12}{B['pair_hit']:>12}")
    print(f"{'數量錯誤組數':<28}{A['qty_wrong']:>12}{B['qty_wrong']:>12}")
    print(f"{'數量未確認組數':<28}{A['qty_unconfirmed']:>12}{B['qty_unconfirmed']:>12}")
    print(f"{'多出的品名(誤配)次數':<28}{A['spurious']:>12}{B['spurious']:>12}")
    print(f"{'產出品項總組數':<28}{A['pred_n']:>12}{B['pred_n']:>12}")
    lo = sum(1 for r in res_legacy if r["order_ok"])
    no_ = sum(1 for r in res_new if r["order_ok"])
    print(f"{'票號完全正確 (張)':<28}{lo:>12}{no_:>12}   / {nimg}")
    print(f"{'後處理耗時 (ms/張)':<28}{t_legacy/max(1,nimg):>12.2f}{t_new/max(1,nimg):>12.2f}")

    if orient_acc:
        ok, bad, bad_list = orient_acc
        print(f"\n[上游方向判定] 與 Claude 驗證過的角度相符 {ok}/{ok+bad} "
              f"({100.0*ok/(ok+bad):.1f}%)")
        for n, want, got in bad_list[:8]:
            print(f"    {n}  應轉 {want}° 實際轉 {got}°")
        if len(bad_list) > 8:
            print(f"    ...另有 {len(bad_list)-8} 張")

    if fails and args.show_failures:
        print(f"\n=== 新版未完全正確的案例 (顯示前 {args.show_failures} / 共 {len(fails)}) ===")
        for f in fails[:args.show_failures]:
            print(f"\n  {f['image']}")
            print(f"    GT   號={f['gt_order']}  {f['gt']}")
            print(f"    新版 號={f['new_order']}  {f['new']}")
    out_path = os.path.join(CACHE_DIR, f"failures_{args.split}.json")
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fails, fh, ensure_ascii=False, indent=1)
    print(f"\n失敗案例明細: {out_path}")


if __name__ == "__main__":
    main()
