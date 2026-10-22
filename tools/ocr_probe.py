# -*- coding: utf-8 -*-
"""
ocr_probe.py
============
對單張（或一整個資料夾的）已裁切影像跑完整 OCR 流程並印出結果。

    ticket  -> 印出 票據編號 + 每個品名及其對應數量
    sticker -> 印出 貼紙上的日文

用法
----
    cd DLA_on_720

    # 單張收據
    py -3.12 tools/ocr_probe.py --type ticket  ../images/order_ticket/xxx.jpg

    # 單張貼紙
    py -3.12 tools/ocr_probe.py --type sticker ../images/order_sticker/xxx.jpg

    # 一整個資料夾（可加 --limit N）
    py -3.12 tools/ocr_probe.py --type ticket ../images/order_ticket --limit 5

    # 指定後端（edge 上用 dla；開發機只有 onnx）
    py -3.12 tools/ocr_probe.py --type ticket xxx.jpg --backend dla

    # 額外資訊：原始 OCR 文字、比對分數、未確認原因
    py -3.12 tools/ocr_probe.py --type ticket xxx.jpg --verbose

    # 機器可讀
    py -3.12 tools/ocr_probe.py --type ticket xxx.jpg --json

輸入必須是**已經裁切好的** order_ticket / order_sticker 影像
（也就是 MMRotate 之後的產物）。本工具不做物件偵測。

數量為「未確認」時會印成 ?，代表讀不到，**不是 1**。
"""

from __future__ import annotations

import argparse
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
from loguru import logger  # noqa: E402

from tools.ocr_backend import OcrBackend, load_vocab  # noqa: E402

def make_ticket_parser(vocab, cfg=None):
    if cfg is not None and cfg.ticket_parser == 'legacy':
        from logic.yoshinoya.order_parser import OrderParser
        from logic.text_utils import FuzzyMatcher
        return OrderParser(menus_ticket=list(vocab),
                           menus_with_digits={n for n in vocab if any(c.isdigit() for c in n)},
                           fuzzy_matcher=FuzzyMatcher(), rec_pred=None)
    from logic.szm.order_parser import OrderParser
    from logic.text_utils import FuzzyMatcher
    fm = FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2,
                      special_score_threshold=35, default_score_threshold=40)
    # rec_pred=None：本工具不載入 StandaloneRec，所以關掉裁圖數量 fallback。
    # 正式流程（logic_engine）會提供 rec_pred，該 fallback 會啟用。
    return OrderParser(menus_ticket=list(vocab), menus_with_digits=set(),
                       fuzzy_matcher=fm, rec_pred=None,
                       vocab_ticket=list(vocab),
                       enable_qty_rec_fallback=False)


def make_sticker_matcher(vocab):
    from logic.readers.sticker_matcher import StickerMatcher
    from logic.text_utils import FuzzyMatcher
    fm = FuzzyMatcher()
    return StickerMatcher(menus_sticker=list(vocab), fuzzy_matcher=fm,
                          vocab_sticker=list(vocab))


def run_ticket(parser, out, verbose):
    if not hasattr(parser, 'parse_detailed'):
        items, order_no = parser.parse(out.frame, out.boxes, out.rec, False)
        return {'order_number': order_no,
                'items': [{'name': n, 'quantity': q, 'source': 'legacy', 'raw': n} for n, q in items],
                'unrecognized_rows': [], 'qty_column_found': None,
                'raw_ocr': [t for t, _ in out.rec] if verbose else None}
    rows, order_no = parser.parse_detailed(out.frame, out.boxes, out.rec, False)
    items = [{"name": r.name, "quantity": r.quantity,
              "source": r.quantity_source, "raw": r.name_raw}
             for r in rows if r.name is not None]
    unrec = [{"raw": r.name_raw, "score": round(r.name_score, 1),
              "margin": round(r.name_margin, 1), "reason": r.name_reason}
             for r in rows if r.name is None]
    return {
        "order_number": order_no,
        "items": items,
        "unrecognized_rows": unrec,
        "qty_column_found": parser.last_qty_column is not None,
        "raw_ocr": [t for t, _ in out.rec] if verbose else None,
    }


def run_sticker(matcher, out, verbose):
    name, decision = matcher.read_text(out.rec, out.boxes)
    return {
        "text": name,
        "raw": None if decision is None else decision.raw,
        "score": None if decision is None else round(decision.score, 1),
        "margin": None if decision is None else round(decision.margin, 1),
        "reason": None if decision is None else decision.reason,
        "raw_ocr": [t for t, _ in out.rec] if verbose else None,
    }


def print_ticket(path, res, out, verbose):
    print(f"\n=== {os.path.basename(path)}  [ticket]  (轉正 {out.rot_deg}°, "
          f"{len(out.boxes)} 個文字框, OCR {out.elapsed_ms:.0f}ms, {out.backend}) ===")
    print(f"票據號碼: {res['order_number'] if res['order_number'] else '未確認'}")
    if res["qty_column_found"] is False:
        print("  ⚠ 找不到「数量」欄表頭 -> 本張所有數量標為未確認")
    if not res["items"]:
        print("  (沒有辨識出任何品名)")
    else:
        w = max(len(i["name"]) for i in res["items"])
        print(f"  {'品名'.ljust(w)}  數量")
        for i in res["items"]:
            q = "?" if i["quantity"] is None else str(i["quantity"])
            line = f"  {i['name'].ljust(w)}  {q}"
            if verbose:
                line += f"   [來源={i['source']}, 原始OCR='{i['raw']}']"
            print(line)
    unconf = [i["name"] for i in res["items"] if i["quantity"] is None]
    if unconf:
        print(f"  ⚠ 數量未確認: {unconf}   (? 表示讀不到，不是 1)")
    if res["unrecognized_rows"]:
        print(f"  ⚠ 有 {len(res['unrecognized_rows'])} 列無法比對到詞彙"
              + (":" if verbose else ""))
        if verbose:
            for u in res["unrecognized_rows"]:
                print(f"      原始OCR='{u['raw']}' score={u['score']} "
                      f"margin={u['margin']} ({u['reason']})")
    if verbose and res["raw_ocr"] is not None:
        print(f"  原始 OCR 全部文字: {res['raw_ocr']}")


def print_sticker(path, res, out, verbose):
    print(f"\n=== {os.path.basename(path)}  [sticker]  (轉正 {out.rot_deg}°, "
          f"{len(out.boxes)} 個文字框, OCR {out.elapsed_ms:.0f}ms, {out.backend}) ===")
    if res["text"]:
        print(f"日文: {res['text']}")
    else:
        print("日文: 未確認  (無法可靠辨識，依規則不猜測)")
    if verbose:
        print(f"  原始OCR='{res['raw']}' score={res['score']} "
              f"margin={res['margin']} ({res['reason']})")
        if res["raw_ocr"] is not None:
            print(f"  原始 OCR 全部文字: {res['raw_ocr']}")
    print("  (貼紙沒有數量欄位，因此不輸出數量)")


def main():
    ap = argparse.ArgumentParser(
        description="對已裁切的 order_ticket / order_sticker 影像跑 OCR 並印出結果")
    ap.add_argument("path", help="影像檔或資料夾")
    ap.add_argument("--type", required=True, choices=["ticket", "sticker"])
    ap.add_argument("--profile", choices=["yoshinoya", "suzumo"], help="同 app 的品牌選擇；本工具固定測 DET")
    ap.add_argument("--backend", default="auto", choices=["auto", "dla", "onnx"])
    ap.add_argument("--limit", type=int, default=0, help="資料夾模式最多處理幾張")
    ap.add_argument("--verbose", action="store_true", help="顯示原始 OCR 與比對分數")
    ap.add_argument("--json", action="store_true", help="輸出 JSON")
    ap.add_argument("--log", default="ERROR", help="loguru 等級 (預設 ERROR)")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=args.log.upper())

    if os.path.isdir(args.path):
        files = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.limit:
            files = files[:args.limit]
    else:
        files = [args.path]
    if not files:
        print("找不到影像", file=sys.stderr)
        return 2

    backend = OcrBackend(prefer=args.backend, profile=args.profile)
    if not args.json:
        note = ("   ← 開發機替身；edge 請用 --backend dla"
                if backend.kind == "onnx" else "")
        print(f"OCR backend = {backend.kind}, profile = {backend.profile}, route = det{note}")

    cfg = backend.cfg
    vocab = ((cfg.vocab_ticket or cfg.menus_ticket) if args.type == 'ticket'
             else (cfg.vocab_sticker or cfg.menus_sticker))
    if not args.json:
        print(f"詞彙 ({args.type}) = {vocab}")
    worker = (make_ticket_parser(vocab, cfg) if args.type == "ticket"
              else make_sticker_matcher(vocab))

    results = []
    for path in files:
        img = cv2.imread(path)
        if img is None:
            rec = {"image": path, "error": "unreadable"}
            results.append(rec)
            if not args.json:
                print(f"\n=== {os.path.basename(path)} === 無法讀取")
            continue
        out = backend.predict(img, args.type)
        if out is None:
            rec = {"image": path, "error": "no_text_detected"}
            results.append(rec)
            if not args.json:
                print(f"\n=== {os.path.basename(path)} === 偵測不到任何文字")
            continue
        if args.type == "ticket":
            res = run_ticket(worker, out, args.verbose)
            if not args.json:
                print_ticket(path, res, out, args.verbose)
        else:
            res = run_sticker(worker, out, args.verbose)
            if not args.json:
                print_sticker(path, res, out, args.verbose)
        res.update({"image": path, "type": args.type, "rot_deg": out.rot_deg,
                    "ocr_ms": round(out.elapsed_ms, 1), "backend": out.backend})
        results.append(res)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
