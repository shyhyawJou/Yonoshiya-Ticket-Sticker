"""吉野家大傳票：CV 輔助（ticket_cv_assist）與現行流程的離線對照。

只能在 AI box 上跑（要 DLA）。不動 app、不寫 config。兩條路都照 AsyncOCR._worker_loop 的步驟走：
  現行    det+cls+rec（含翻轉比對 / 轉 180 重讀）-> 轉正 -> legacy 解析
  CV輔助  CV 版面先決定方向 -> 方向可信就只跑一次 det、一輪 rec（predict_oriented），不可信照舊
          -> 轉正 -> 版面交給解析器（hand_off）-> CV 輔助解析
耗時是 OCR + 解析的合計（不含 app 主迴圈排隊）。

輸出到 --out：每張傳票一張對照圖（左：CV 版面 + det 框，橘色 = CV 補救的格子；
右：CV 輔助的逐列結果與兩種解析），加一份 summary.csv。

    cd /home/root/ocr_mmr_dla
    python3 tools/ticket_cv_probe.py --out /mnt/reserved/record/stream/special_test/cv_probe "/tmp/cvtk/crops/*.png"
    python3 tools/ticket_cv_probe.py --frames --out ... "/mnt/reserved/record/stream/special_test/*.jpg"
                                    # 整張畫面：先用 mmr 切出 order_ticket
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config import load_config  # noqa: E402
from logic.text_utils import FuzzyMatcher, normalize_text  # noqa: E402
from logic.yoshinoya.order_parser import OrderParser  # noqa: E402
from logic.yoshinoya.ticket_cv_layout import hand_off, rebase  # noqa: E402
from ocr_engine import AsyncOCR, StandaloneRecDLA, TextSystemDLA  # noqa: E402

FONT = os.path.join(ROOT, "tasks/ocr/NotoSansJP-Bold.otf")


def load_parts(cfg_path: str):
    cfg = load_config(cfg_path)
    m = cfg.runtime.model
    ocr = TextSystemDLA(m.ocr_det, m.ocr_cls, m.ocr_rec, m.text)
    rec_pred = StandaloneRecDLA(rec_path=m.ocr_rec, dict_path=m.text)
    menus = [normalize_text(x, ignore_space=True) for x in cfg.menus_ticket]
    fm = FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2,
                      special_score_threshold=35, default_score_threshold=40)
    kw = dict(menus_ticket=menus, menus_with_digits={x for x in menus if re.search(r"[1-9]", x)},
              fuzzy_matcher=fm, rec_pred=rec_pred, menu_index=getattr(cfg, "ticket_index", None))
    return cfg, ocr, OrderParser(**kw), OrderParser(cv_assist=True, **kw)


_ROT = {(True, True): (cv2.ROTATE_90_COUNTERCLOCKWISE, 270), (True, False): (cv2.ROTATE_90_CLOCKWISE, 90),
        (False, True): (cv2.ROTATE_180, 180)}


def run_ocr(ocr, crop, cv_orient: bool):
    """跟 AsyncOCR._worker_loop 一樣：(CV 定向) det -> rec，依方向把 crop 轉正。回傳 (crop, 框, 字, 路徑)。"""
    h0, w0 = crop.shape[:2]
    layout, note = (AsyncOCR._cv_ticket_layout(crop) if cv_orient else (None, None))
    if layout is not None and layout.ok and layout.orientation_ok:
        rec_res, dt_boxes, is_flip, is_rot90, _ = ocr.predict_oriented(crop, layout.rotate, obj_type="ticket")
    else:
        rec_res, dt_boxes, is_flip, is_rot90, _ = ocr.predict(crop, obj_type="ticket")
    path = ([note] if note else []) + list(ocr.prof["path"])
    code, applied = _ROT.get((bool(is_rot90), bool(is_flip)), (None, 0))
    if code is not None:
        crop = cv2.rotate(crop, code)
    if layout is not None:
        hand_off(crop, rebase(layout, applied, w0, h0))
    return crop, dt_boxes, rec_res, "+".join(path) or "-"


def fmt(items):
    return ", ".join(f"{n}×{q}" for n, q in items)


def render(path, crop, dt_boxes, cv_res, rows, results):
    s = 2
    if cv_res is not None:
        L = cv_res.layout
        left = cv2.resize(L.image, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        M = np.asarray(L.matrix)
        for i, b in enumerate(dt_boxes):  # det 框（灰）換到 CV 影像座標
            p = (np.asarray(b, np.float64).reshape(-1, 2) @ M[:, :2].T + M[:, 2]) * s
            cv2.polylines(left, [p.astype(np.int32)], True, (150, 150, 150), 1)
            cv2.putText(left, str(i), (int(p[0][0]), int(p[0][1]) - 2), 0, 0.4, (120, 120, 120), 1)
        for i in range(4):   # 欄線可能是斜的（長傳票 + 透視）
            cv2.line(left, (int(L.col_x(i, L.y_top) * s), int(L.y_top * s)),
                     (int(L.col_x(i, L.y_bottom) * s), int(L.y_bottom * s)), (255, 200, 0), 1)
        rescued = set(cv_res.rescued)
        for n, r in enumerate(L.rows, 1):
            for q, key in ((r.name_quad, f"第{n}列品名"), (r.qty_quad, f"第{n}列數量")):
                if q is not None:
                    hit = key in rescued
                    cv2.polylines(left, [(q * s).astype(np.int32)], True, (0, 140, 255) if hit else (255, 0, 0),
                                  2 if hit else 1)
            if r.check_box is not None:
                x0, y0, x1, y1 = r.check_box
                cv2.rectangle(left, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 0, 255), 2)
            cv2.putText(left, str(n), (2, int((r.y0 + r.y1) / 2 * s) + 5), 0, 0.5,
                        (0, 160, 0) if r.kind == "item" else (140, 140, 140), 1)
    else:
        left = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

    W, H = 860, max(left.shape[0], 700)
    panel = Image.new("RGB", (W, H), (250, 250, 250))
    dr = ImageDraw.Draw(panel)
    f, fs = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 16)
    y = 8
    dr.text((8, y), os.path.basename(path) + ("" if cv_res else "   CV 找不到表格 → 原本的解析"), font=f, fill=(0, 0, 0))
    y += 34
    for n, d in enumerate(rows, 1):
        if d["status"] == "段落標題":
            txt, col = f"{n:>2} 標題 「{d['raw']}」", (120, 120, 120)
        else:
            txt = f"{n:>2} 「{d['raw']}」→ {d['name']} {d['score']:.0f}  ×{d['qty']} [{d['qty_src']}]"
            col = (0, 110, 0) if d["status"] == "ok" else (200, 0, 0)
        dr.text((8, y), txt, font=fs, fill=col)
        y += 24
    y += 10
    for label, color in (("CV輔助", (0, 0, 160)), ("現行", (160, 80, 0))):
        order, items, ms = results[label]
        dr.text((8, y), f"{label}: 單號 {order}   ({ms:.0f} ms)", font=f, fill=color); y += 28
        dr.text((24, y), fmt(items) or "（無）", font=fs, fill=color); y += 30

    left_full = np.full((H, left.shape[1], 3), 255, np.uint8)
    left_full[:left.shape[0]] = left
    return np.hstack([left_full, cv2.cvtColor(np.array(panel), cv2.COLOR_RGB2BGR)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", action="store_true", help="輸入是整張畫面，先用 mmr 切出 order_ticket")
    ap.add_argument("--config", default="tasks/ocr/yoshinoya/config.yaml")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg, ocr, legacy, assisted = load_parts(args.config)
    mmr = None
    if args.frames:
        from mmr_engine import Rotated_RTMDET
        mmr = Rotated_RTMDET(path=cfg.runtime.model.object_det, classes=[c.name for c in cfg.classes],
                             conf_thresh=0.5, iou_thresh=0.4)

    summary = []
    for p in sorted(p for g in args.inputs for p in glob.glob(g)):
        img = cv2.imread(p)
        crops = [img]
        if mmr is not None:
            _, dets = mmr.detect(img, False)
            crops = []
            for d in sorted([d for d in dets if d.cls_name == "order_ticket"], key=lambda d: -d.conf)[:1]:
                cx, cy, w, h, r = d.xywhr
                crops.append(mmr.crop_by_angle(img, cx, cy, w, h, r)[0])
        for crop0 in crops:
            results, paths = {}, {}
            for label, parser, cv in (("現行", legacy, False), ("CV輔助", assisted, True)):
                t = time.perf_counter()
                crop, dt_boxes, rec_res, paths[label] = run_ocr(ocr, crop0.copy(), cv)
                items, order = parser.parse(crop, [b.copy() for b in dt_boxes], rec_res, False)
                results[label] = (order, items, 1000 * (time.perf_counter() - t))
            sheet = render(p, crop, dt_boxes, assisted.last_cv_result, assisted.last_debug_rows, results)
            name = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(args.out, name + ".jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            res = assisted.last_cv_result
            summary.append({"file": name, "cv_ok": res is not None,
                            "rescued": "、".join(res.rescued) if res else "",
                            **{f"{k}_order": v[0] for k, v in results.items()},
                            **{f"{k}_items": fmt(v[1]) for k, v in results.items()},
                            **{f"{k}_ms": round(v[2]) for k, v in results.items()}})
            summary[-1].update({f"{k}_path": v for k, v in paths.items()})
            print(name, " || ".join(f"{k} {o} [{fmt(it)}] {ms:.0f}ms" for k, (o, it, ms) in results.items()),
                  "|| 補救:", summary[-1]["rescued"] or "-", "|| 路徑:", " / ".join(paths.values()))

    if summary:
        with open(os.path.join(args.out, "summary.csv"), "w", encoding="utf-8-sig", newline="") as fp:
            w = csv.DictWriter(fp, fieldnames=list(summary[0].keys()))
            w.writeheader(); w.writerows(summary)


if __name__ == "__main__":
    main()
