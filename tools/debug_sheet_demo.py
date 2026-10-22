"""
debug_sheet_demo.py
===================
用合成資料渲染一張範例總覽圖，不需要相機 / NPU / 模型。
用途：調整版面、確認字型有裝、給不熟悉的人看「這張圖長怎樣」。

    python3 tools/debug_sheet_demo.py [輸出路徑.jpg]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from types import SimpleNamespace                                  # noqa: E402
from utils.debug_sheet import DebugContext, DebugSheet, Evidence   # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None


def _font(size):
    return ImageFont.truetype(str(ROOT / "tasks/ocr/NotoSansJP-Bold.otf"), size)


def fake_ticket():
    """合成傳票：白底黑字，回傳 (crop, dt_boxes, rec_res, rows)。"""
    W, H = 900, 1200
    img = Image.new("RGB", (W, H), (250, 250, 246))
    d = ImageDraw.Draw(img)
    lines = [("1", 0.0), ("牛丼･小 つゆだく", 2), ("ﾆﾝﾆｸ唐揚超特", 1), ("豚丼･並", 1), ("おしんこ", 1)]
    d.text((640, 40), "123", font=_font(64), fill=(0, 0, 0))
    d.text((620, 150), "数", font=_font(40), fill=(0, 0, 0))
    boxes, rec = [], []

    def add(x, y, text, size, conf):
        f = _font(size)
        w = int(f.getlength(text))
        d.text((x, y), text, font=f, fill=(10, 10, 10))
        boxes.append(np.array([[x - 4, y], [x + w + 4, y], [x + w + 4, y + size * 1.25], [x - 4, y + size * 1.25]], np.float32))
        rec.append((text, conf))

    add(640, 40, "123", 64, 0.99)
    add(620, 150, "数", 40, 0.96)
    y = 260
    for name, qty in [("牛丼ﾂﾕﾀﾞｸ小", 2), ("ﾆﾝﾆｸ唐揚超特", 1), ("豚丼並", 1), ("お新香", 1)]:
        add(60, y, name, 44, 0.97 if name != "お新香" else 0.62)
        add(640, y, str(qty), 44, 0.98)
        y += 120
    add(60, 900, "テーブNO 12", 36, 0.88)
    rows = [
        dict(raw="牛丼ﾂﾕﾀﾞｸ小", name="牛丼･小 つゆだく", score=93.0, qty=2, qty_src="獨立數字框", status="ok", box_idx=[2, 3]),
        dict(raw="ﾆﾝﾆｸ唐揚超特", name="ﾆﾝﾆｸ唐揚超特", score=100.0, qty=1, qty_src="獨立數字框", status="ok", box_idx=[4, 5]),
        dict(raw="豚丼並", name="豚丼･並", score=88.0, qty=1, qty_src="數量欄crop", status="ok", box_idx=[6, 7]),
        dict(raw="お新香", name=None, score=41.0, qty=None, qty_src="", status="low_score", box_idx=[8, 9]),
    ]
    return np.asarray(img)[:, :, ::-1].copy(), boxes, rec, rows


def fake_sticker(text, w=520, h=220, bg=(245, 245, 235), conf=0.95, tilt=0):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    f = _font(52)
    tw = int(f.getlength(text))
    x, y = (w - tw) // 2, 70
    d.text((x, y), text, font=f, fill=(15, 15, 15))
    box = np.array([[x - 6, y], [x + tw + 6, y], [x + tw + 6, y + 66], [x - 6, y + 66]], np.float32)
    return np.asarray(img)[:, :, ::-1].copy(), [box], [(text, conf)]


def build_entries():
    entries = []
    uid = 0

    def nxt():
        nonlocal uid
        uid += 1
        return uid

    crop, boxes, rec, rows = fake_ticket()
    ds = DebugSheet(SimpleNamespace(enabled=True), "/tmp", ROOT, DebugContext())
    ev = ds.record_ocr("single_order", "ticket", crop, boxes, rec, dict(
        order_number="123", rows=rows, accepted=True,
        outcome="主票確認 → TICKET_READY",
        combos=[{"牛丼小ﾂﾕﾀﾞｸ": 2, "ﾆﾝﾆｸから揚げ超特盛丼": 1, "追い飯": 1}],
        unconfirmed_qty=[],
    ))
    return ds, ev, entries


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/debug_sheet_demo.jpg")
    variants = {
        "牛丼･小 つゆだく": [["牛丼小ﾂﾕﾀﾞｸ"]],
        "ﾆﾝﾆｸ唐揚超特": [["ﾆﾝﾆｸから揚げ超特盛丼", "追い飯", "ﾆﾝﾆｸﾏｼﾏｼだれ", "玉子"],
                      ["ﾆﾝﾆｸから揚げ超特盛丼", "追い飯", "ﾆﾝﾆｸﾏｼﾏｼだれ", "半熟玉子"]],
        "豚丼･並": [["豚丼 並盛"]],
    }
    ctx = DebugContext(menu_variants=variants,
                       object_class_items={"sesame_front": "ごまドレッシング", "chopsticks": "箸"},
                       profile="yoshinoya")
    ds = DebugSheet(SimpleNamespace(enabled=True), "/tmp", ROOT, ctx,
                    crop_fn=lambda frame, xywhr: frame[0:120, 0:200])
    crop, boxes, rec, rows = fake_ticket()
    ds.record_ocr("t1", "ticket", crop, boxes, rec, dict(
        order_number="123", rows=rows, accepted=True, outcome="主票確認 → TICKET_READY",
        combos=[{"牛丼小ﾂﾕﾀﾞｸ": 2, "ﾆﾝﾆｸから揚げ超特盛丼": 1, "追い飯": 1, "ﾆﾝﾆｸﾏｼﾏｼだれ": 1, "玉子": 1}],
    ))

    class _Obj:   # 模擬 TrackedItem：可以掛屬性
        pass

    def sticker(text, verdict, matched, score, note, conf=0.95, status="MATCHED", reads=1, top=None, ticket_names=None):
        track = ds.track_uid(_Obj())
        for k in range(reads):
            c, b, r = fake_sticker(text, conf=conf)
            ds.record_ocr("t1", "sticker", c, b, r, dict(
                raw=text, conf=conf, matched=matched, reported=matched, status=status, score=score,
                exact=score >= 100, top=top or [], verdict=verdict if k == reads - 1 else "SUSPECT",
                note=note, expected=["牛丼小ﾂﾕﾀﾞｸ", "牛丼小ﾂﾕﾀﾞｸ", "豚丼 並盛"], checked=[],
                ticket_names=ticket_names or []), track=track)

    sticker("牛丼小ﾂﾕﾀﾞｸ", "OK", "牛丼小ﾂﾕﾀﾞｸ", 100, "核對成功 1/2", top=[("牛丼小ﾂﾕﾀﾞｸ", 100), ("牛丼並ﾂﾕﾀﾞｸ", 86), ("牛丼大ﾂﾕﾀﾞｸ", 86)], ticket_names=["牛丼･小 つゆだく"])
    sticker("牛丼小ﾂﾕﾀﾞｸ", "OK", "牛丼小ﾂﾕﾀﾞｸ", 100, "核對成功 2/2", top=[("牛丼小ﾂﾕﾀﾞｸ", 100)], ticket_names=["牛丼･小 つゆだく"])
    sticker("豚丼 特盛", "NG", "豚丼 特盛", 100, "錯誤品項成立: 豚丼 特盛（連續 2 次沒比對成功）", status="WRONG_ITEM", reads=2,
            top=[("豚丼 特盛", 100), ("豚丼 並盛", 77)])
    sticker("豚丼 並盛", "OK", "豚丼 並盛", 100, "核對成功", ticket_names=["豚丼･並"], conf=0.91)
    sticker("ﾆﾝﾆｸﾏｼ", "SUSPECT", "ﾆﾝﾆｸﾏｼﾏｼだれ", 74, "確認中 1/2: WRONG ﾆﾝﾆｸﾏｼﾏｼだれ", conf=0.58, status="WRONG_ITEM", top=[("ﾆﾝﾆｸﾏｼﾏｼだれ", 74)])
    sticker("", "UNREAD", None, 0, "沒讀到文字", conf=0.2, status="UNRECOGNIZED")
    for cls, item, v, note in (("sesame_front", "ごまドレッシング", "OK", "核對成功"), ("chopsticks", "箸", "NG", "錯誤品項成立: 箸（不在待核對清單）")):
        frame = np.full((600, 800, 3), (200, 190, 160), np.uint8)
        cv2.putText(frame, cls, (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (30, 30, 30), 4)
        ds.set_frame(frame)
        ds.record_known("t1", (100, 60, 200, 120, 0.0), dict(
            cls_name=cls, item_name=item, matched=item, reported=item, status="MATCHED" if v == "OK" else "WRONG_ITEM",
            verdict=v, note=note,
            expected=["牛丼小ﾂﾕﾀﾞｸ", "ごまドレッシング"], checked=[], ticket_names=[]), track=ds.track_uid(_Obj()))

    summary = dict(status="NGtoOK", tray_id="t1", order_number="123",
                   start_time="2026-10-22T12:03:11", end_time="2026-10-22T12:04:02", duration_sec=51,
                   expected=[{"牛丼小ﾂﾕﾀﾞｸ": 2}, {"豚丼 並盛": 1}, {"ごまドレッシング": 1}],
                   achieved=[{"牛丼小ﾂﾕﾀﾞｸ": 2}, {"豚丼 並盛": 1}, {"ごまドレッシング": 1}],
                   missing=[], wrong=[{"豚丼 特盛": 1}, {"箸": 1}], timed_out=False,
                   video_file="20261022_120311_event.mp4")
    entries = ds._evidence["t1"]
    t0 = time.perf_counter()
    img = ds.render_to_array(entries, summary)
    print(f"render {time.perf_counter() - t0:.2f}s  shape={img.shape}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    print("saved", out, f"{out.stat().st_size / 1e6:.2f}MB")


if __name__ == "__main__":
    main()
