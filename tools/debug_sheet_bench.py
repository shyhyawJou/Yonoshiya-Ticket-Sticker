"""
debug_sheet_bench.py
====================
量測「除錯總覽圖」到底多耗時。用接近真實的尺寸跑一整筆訂單：
傳票 1400x2000（30 個 det 框）、N 張貼紙、M 個醬料包、4024x3036 原圖，
走真實的 TrayStateMachine -> DebugSheet，最後印出跟正式執行時一樣的 [debug_sheet][timing] 摘要。

**在邊緣機上跑這支，就能拿到 ARM 的真實數字**（不需要相機 / NPU / 模型）：

    python3 tools/debug_sheet_bench.py            # 預設 10 張貼紙、3 個醬料包、重複 5 筆訂單
    python3 tools/debug_sheet_bench.py --stickers 20 --orders 10

看兩個數字：
  * 「主迴圈」= 每筆訂單在辨識主迴圈多花的時間（會直接拖慢辨識），越小越好。
  * 「背景」   = 獨立 thread 的渲染 + 編碼 + 寫檔；只要小於訂單的間隔就不會堆積。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unit_test"))

import test_debug_sheet as t                     # noqa: E402
from utils.debug_sheet import DebugContext, DebugSheet   # noqa: E402


def one_order(ds, args, rng):
    sm, _ = t.make_sm(Path(args.out), ds)
    tray = t.make_tray()
    trays = {"t1": tray}

    # 傳票：1400x2000、30 個框
    crop = (rng.random((2000, 1400, 3)) * 255).astype(np.uint8)
    dt = [t.box(60, 60 + i * 60, 900, 100 + i * 60) for i in range(30)]
    rec = [("牛丼小" if i % 2 else "サラダセット", 0.9) for i in range(30)]
    sm.apply_ocr_result(trays, "t1", "ticket", crop, dt, rec, False, tray.ticket.bbox)

    # 貼紙：400x200 的裁切圖
    for i in range(args.stickers):
        x = 120 + (i % 8) * 110
        y = 120 + (i // 8) * 80
        ts = t.add_sticker(tray, t.poly(x, y, x + 100, y + 60))
        tray.state = t.TrayState.CHECKING_STICKERS
        ts.is_ocr_busy = True
        sm.apply_ocr_result(trays, "t1", "sticker", (rng.random((200, 400, 3)) * 255).astype(np.uint8),
                            [t.box(10, 30, 380, 170)], [(["牛丼小", "豚丼特盛", "豚丼並"][i % 3], 0.9)],
                            False, ts.bbox)

    # 醬料包：從 4024x3036 原圖裁
    ds.set_frame((rng.random((3036, 4024, 3)) * 255).astype(np.uint8))
    tray.state = t.TrayState.WAITING_STICKERS
    for i in range(args.known):
        s = t.add_sticker(tray, t.poly(60 * i, 400, 60 * i + 50, 450), cls="sesame")
        s.stable_frames = 99
        sm.resolve_known_class_items(trays)
    ds.set_frame(None)

    t0 = time.perf_counter()
    path = ds.finalize("t1", dict(t.SUMMARY))
    return time.perf_counter() - t0, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stickers", type=int, default=10)
    ap.add_argument("--known", type=int, default=3)
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--out", default=None, help="輸出資料夾（預設暫存，跑完刪除）")
    args = ap.parse_args()
    tmp = None
    if args.out is None:
        tmp = tempfile.TemporaryDirectory()
        args.out = tmp.name

    rng = np.random.default_rng(0)
    cfg = SimpleNamespace(enabled=True, mode="all", root_dir=args.out, folder="debug_sheet", canvas_w=4024,
                          canvas_h=3036, jpeg_quality=85, save_json=True, font_path="", max_entries=80,
                          crop_max_side=1600, slow_record_ms=50.0, slow_render_s=5.0)
    ctx = DebugContext(menu_variants=t.VARIANTS, object_class_items={"sesame": "ごまドレッシング"})
    ds = DebugSheet(cfg, args.out, ROOT, ctx,
                    crop_fn=lambda f, xywhr: cv2.warpAffine(f, np.float32([[1, 0, -100], [0, 1, -100]]), (300, 200)))
    fin = []
    for _ in range(args.orders):
        dt, _path = one_order(ds, args, rng)
        fin.append(dt)
    ds.shutdown(timeout=120)

    print("\n================ 摘要 ================")
    print(f"{args.orders} 筆訂單，每筆 傳票1 + 貼紙{args.stickers} + 醬料包{args.known}")
    print(f"finalize（主迴圈）  平均 {statistics.mean(fin) * 1000:.2f}ms  最大 {max(fin) * 1000:.2f}ms")
    tot = ds._totals
    if tot["n"]:
        print(f"背景每張圖          平均 {tot['total_sum'] / tot['n']:.2f}s  最大 {tot['total_max']:.2f}s  "
              f"(其中渲染平均 {tot['render_sum'] / tot['n']:.2f}s)")
    if tmp:
        tmp.cleanup()


if __name__ == "__main__":
    main()
