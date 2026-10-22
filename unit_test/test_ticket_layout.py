# -*- coding: utf-8 -*-
"""
test_ticket_layout.py
=====================
logic/szm/ticket_layout.py 的單元測試。純幾何 / 純字串，不需要模型或 NPU，
所以可以在開發機直接跑：

    cd DLA_on_720
    py -3.12 unit_test/test_ticket_layout.py

版面座標取自 images/order_ticket 實測比例（表格寬為 1.0 時欄界在
[0, 0.105, 0.576, 0.680, 0.994]）。這裡用表格寬 400px、左緣 60px 換算：
    品名欄  102 ~ 290
    数量欄  290 ~ 332
    金額欄  332 ~ 458
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from logic.szm.ticket_layout import (
    TextItem, build_items, find_order_number, find_qty_column,
    group_rows, parse_rows, to_legacy_items,
)

VOCAB = ["のり弁当", "サラダ・味噌汁セ", "箸", "醤油", "おしぼり", "紅しょうが",
         "ドレッシング", "数量"]

NAME_X = (104, 200)
QTY_X = (305, 318)
AMOUNT_X = (420, 455)
HEADER_Y = 130
ROW0_Y = 170
ROW_PITCH = 27

_failures = []


def check(cond, label, extra=""):
    if cond:
        print(f"[PASS] {label}")
    else:
        print(f"[FAIL] {label} {extra}")
        _failures.append(label)


def quad(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def item(text, x0, x1, cy, h=16.0, score=0.95):
    return TextItem(text=text, score=score,
                    quad=quad(x0, cy - h / 2, x1, cy + h / 2))


def ticket(rows, with_header=True, with_amounts=True, order_no="00821-01"):
    """rows: list of (name, qty_text_or_None, amount_text)"""
    items = []
    if order_no:
        items.append(item(f"伝票No {order_no}", 230, 300, 40))
    if with_header:
        items.append(item("品名", 150, 190, HEADER_Y))
        items.append(item("数量", 295, 328, HEADER_Y))
        items.append(item("金額", 400, 440, HEADER_Y))
    for i, (name, qty, amount) in enumerate(rows):
        y = ROW0_Y + i * ROW_PITCH
        items.append(item(name, NAME_X[0], NAME_X[0] + 22 * len(name), y))
        if qty is not None:
            items.append(item(qty, QTY_X[0], QTY_X[1], y))
        if with_amounts and amount is not None:
            items.append(item(amount, AMOUNT_X[0], AMOUNT_X[1], y))
    return items


def parse(items, **kw):
    qc = find_qty_column(items)
    return parse_rows(items, VOCAB, qc, exclude_above_cy=HEADER_Y - 5, **kw), qc


# ---------------------------------------------------------------------------

def test_amount_column_not_taken_as_quantity():
    """金額欄的 100 / 0 絕不可被當成數量。這是舊版最主要的錯配來源。"""
    items = ticket([("のり弁当", "1", "100"), ("醤油", "2", "0")])
    rows, qc = parse(items)
    got = to_legacy_items(rows)
    check(got == [("のり弁当", 1), ("醤油", 2)], "amount column not used as quantity", got)
    check(qc is not None and not qc.contains(437.5),
          "qty column gate excludes amount column x", qc)


def test_amount_only_row_gives_unconfirmed_not_amount():
    """數量欄空白、金額欄有數字時，必須是未確認，不能拿金額。"""
    items = ticket([("のり弁当", None, "100")])
    rows, _ = parse(items)
    check(len(rows) == 1 and rows[0].name == "のり弁当", "row kept", rows)
    check(rows[0].quantity is None, "missing qty -> None (not the amount)", rows[0])
    check(rows[0].quantity_source == "unconfirmed", "source marked unconfirmed", rows[0])


def test_never_defaults_to_one():
    items = ticket([("箸", None, "0")])
    rows, _ = parse(items)
    check(rows and rows[0].quantity is None, "quantity never defaults to 1", rows)


def test_duplicate_names_kept_per_row():
    """同一品名出現在不同列必須各自保留，數量不可合併。"""
    items = ticket([
        ("のり弁当", "1", "100"), ("醤油", "2", "0"),
        ("のり弁当", "1", "100"), ("醤油", "3", "0"),
    ])
    rows, _ = parse(items)
    got = to_legacy_items(rows)
    check(got == [("のり弁当", 1), ("醤油", 2), ("のり弁当", 1), ("醤油", 3)],
          "duplicate item names preserved per row with own quantities", got)


def test_order_number_not_the_date():
    """票據抬頭印 2026-08-26，寬鬆式 regex 會誤中，必須排除日期。"""
    items = ticket([("箸", "1", "0")], order_no="00821-01")
    items.append(item("2026-08-26", 60, 140, 62))
    no, _ = find_order_number(items)
    check(no == "00821-01", "date not mistaken for order number", no)


def test_order_number_date_only_gives_none():
    """只有日期、沒有票號時，寧可回 None 也不要回日期。"""
    items = [item("2026-08-26", 60, 140, 62), item("箸", 104, 130, ROW0_Y)]
    no, _ = find_order_number(items)
    check(no is None, "date alone -> None", no)


def test_order_number_full_string_with_leading_zeros():
    items = ticket([("箸", "1", "0")], order_no="00821-01")
    no, _ = find_order_number(items)
    check(no == "00821-01", "order number keeps leading zeros and hyphen", no)


def test_order_number_not_confused_with_quantity():
    items = ticket([("のり弁当", "1", "100")], order_no="00824-01")
    rows, _ = parse(items)
    got = to_legacy_items(rows)
    check(got == [("のり弁当", 1)], "order number row excluded from items", got)


def test_no_qty_header_all_unconfirmed():
    """找不到「数量」表頭時，寧可全部未確認，也不用其他數字硬猜。"""
    items = ticket([("のり弁当", "1", "100")], with_header=False)
    qc = find_qty_column(items)
    rows = parse_rows(items, VOCAB, qc, exclude_above_cy=HEADER_Y - 5)
    check(qc is None, "qty column not found without header", qc)
    names = [r.name for r in rows if r.name]
    check("のり弁当" in names, "item name still recovered", names)
    check(all(r.quantity is None for r in rows),
          "all quantities unconfirmed when column unknown", rows)


def test_ocr_noise_rejected():
    """小計 / 合計 / 純數字不得被模糊比對硬配成品名。"""
    items = ticket([("のり弁当", "1", "100")])
    items.append(item("小計", 150, 190, ROW0_Y + 6 * ROW_PITCH))
    items.append(item("合計", 150, 190, ROW0_Y + 7 * ROW_PITCH))
    items.append(item("200円", 400, 450, ROW0_Y + 7 * ROW_PITCH))
    rows, _ = parse(items)
    got = to_legacy_items(rows)
    check(got == [("のり弁当", 1)], "subtotal/total rows rejected", got)


def test_ocr_typo_tolerated():
    """OCR 常把のり弁当的り誤成リ；仍應配到正確品名。"""
    items = ticket([("のリ弁当", "2", "100")])
    rows, _ = parse(items)
    got = to_legacy_items(rows)
    check(got == [("のり弁当", 2)], "katakana-ri typo still matched", got)


def test_ambiguous_match_rejected():
    """與第二名差距不足時應拒絕，不可硬選第一名。"""
    from logic.text_utils import match_restricted
    d = match_restricted("醤", ["醤油", "醤汁"], min_score=50.0, min_margin=20.0)
    check(not d.accepted and d.reason == "ambiguous_margin",
          "ambiguous candidates rejected by margin", d)


def test_row_grouping_by_overlap():
    """同列的大字品名與小字數量字高不同，仍須分到同一列。"""
    items = [item("のり弁当", 104, 190, 200, h=20),
             item("1", 305, 318, 202, h=11),
             item("100", 420, 455, 201, h=12)]
    rows = group_rows(items)
    check(len(rows) == 1 and len(rows[0]) == 3,
          "different glyph heights still group into one row", rows)


def test_build_items_length_mismatch():
    boxes = [quad(0, 0, 10, 10), quad(0, 20, 10, 30)]
    rec = [("a", 0.9)]
    got = build_items(boxes, rec)
    check(len(got) == 1, "build_items tolerates length mismatch", got)


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        fn()
    print()
    if _failures:
        print(f"FAILED {len(_failures)}: {_failures}")
        sys.exit(1)
    print("全部測試通過 ✅")
