"""
簡易單元測試：驗證 OrderParser / StickerMatcher 拆分後行為正確。

這是 Step 2 拆分帶來的直接好處：以前這些邏輯黏在 LogicEngine 上，
必須組出整個 LogicEngine（含 MqttBus、Rotated_RTMDET、CsvWriter 等
依賴）才能測試；現在可以直接餵假的 dt_boxes / rec_res 進去驗證。

執行方式：
    python3 -m pytest tests/test_order_parser_and_sticker_matcher.py -v
或直接：
    python3 tests/test_order_parser_and_sticker_matcher.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from logic.text_utils import FuzzyMatcher
from logic.szm.order_parser import OrderParser
from logic.readers.sticker_matcher import StickerMatcher


def box(x1, y1, x2, y2):
    """建立一個水平無傾斜的 4 點框：[左上, 右上, 右下, 左下]"""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class DummyRecPred:
    """假的 rec_pred，數量判斷方法三不會被觸發到就不會被呼叫。"""
    def predict_quantity(self, *args, **kwargs):
        raise AssertionError("方法三不應該在有 leading_digit 時被呼叫")


def test_order_parser_leading_digit():
    """情境：訂單上寫「2味噌ラーメン」，應該用方法一 (前綴數字) 判斷數量=2"""
    menus_ticket = ["味噌ラーメン", "醤油ラーメン"]
    matcher = FuzzyMatcher()
    parser = OrderParser(
        menus_ticket=menus_ticket,
        menus_with_digits=set(),  # 沒有品項名稱本身含數字
        fuzzy_matcher=matcher,
        rec_pred=DummyRecPred(),
    )

    dt_boxes = [box(0, 0, 30, 30)]
    rec_res = [("2味噌ラーメン", 0.9)]
    frame_crop = np.zeros((100, 200, 3), dtype=np.uint8)

    parsed_items, order_number = parser.parse(frame_crop, dt_boxes, rec_res, is_flip=False)

    assert parsed_items == [("味噌ラーメン", 2)], f"預期數量=2，實際: {parsed_items}"
    print("[PASS] test_order_parser_leading_digit:", parsed_items)


def test_order_parser_no_match():
    """情境：完全辨識不到任何菜單品項，應該回傳空清單"""
    menus_ticket = ["味噌ラーメン"]
    matcher = FuzzyMatcher()
    parser = OrderParser(
        menus_ticket=menus_ticket,
        menus_with_digits=set(),
        fuzzy_matcher=matcher,
        rec_pred=DummyRecPred(),
    )

    dt_boxes = [box(0, 0, 30, 30)]
    rec_res = [("完全無關的文字", 0.9)]
    frame_crop = np.zeros((100, 200, 3), dtype=np.uint8)

    parsed_items, order_number = parser.parse(frame_crop, dt_boxes, rec_res, is_flip=False)
    assert parsed_items == [], f"預期無比對結果，實際: {parsed_items}"
    print("[PASS] test_order_parser_no_match:", parsed_items)


def test_sticker_matcher_matched():
    """情境：貼紙 OCR 結果對應到 expected_items 中還沒核對過的品項 -> MATCHED"""
    menus_sticker = ["味噌ラーメン", "醤油ラーメン"]
    matcher = FuzzyMatcher()
    sticker_matcher = StickerMatcher(menus_sticker=menus_sticker, fuzzy_matcher=matcher)

    dt_boxes = [box(0, 0, 30, 30)]
    rec_res = [("味噌ラーメン", 0.9)]
    expected_items = ["味噌ラーメン", "醤油ラーメン"]
    checked_items = []

    matched_item, status = sticker_matcher.match(
        rec_res, dt_boxes, is_flip=False,
        expected_items=expected_items, checked_items=checked_items,
    )
    assert status == "MATCHED"
    assert matched_item == "味噌ラーメン"
    print("[PASS] test_sticker_matcher_matched:", matched_item, status)


def test_sticker_matcher_wrong_item():
    """情境：貼紙比對到的品項不在剩餘待核對清單中 -> WRONG_ITEM"""
    menus_sticker = ["味噌ラーメン", "醤油ラーメン"]
    matcher = FuzzyMatcher()
    sticker_matcher = StickerMatcher(menus_sticker=menus_sticker, fuzzy_matcher=matcher)

    dt_boxes = [box(0, 0, 30, 30)]
    rec_res = [("醤油ラーメン", 0.9)]
    expected_items = ["味噌ラーメン"]  # 訂單裡沒有醤油拉麵
    checked_items = []

    matched_item, status = sticker_matcher.match(
        rec_res, dt_boxes, is_flip=False,
        expected_items=expected_items, checked_items=checked_items,
    )
    assert status == "WRONG_ITEM"
    assert matched_item == "醤油ラーメン"
    print("[PASS] test_sticker_matcher_wrong_item:", matched_item, status)


def test_sticker_matcher_already_all_checked():
    """情境：所有品項都已核對完畢 -> UNRECOGNIZED (remaining_items 為空提早 return)"""
    menus_sticker = ["味噌ラーメン"]
    matcher = FuzzyMatcher()
    sticker_matcher = StickerMatcher(menus_sticker=menus_sticker, fuzzy_matcher=matcher)

    dt_boxes = [box(0, 0, 30, 30)]
    rec_res = [("味噌ラーメン", 0.9)]
    expected_items = ["味噌ラーメン"]
    checked_items = ["味噌ラーメン"]  # 已經核對過了

    matched_item, status = sticker_matcher.match(
        rec_res, dt_boxes, is_flip=False,
        expected_items=expected_items, checked_items=checked_items,
    )
    assert status == "UNRECOGNIZED"
    assert matched_item is None
    print("[PASS] test_sticker_matcher_already_all_checked:", matched_item, status)


if __name__ == "__main__":
    test_order_parser_leading_digit()
    test_order_parser_no_match()
    test_sticker_matcher_matched()
    test_sticker_matcher_wrong_item()
    test_sticker_matcher_already_all_checked()
    print("\n全部測試通過 ✅")