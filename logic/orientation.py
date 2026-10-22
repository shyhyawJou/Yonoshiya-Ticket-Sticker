# -*- coding: utf-8 -*-
"""
orientation.py
==============
判斷整張裁切影像是否被轉成 90 / 270 度（文字變直式）。純幾何，可離線測試。

為什麼要改判準
--------------
原本的判準在 ocr_engine.TextSystemDLA._check_boxes_need_rotate_90：

    vertical = #{box : h/w >= 1.4}
    需要轉 1/4 圈  <=>  vertical / #box >= 0.2

票據下半部的「小計 / 深夜料金 / 外税 / 合計」金額區印了一整排**直式的「円」**，
再加上單字框（箸、1、0），在**正立**的票據上就足以讓「直式框數 / 總框數」
超過 0.2，於是把正立票據誤轉 90 度。文字被轉成直式後，任何依「同一列」
配對品名↔數量的後處理都會失效。

改用**面積加權**即可完全分開兩類。images/order_ticket 99 張實測
（GT 取自 images/tool/orient_final.json，Claude 逐張看圖驗證過的角度）：

    需要轉 1/4 圈 (26 張)  面積加權直式比例  0.529 ~ 0.743
    不需要轉      (73 張)  面積加權直式比例  0.021 ~ 0.147

    判準                       門檻    正確/99   誤轉   漏轉
    框數比例 (現行)            0.20      48       51      0   48.5%
    框數比例                   0.40      97        0      2   98.0%
    面積加權比例               0.20~0.50 99        0      0  100.0%

門檻取 0.35（安全區 0.20~0.50 的中點），兩側各有約 0.2 的餘裕。
面積加權之所以有效：直式的「円」與單字框數量多但面積小，
而橫式的品名文字行面積大，加權後主導判斷。

量測腳本：tools/eval_orientation.py
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# 單一框被視為「直式」的長寬比門檻（沿用原本的 1.4）
VERTICAL_ASPECT = 1.4
# 面積加權直式比例的門檻。安全區 0.20~0.50，取中點。
AREA_RATIO_THRESHOLD = 0.35


def vertical_area_ratio(dt_boxes: Sequence, vertical_aspect: float = VERTICAL_ASPECT) -> float:
    """直式框的面積 / 所有框的面積。無框時回 0。

    dt_boxes: (N, 4, 2)，每個框為 [左上, 右上, 右下, 左下]。
    """
    if dt_boxes is None or len(dt_boxes) == 0:
        return 0.0
    area_vertical = 0.0
    area_total = 0.0
    for box in dt_boxes:
        b = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        w = float(np.linalg.norm(b[0] - b[1]))
        h = float(np.linalg.norm(b[0] - b[3]))
        if w <= 0.0 or h <= 0.0:
            continue
        area = w * h
        area_total += area
        if h / w >= vertical_aspect:
            area_vertical += area
    if area_total <= 0.0:
        return 0.0
    return area_vertical / area_total


def needs_quarter_turn(
    dt_boxes: Sequence,
    area_ratio_threshold: float = AREA_RATIO_THRESHOLD,
    vertical_aspect: float = VERTICAL_ASPECT,
) -> bool:
    """整張影像是否需要轉 1/4 圈才能讓文字變橫式。"""
    return vertical_area_ratio(dt_boxes, vertical_aspect) >= area_ratio_threshold
