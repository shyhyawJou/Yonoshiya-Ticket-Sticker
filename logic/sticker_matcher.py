"""
sticker_matcher.py
===================
負責把一次「貼紙 (sticker)」的 OCR 結果比對到某個訂單的
expected_items / checked_items，判斷這張貼紙是：
    - MATCHED       比對成功且還沒被核對過的品項
    - WRONG_ITEM    有比對到品項，但不在剩餘待核對清單中
    - UNRECOGNIZED  比對不到任何品項

跟 OrderParser 一樣，這裡完全不知道 tray 是什麼；輸入是 OCR 結果 +
一份「還剩什麼品項要核對」的清單，輸出是比對結果。tray 只是目前
呼叫端剛好用來持有 expected_items / checked_items 而已。
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

from loguru import logger

from logic.text_utils import FuzzyMatcher, normalize_text

MatchStatus = str  # "MATCHED" | "WRONG_ITEM" | "UNRECOGNIZED"


class StickerMatcher:
    def __init__(self, menus_sticker: List[str], fuzzy_matcher: FuzzyMatcher):
        self.menus_sticker = menus_sticker
        self.fuzzy_matcher = fuzzy_matcher

    def match(
        self,
        rec_res: list,
        dt_boxes: list,
        is_flip: bool,
        expected_items: List[str],
        checked_items: List[str],
    ) -> Tuple[Optional[str], MatchStatus]:
        """
        Returns:
            (matched_item, status)
            matched_item 在 UNRECOGNIZED 時為 None
        """
        if not expected_items:
            return None, "UNRECOGNIZED"

        remaining_items = copy.deepcopy(expected_items) if expected_items else []
        for item in checked_items:
            if item in remaining_items:
                remaining_items.remove(item)

        if not remaining_items:
            return None, "UNRECOGNIZED"

        # 收集所有 rec_res 中分數夠高的候選比對結果，最後選最佳
        best_matched_str = None
        best_match_score = 0
        last_text = ""

        if dt_boxes:
            # 貼紙轉正後，挑選 y 軸最小的（最靠上方的文字）
            def box_top_y(box):
                return min([pt[1] for pt in box])

            top_idx = min(range(len(dt_boxes)), key=lambda i: box_top_y(dt_boxes[i]))
            rec_res = [rec_res[top_idx]] if top_idx < len(rec_res) else rec_res

        for text, score in rec_res:
            if score < 0.5:
                continue
            last_text = normalize_text(text)

            match_result = self.fuzzy_matcher.match(text, self.menus_sticker)
            if match_result:
                matched_str, match_score = match_result
                if match_score > 60 and match_score > best_match_score:
                    best_match_score = match_score
                    best_matched_str = matched_str

        if best_matched_str is not None:
            if best_matched_str in remaining_items:
                logger.info(f"[餐點確認成功]: 最佳OCR {last_text} 對應 -> '{best_matched_str}' (score: {best_match_score})")
                return best_matched_str, "MATCHED"
            else:
                logger.warning(f"[錯誤餐點]: 最佳OCR {last_text}  對應 -> '{best_matched_str}'，不在該餐盤訂單中")
                return best_matched_str, "WRONG_ITEM"

        logger.warning(f"[餐點確認失敗]: OCR -> '{last_text}'")
        return None, "UNRECOGNIZED"