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
from collections import Counter
from loguru import logger

from logic.menu_alias import FoldedMatcher
from logic.text_utils import FuzzyMatcher, match_restricted, normalize_text

MatchStatus = str  # "MATCHED" | "WRONG_ITEM" | "UNRECOGNIZED"


class StickerMatcher:
    """貼紙比對。

    2026-09-10：貼紙上只印一個品名，沒有數量欄位，所以這裡不產生任何
    數量資訊（呼叫端也不得預設成 1）。比對改用 match_restricted：
    小候選集 + 與第二名的差距門檻，避免「最相近就接受」。
    """

    def __init__(
        self,
        menus_sticker: List[str],
        fuzzy_matcher: FuzzyMatcher,
        vocab_sticker: Optional[List[str]] = None,
        min_score: float = 65.0,
        min_margin: float = 10.0,
        menu_index=None,
    ):
        self.menus_sticker = menus_sticker
        self.fuzzy_matcher = fuzzy_matcher
        self.vocab_sticker = list(vocab_sticker) if vocab_sticker else None
        self.min_score = min_score
        self.min_margin = min_margin
        # 菜名等價層（logic/menu_alias.MenuIndex）。None 代表退回舊行為。
        self.menu_index = menu_index
        # 最近一次比對的細節，供除錯/輸出使用
        self.last_decision = None
        # 最近一次成功比對的原始 OCR 文字。前端顯示「現場貼紙實際印什麼」
        # 要靠它，不然畫面上的寫法可能跟店家的貼紙不一樣。
        self.last_matched_text = None
        # 最近一次 match() 實際拿來比對的那個文字框的原始文字（不看信心值）。
        # 比對不到菜名時，前端用它顯示「Unknown:讀到的字」，快速找出有問題的貼紙。
        # 沒有讀到任何文字（或根本沒做比對）為 None。
        self.last_raw_text = None
        # 最近一次 match() 最佳比對的分數（0~100）與該文字的 rec 信心值；沒比對到菜名為 0。
        # 錯誤品項確認用它分強弱證據：一字不差讀成別的菜名幾乎都是真的放錯。
        self.last_match_score = 0.0
        self.last_match_conf = 0.0
        # 最佳比對是不是「真的一字不差」：等價層直接命中，或折疊後字串完全相同。
        # 分數 100 不等於一字不差 —— ねぎ 前綴邏輯只要字數、前綴相同又唯一就給 100（'ネギ塩丼小' -> 'ねぎラー油'）。
        self.last_match_exact = False
        # 折疊比對工具（傳票側的 OrderParser 用同一個 class）
        self.folded = FoldedMatcher(menu_index)

    # ------------------------------------------------------------------
    def _fold(self, text: str) -> str:
        return self.folded.fold(text)

    def _resolve_name(self, text: str, candidates: List[str]):
        return self.folded.resolve(text, candidates, self.fuzzy_matcher)

    # ------------------------------------------------------------------
    def read_text(self, rec_res: list, dt_boxes: list):
        """只做「這張貼紙上的日文是什麼」，不涉及訂單核對。

        回傳 (name|None, MatchDecision|None)。找不到可信結果時 name 為 None，
        呼叫端應輸出未確認，不可猜測。
        """
        candidates = self.vocab_sticker or self.menus_sticker
        picked = self._pick_primary(rec_res, dt_boxes)
        if picked is None:
            self.last_decision = None
            return None, None
        text, score = picked
        # key=self._fold：在折疊後的座標系計分，同菜異寫才會落在同一點上
        d = match_restricted(text, candidates,
                             min_score=self.min_score, min_margin=self.min_margin,
                             key=self._fold)
        self.last_decision = d
        return (d.name if d.accepted else None), d

    @staticmethod
    def _pick_primary(rec_res: list, dt_boxes: list):
        """貼紙轉正後，取最靠上方（y 最小）的那個框當主標題。"""
        if not rec_res:
            return None
        idx = 0
        if dt_boxes is not None and len(dt_boxes) == len(rec_res) and len(dt_boxes) > 1:
            idx = min(range(len(dt_boxes)),
                      key=lambda i: min(pt[1] for pt in dt_boxes[i]))
        text, score = rec_res[idx][0], float(rec_res[idx][1])
        return (text, score)

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
        self.last_raw_text = None
        self.last_match_score = 0.0
        self.last_match_conf = 0.0
        self.last_match_exact = False
        if not expected_items:
            return None, "UNRECOGNIZED"

        remaining_items = copy.deepcopy(expected_items) if expected_items else []
        for item in checked_items:
            if item in remaining_items:
                remaining_items.remove(item)

        # remaining_items 空了（品項都核對過、訂單被錯誤品項擋著）也照樣讀：讀得出品名就是
        # 多放的（WRONG_ITEM）。以前這裡直接回 UNRECOGNIZED，多放的那張永遠不會被報出來；
        # 啟用 wrong_item_confirm 後更會讓確認中的貼紙累積不到次數，訂單卡住又沒有警報。

        # 收集所有 rec_res 中分數夠高的候選比對結果，最後選最佳
        best_matched_str = None
        best_match_score = 0
        best_raw_text = None
        best_conf = 0.0
        best_exact = False
        last_text = ""
        candidates = self.vocab_sticker or self.menus_sticker

        if dt_boxes:
            # 貼紙轉正後，挑選 y 軸最小的（最靠上方的文字）
            def box_top_y(box):
                return min([pt[1] for pt in box])

            top_idx = min(range(len(dt_boxes)), key=lambda i: box_top_y(dt_boxes[i]))
            rec_res = [rec_res[top_idx]] if top_idx < len(rec_res) else rec_res

        if rec_res:
            self.last_raw_text = normalize_text(rec_res[0][0]) or None

        for text, score in rec_res:
            if score < 0.5:
                continue
            last_text = normalize_text(text)

            matched_str, match_score = self._resolve_name(text, candidates)
            if matched_str is not None and match_score > 60 and match_score > best_match_score:
                best_match_score = match_score
                best_matched_str = matched_str
                best_raw_text = last_text
                best_conf = float(score)
                best_exact = (
                    (self.menu_index is not None and self.menu_index.canonical(text) is not None)
                    or self._fold(text) == self._fold(matched_str)
                )

        self.last_matched_text = best_raw_text
        self.last_match_score = float(best_match_score)
        self.last_match_conf = best_conf
        self.last_match_exact = bool(best_matched_str is not None and best_exact)

        if best_matched_str is not None:
            if best_matched_str in remaining_items:
                logger.info(f"[餐點確認成功]: 最佳OCR {best_raw_text} 對應 -> '{best_matched_str}' (score: {best_match_score})")
                return best_matched_str, "MATCHED"
            else:
                logger.warning(
                    f"[錯誤餐點]: 最佳OCR {best_raw_text} 對應 -> '{best_matched_str}' "
                    f"(score: {best_match_score})，不在該餐盤待核對清單 {sorted(set(remaining_items))}"
                )
                return best_matched_str, "WRONG_ITEM"

        logger.warning(f"[餐點確認失敗]: OCR -> '{last_text}' (折疊後='{self._fold(last_text)}')")
        return None, "UNRECOGNIZED"

    def match_known_item(self, item_name: str, expected_items: List[str], checked_items: List[str]):
        """
        身份已由分類器直接給定(不需要 OCR 文字比對)時的比對邏輯。
        回傳 (item_name, match_status),status 為 "MATCHED" 或 "WRONG_ITEM"。
        """
        remaining = Counter(expected_items) - Counter(checked_items)
        if remaining.get(item_name, 0) > 0:
            return item_name, "MATCHED"
        return item_name, "WRONG_ITEM"