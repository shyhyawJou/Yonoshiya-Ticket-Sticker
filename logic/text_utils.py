"""
text_utils.py
=============
文字正規化 + 模糊比對工具。

這裡刻意把「原本混雜在 LogicEngine 上的 free function + method」
統整成一個 FuzzyMatcher class，讓 OrderParser / StickerMatcher
可以共用同一份設定 (special_prefixes / score_threshold)，
而不用各自 import 孤立的 function 並重複傳參數。
"""

from __future__ import annotations

import unicodedata
from typing import List, Optional, Sequence, Tuple

from loguru import logger
from rapidfuzz import fuzz, process


def normalize_text(text: str, lower: bool = True, ignore_space: bool = True) -> str:
    """NFKC-normalize unicode 並移除前後空白，可選擇轉小寫與移除所有空白。"""
    text = unicodedata.normalize("NFKC", text).strip()
    if lower:
        text = text.lower()
    if ignore_space:
        text = text.replace(" ", "")
    return text


class FuzzyMatcher:
    """
    對一段 OCR 文字，在候選清單中找出最相近的品項名稱。

    行為分兩階段：
        1. 若文字前綴命中 special_prefixes（例如「ねぎ」開頭的品項，
           這類品項彼此外觀差異小，容易被一般 fuzzy match 誤判），
           改用「長度完全相同 + 前綴相同」的精準邏輯先篩一輪。
        2. 其餘情況（或特殊邏輯 fallback 失敗時）走標準的
           token_set_ratio 模糊比對。
    """

    def __init__(
        self,
        special_prefixes: Sequence[str] = ("ねぎ",),
        prefix_len: int = 2,
        special_score_threshold: int = 35,
        default_score_threshold: int = 40,
    ):
        self.special_prefixes = tuple(special_prefixes)
        self.prefix_len = prefix_len
        self.special_score_threshold = special_score_threshold
        self.default_score_threshold = default_score_threshold

    def match(
        self,
        text: str,
        candidates: List[str],
    ) -> Optional[Tuple[str, int]]:
        text = normalize_text(text, ignore_space=True)
        text_len = len(text)
        prefix = text[: self.prefix_len]

        # 只有在前綴命中 special_prefixes 時才走特殊邏輯
        if any(text.startswith(p) for p in self.special_prefixes):
            same_len_candidates = [
                c for c in candidates if len(c) == text_len and c.startswith(prefix)
            ]
            if len(same_len_candidates) == 1:
                logger.info(f"[前綴唯一] '{text}' → '{same_len_candidates[0]}'")
                return same_len_candidates[0], 100
            if len(same_len_candidates) > 1:
                result = process.extractOne(text, same_len_candidates, scorer=fuzz.ratio)
                if result and result[1] > self.special_score_threshold:
                    logger.info(f"[前綴+字數] '{text}' → '{result[0]}' score={result[1]}")
                    return result[0], result[1]
            # 特殊前綴但比不到，也 fallback 下去

        # 一般邏輯，其他品項完全不受影響
        result = process.extractOne(text, candidates, scorer=fuzz.token_set_ratio)
        if result and result[1] > self.default_score_threshold:
            return result[0], result[1]

        return None