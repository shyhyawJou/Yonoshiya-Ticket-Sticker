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

# ----------------------------------------------------------------------------
# 詞彙受限比對（margin-aware）
# ----------------------------------------------------------------------------
# 部署到鈴茂資料時發現：候選清單若是吉野家 1235 筆菜單，而實際票據只會出現
# images/日文ticket.txt 的 8 個詞，則即使 OCR 文字完全正確，
#   のり弁当 / 箸 / 醤油 / おしぼり / 紅しょうが -> None（整列被丟棄）
#   サラダ・味噌汁セ -> 'サラダみそ汁セット' (58.8)   ← 錯配
#   ドレッシング     -> 'ごまドレッシング'   (85.7)   ← 錯配
# 也就是 0/8。修正的關鍵是「候選清單必須是該部署真正會出現的詞」，
# 再加上「第一名與第二名要有足夠差距」才接受，避免相近候選硬配。
# ----------------------------------------------------------------------------


class MatchDecision:
    """一次詞彙比對的完整結果，保留原始 OCR 文字以便追查。

    accepted 為 False 時 name 仍可能有值（最相近候選），呼叫端應以
    accepted 判斷是否採用，不要只看 name。
    """

    __slots__ = ("raw", "normalized", "name", "score", "runner_up", "margin",
                 "accepted", "reason")

    def __init__(self, raw, normalized, name, score, runner_up, margin,
                 accepted, reason):
        self.raw = raw
        self.normalized = normalized
        self.name = name
        self.score = score
        self.runner_up = runner_up
        self.margin = margin
        self.accepted = accepted
        self.reason = reason

    def __repr__(self):
        return (f"MatchDecision(raw={self.raw!r}, name={self.name!r}, "
                f"score={self.score:.1f}, margin={self.margin:.1f}, "
                f"accepted={self.accepted}, reason={self.reason})")


def match_restricted(
    text: str,
    candidates: Sequence[str],
    min_score: float = 62.0,
    min_margin: float = 8.0,
    scorer=None,
    key=None,
) -> MatchDecision:
    """在一個「小而封閉」的候選清單中比對，並要求與第二名有足夠差距。

    Args:
        min_score:  第一名的最低分數。
        min_margin: 第一名與第二名的最小差距；候選只有一個時不套用。
        key:        可選的座標轉換函式，套用在 OCR 文字與候選兩側之後才計分
                    （例如 logic.menu_alias 的折疊，讓 唐揚げ 與 から揚げ
                    落在同一個點上）。回傳的候選仍是**原始字串**，不是轉換後
                    的結果 —— 折疊後的字串只能用來比對，不可外流顯示。

    設計理由：token_set_ratio 對「候選是 OCR 文字的超集」過於寬容
    （'ドレッシング' vs 'ごまドレッシング' 可得 85.7），因此這裡預設改用
    fuzz.ratio，它會懲罰長度差異。
    """
    scorer = scorer or fuzz.ratio
    norm = normalize_text(text, ignore_space=True)
    if not norm or not candidates:
        return MatchDecision(text, norm, None, 0.0, 0.0, 0.0, False, "empty")

    probe = key(norm) if key else norm
    scored = sorted(
        ((float(scorer(probe, key(c) if key else c)), c) for c in candidates),
        key=lambda t: -t[0],
    )
    best_score, best_name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    margin = best_score - runner_up

    if best_score < min_score:
        return MatchDecision(text, norm, best_name, best_score, runner_up,
                             margin, False, "below_min_score")
    if len(scored) > 1 and margin < min_margin:
        return MatchDecision(text, norm, best_name, best_score, runner_up,
                             margin, False, "ambiguous_margin")
    return MatchDecision(text, norm, best_name, best_score, runner_up,
                         margin, True, "ok")
