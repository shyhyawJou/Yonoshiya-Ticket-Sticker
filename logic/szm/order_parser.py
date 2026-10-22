"""
order_parser.py
===============
票據 OCR 結果 -> (品名, 數量) 列表 + 票據編號。

2026-09-10 改版摘要
-------------------
舊版的三個具體問題（實測證據見 tools/eval_ticket_parsing.py 的輸出）：

1. 候選詞庫錯誤。舊版拿 cfg.menus_ticket（menus_mapping.yaml 的 1235 筆
   吉野家菜名）當候選，但本部署的票據只會出現 images/日文ticket.txt 的
   8 個詞。即使 OCR 文字完全正確：
       のり弁当 / 箸 / 醤油 / おしぼり / 紅しょうが -> None（整列被丟棄）
       サラダ・味噌汁セ -> 'サラダみそ汁セット' (58.8)  ← 錯配
       ドレッシング     -> 'ごまドレッシング'  (85.7)  ← 錯配
   -> 現在改由 cfg 注入該部署真正的詞彙（vocab_ticket），並用
      match_restricted（fuzz.ratio + 與第二名的差距門檻）取代
      token_set_ratio，後者對「候選是 OCR 文字的超集」過度寬容。

2. 數量沒有欄位限制。舊版取「與列中心 Y 最近的純數字框」，同一列的
   金額欄（100 / 0）與訂單編號都可能被選中。
   -> 現在先用「数量」表頭定出欄位 X 閘門，只接受落在閘門內的數字。

3. 取不到數量時預設 1。舊版 match_num 初始化為 1，且用 `if match_num == 1`
   當作「還沒取到」的旗標，讓「真的是 1」與「不知道」無法區分。
   -> 現在取不到就是 None（未確認），並在 parse() 回傳值與 last_rows 中
      明確標示，由呼叫端決定如何處理。

保留：SkewCorrector 去斜、quad_validator 形狀檢查、predict_quantity
（數字受限辨識）當第二層 fallback —— 這三個都是既有且在 edge 上已驗證的能力。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from logic.text_utils import FuzzyMatcher
from logic.szm.ticket_layout import (
    QUANTITY_RE, TicketRow, build_items, find_order_number, find_qty_column,
    parse_rows, to_legacy_items,
)
from utils.skew_corrector import SkewCorrector
from utils.quad_validator import validate_and_fix_boxes, find_angle_outliers

# 數量可能是 None（未確認）。呼叫端務必檢查，不可直接當 1 使用。
ParsedItem = Tuple[str, Optional[int]]


class OrderParser:
    def __init__(
        self,
        menus_ticket: List[str],
        menus_with_digits,
        fuzzy_matcher: FuzzyMatcher,
        rec_pred,  # StandaloneRecDLA
        vocab_ticket: Optional[Sequence[str]] = None,
        min_name_score: float = 62.0,
        min_name_margin: float = 8.0,
        enable_qty_rec_fallback: bool = True,
        min_box_score: float = 0.5,
    ):
        self.menus_ticket = menus_ticket
        self.menus_with_digits = menus_with_digits
        self.fuzzy_matcher = fuzzy_matcher
        self.rec_pred = rec_pred

        # vocab_ticket 為該部署真正會出現的詞彙；未提供時退回 menus_ticket
        # （舊行為），但會警告，因為大詞庫在小場景下錯配率很高。
        if vocab_ticket:
            self.vocab_ticket = list(vocab_ticket)
        else:
            self.vocab_ticket = list(menus_ticket)
            logger.warning(
                "[OrderParser] 未提供 vocab_ticket，退回使用 menus_ticket "
                f"({len(self.vocab_ticket)} 筆)。小場景建議在 config 的 includes 下"
                "設定 vocab_ticket 指向該部署的詞彙檔，以降低錯配。"
            )

        self.min_name_score = min_name_score
        self.min_name_margin = min_name_margin
        self.enable_qty_rec_fallback = enable_qty_rec_fallback
        self.min_box_score = min_box_score

        # 最近一次解析的完整結果，供呼叫端取得「未確認」等細節而不必改簽章
        self.last_rows: List[TicketRow] = []
        self.last_order_number: Optional[str] = None
        self.last_qty_column = None

    # ------------------------------------------------------------------
    def parse(
        self,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool,
    ) -> Tuple[List[ParsedItem], Optional[str]]:
        """回傳 ((品名, 數量|None) 列表, 票據編號|None)。

        注意：frame_crop / dt_boxes 由 AsyncOCR._worker_loop 事先旋轉成
        一致的正立座標系，且 is_flip 已被設為 False，因此這裡不再處理翻轉。
        """
        rows, order_number = self.parse_detailed(frame_crop, dt_boxes, rec_res, is_flip)
        items = to_legacy_items(rows)
        logger.info(f"[訂單解析結果]: {items}, 編號: {order_number}")
        return items, order_number

    # ------------------------------------------------------------------
    def parse_detailed(
        self,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool = False,
    ) -> Tuple[List[TicketRow], Optional[str]]:
        """完整版：回傳每一列的 TicketRow（含未確認狀態與原始 OCR 文字）。"""
        self.last_rows, self.last_order_number, self.last_qty_column = [], None, None

        if dt_boxes is None or len(dt_boxes) == 0 or not rec_res:
            logger.warning("[OrderParser] 沒有文字框，無法解析")
            return [], None

        # 1. 去斜：讓後續的「同一列」判斷在一致的直角座標系上進行
        skew = SkewCorrector.from_dt_boxes(dt_boxes)
        logger.info(f"[傾斜校正] 估計角度: {skew.theta_deg:.2f} 度")
        corrected = skew.correct_boxes(dt_boxes)

        # 2. 形狀檢查：菱形/斗笠形狀的框座標不可信，直接剔除
        corrected, bad_idx = validate_and_fix_boxes(corrected, log_fn=logger.warning)
        bad = set(bad_idx) | set(find_angle_outliers(corrected, max_dev_deg=20.0))
        if bad:
            logger.warning(f"[形狀異常框] 剔除 {len(bad)} 個，index={sorted(bad)}")

        keep_dt, keep_corr, keep_rec = [], [], []
        for i in range(min(len(corrected), len(rec_res))):
            if i in bad:
                continue
            if float(rec_res[i][1]) < self.min_box_score:
                continue
            keep_dt.append(dt_boxes[i])
            keep_corr.append(corrected[i])
            keep_rec.append(rec_res[i])

        items = build_items(keep_corr, keep_rec, orig_boxes=keep_dt)
        if not items:
            logger.warning("[OrderParser] 所有文字框都被剔除")
            return [], None

        # 3. 票據編號（整串，保留前導零）
        order_number, no_item = find_order_number(items)
        self.last_order_number = order_number

        # 4. 數量欄閘門
        qty_col = find_qty_column(items)
        self.last_qty_column = qty_col
        if qty_col is None:
            logger.warning(
                "[OrderParser] 找不到「数量」表頭，本張所有數量標為未確認"
                "（不改用最近數字猜測，避免誤取金額）"
            )
        else:
            logger.info(
                f"[數量欄] x=[{qty_col.x0:.1f}, {qty_col.x1:.1f}] "
                f"(表頭 cx={qty_col.header_cx:.1f}, w={qty_col.header_w:.1f})"
            )

        # 表格開始前的抬頭區（新規/日期/編號/テーブル）不當品項列
        cutoff = None
        if qty_col is not None:
            cutoff = qty_col.header_cy - 0.5 * float(np.median([it.h for it in items]))
        elif no_item is not None:
            cutoff = no_item.cy

        rows = parse_rows(
            items, self.vocab_ticket, qty_col,
            min_score=self.min_name_score,
            min_margin=self.min_name_margin,
            exclude_above_cy=cutoff,
            names_with_digits=self.menus_with_digits or (),
        )

        # 5. 第二層 fallback：數量欄該列沒有 det 框時，直接在數量欄中心裁圖，
        #    用既有的數字受限辨識（predict_quantity 只允許 blank + 0-9）再試一次。
        #    det 常漏掉孤立的單一位數，這一層可救回，但每列一次推論，
        #    因此只對「已辨識出品名但數量未確認」的列執行。
        if self.enable_qty_rec_fallback and qty_col is not None and self.rec_pred is not None:
            for row in rows:
                if row.name is None or row.quantity is not None:
                    continue
                try:
                    rx, ry = skew.inverse_correct_point(qty_col.header_cx, row.cy)
                    text = self.rec_pred.predict_quantity(
                        frame_crop, cx=rx, cy=ry,
                        h=row.height, w=row.height, is_flip=False,
                    )
                    if isinstance(text, tuple):
                        text = text[0]
                    cand = "".join(ch for ch in str(text) if ch.isdigit())
                    m = QUANTITY_RE.match(cand)
                    if m:
                        row.quantity = int(m.group(1))
                        row.quantity_raw = str(text)
                        row.quantity_source = "rec_crop"
                        logger.info(
                            f"[數量-裁圖辨識] '{row.name}' 列 -> {row.quantity} (raw={text!r})"
                        )
                    else:
                        logger.warning(
                            f"[數量-裁圖辨識] '{row.name}' 列辨識結果={text!r} 不是有效數量，"
                            "維持未確認"
                        )
                except Exception as exc:  # 單列失敗不影響其他列
                    logger.error(f"[數量-裁圖辨識] '{row.name}' 列失敗: {exc}")

        self.last_rows = rows
        for r in rows:
            if r.name is None:
                logger.warning(
                    f"[未辨識列] 原始OCR='{r.name_raw}' 最相近='{r.name_reason}' "
                    f"score={r.name_score:.1f} margin={r.name_margin:.1f}"
                )
            elif r.quantity is None:
                logger.warning(f"[數量未確認] '{r.name}' (原始OCR='{r.name_raw}')")
            else:
                logger.info(
                    f"[列確認] '{r.name}' x{r.quantity} "
                    f"(來源={r.quantity_source}, 原始OCR='{r.name_raw}')"
                )
        return rows, order_number

    # ------------------------------------------------------------------
    def summary(self) -> dict:
        """把 last_rows 整理成可直接發布/列印的結構。"""
        rows = self.last_rows
        return {
            "order_number": self.last_order_number,
            "items": [
                {"name": r.name, "quantity": r.quantity,
                 "quantity_source": r.quantity_source, "raw": r.name_raw}
                for r in rows if r.name is not None
            ],
            "unconfirmed_quantity": [
                r.name for r in rows if r.name is not None and r.quantity is None
            ],
            "unrecognized_rows": [
                {"raw": r.name_raw, "score": r.name_score, "margin": r.name_margin,
                 "reason": r.name_reason}
                for r in rows if r.name is None
            ],
            "qty_column_found": self.last_qty_column is not None,
        }
