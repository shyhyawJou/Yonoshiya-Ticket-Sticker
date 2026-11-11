"""
stub_parser.py
==============
吉野家客戶聯小票（stub / small_order_ticket）解析器。

負責：
1. 表頭解析（時間、桌號、單號）
2. 品項區解析（4 碼 POS 代碼、易混淆 POS 修正、品名模糊比對、品名反查 POS 碼）
3. 數量欄解析與 CV 補強重讀（StandaloneRecDLA 重讀、垂直對齊數量欄補框、右側漏檢墨跡補抓、細長字體 21->1 校正）
4. 轉換為標準客戶聯 JSON 結構（to_order_json）
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import yaml
from loguru import logger
from rapidfuzz import fuzz, process

from logic.menu_alias import FoldedMatcher
from logic.text_utils import normalize_text
from logic.yoshinoya.stub_cv_det import (
    find_horizontal_lines,
    find_horizontal_line_spans,
    find_rightmost_ink,
    compute_qty_column_x,
    build_aligned_qty_rect,
)
from logic.yoshinoya.stub_layout import (
    Box,
    StubRow,
    crop_box,
    crop_rect,
    filter_Rotated_boxes,
    group_rows,
    split_row_columns,
    to_boxes,
)


_TIME_RE = re.compile(r"(\d{1,2})\s*[:：.\-]\s*(\d{2})")
_POS_CODE_RE = re.compile(r"(?<!\d)(\d{4})(?!\d)")
_PURE_DIGITS_RE = re.compile(r"^\d{1,4}$")
_QTY_RE = re.compile(r"^([1-9]\d?)$")
_SERVICE_MARK_PREFIX_RE = re.compile(r"^(?:TO|T0|1O|7O|TD)(?=[\s　]|[^\x00-\x7F]|$)\s*", re.IGNORECASE)

# 對應 StandaloneRecDLA.predict_quantity 內部的 cx = int(cx) + 10 偏移補償
QTY_REC_X_OFFSET = 10.0

_DIGIT_CONFUSION = str.maketrans({
    "O": "0", "o": "0", "Ｏ": "0", "〇": "0",
    "I": "1", "l": "1", "|": "1", "｜": "1", "丨": "1", "！": "1", "!": "1",
    "Z": "2", "z": "2",
    "S": "5", "s": "5",
    "B": "8",
    "０": "0", "１": "1", "２": "2", "３": "3", "４": "4",
    "５": "5", "６": "6", "７": "7", "８": "8", "９": "9",
})

_DEFAULT_POS_LOOKUP: Dict[str, str] = {
    "みそ汁": "5996", "お新香": "5991", "半熟玉子": "5982", "玉子": "5981",
    "生野菜サラダ": "5993", "ポテトサラダ": "5994", "キムチ": "5992",
    "ねぎ玉子": "5984", "チーズ": "9748",
    "牛皿並盛": "1122", "牛皿大盛": "1123", "牛皿特盛": "1124",
    "牛丼小盛": "1101", "牛丼並盛": "1102", "牛丼アタマの大盛": "1105",
    "牛丼大盛": "1103", "牛丼特盛": "1104", "牛丼超特盛": "1106",
    "豚丼小盛": "1201", "豚丼並盛": "1202", "豚丼アタマの大盛": "1205",
    "豚丼大盛": "1203", "豚丼特盛": "1204", "豚丼超特盛": "1206",
    "牛鮭定食": "2101", "牛さば定食": "2102", "焼魚定食": "2103",
    "特朝定食": "2104", "納豆定食": "2105", "Wハムエッグ納豆定食": "2106",
    "ハムエッグ牛小鉢定食": "2107", "から揚げ定食": "2201", "牛皿定食": "2110",
    "牛皿から揚げ定食": "2205", "ねぎ塩豚定食": "2120", "ねぎ塩牛カルビ定食": "2121",
    "牛カルビ定食": "2115", "牛すき鍋膳": "3101",
    "黒カレー": "9671", "牛黒カレー": "4102", "チーズ黒カレー": "4103", "から揚げ黒カレー": "4104",
    "うなぎ重": "3201", "鰻重一枚盛": "3201", "鰻重二枚盛": "3202",
    "とん汁": "5995", "しじみ汁": "5997", "あさり汁": "5998",
    "唐揚1個": "9677", "から揚げ1個": "9677", "ご飯増量": "9501",
    "ご飯並盛": "5901", "ご飯大盛": "5902",
}

_DEFAULT_POS_CONFUSION: Dict[str, str] = {
    "9996": "5996",
    "5677": "9677",
    "9948": "9748",
}

_DEFAULT_QTY_CORRECTIONS: Dict[str, int] = {
    "21": 1,
}


def _normalize_digits(s: str) -> str:
    return (s or "").translate(_DIGIT_CONFUSION).strip()


@dataclass
class StubItem:
    pos_code: Optional[str]
    name: Optional[str]
    raw_name: str
    quantity: Optional[int]
    score: float = 0.0
    qty_source: str = "det"


@dataclass
class StubParse:
    time_str: Optional[str] = None
    table_no: Optional[str] = None
    ticket_no: Optional[str] = None
    items: List[StubItem] = field(default_factory=list)
    total_qty: Optional[int] = None
    raw_lines: List[str] = field(default_factory=list)
    fallback_rects: List[Tuple[int, int, int, int]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.ticket_no or self.items)

    def as_pairs(self) -> List[Tuple[str, Optional[int]]]:
        return [(it.name, it.quantity) for it in self.items if it.name]

    def to_order_json(self) -> Dict[str, Any]:
        """產生標準客戶聯 JSON 結構。"""
        order_no_val: Any = 0
        if self.ticket_no:
            digits = "".join(ch for ch in str(self.ticket_no) if ch.isdigit())
            order_no_val = int(digits) if digits else self.ticket_no

        item_dict: Dict[str, Dict[str, Any]] = {}
        total_qty = 0
        for idx, it in enumerate(self.items, start=1):
            q = it.quantity if it.quantity is not None else 1
            total_qty += q
            pos_val: Any = 0
            if it.pos_code:
                p_digits = "".join(ch for ch in str(it.pos_code) if ch.isdigit())
                pos_val = int(p_digits) if p_digits else it.pos_code
            conf = round(float(it.score) / 100.0, 2) if it.score else 0.85
            item_dict[str(idx)] = {
                "pos_no": pos_val,
                "name": it.name or it.raw_name or "",
                "quantity": q,
                "confidence": conf,
            }

        return {
            "order_no": order_no_val,
            "quantity": len(self.items),
            "total": total_qty,
            "item": item_dict,
        }


@dataclass
class StubTicketConfig:
    enabled: bool = False
    class_name: str = "small_order_ticket"
    stable_frames: int = 2
    max_attempts: int = 3
    enable_candidate_fallback: bool = False
    enable_full_image_fallback: bool = False
    enable_roi_refinement: bool = False
    max_tilt_deg: float = 25.0
    row_y_overlap: float = 0.45
    qty_min_x_ratio: float = 0.55
    qty_default_x_ratio: float = 0.87
    qty_max_width_ratio: float = 0.25
    header_max_y_ratio: float = 0.28
    footer_min_y_ratio: float = 0.82
    fuzzy_cutoff: float = 55.0
    noise_keywords: Tuple[str, ...] = (
        "テーブル", "ﾃｰﾌﾞﾙ", "人数", "伝票", "合計", "小計",
        "預り", "お釣", "釣り", "領収", "番号", "No", "NO",
        "係", "担当", "レジ", "税込", "税抜", "内税", "外税", "対象", "点", "名",
    )
    pos_lookup: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_POS_LOOKUP))
    pos_confusion: Dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_POS_CONFUSION))
    qty_corrections: Dict[str, int] = field(default_factory=lambda: dict(_DEFAULT_QTY_CORRECTIONS))

    @classmethod
    def from_yaml(cls, path: Optional[str]) -> "StubTicketConfig":
        if not path or not os.path.isfile(path):
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        layout = raw.get("layout") or {}
        mmr_cv = raw.get("mmr_cv_enhance") or {}
        kw = layout.get("noise_keywords")

        pos_lookup = dict(_DEFAULT_POS_LOOKUP)
        if isinstance(layout.get("pos_lookup"), dict):
            for k, v in layout["pos_lookup"].items():
                pos_lookup[str(k)] = str(v)

        pos_confusion = dict(_DEFAULT_POS_CONFUSION)
        if isinstance(layout.get("pos_confusion"), dict):
            for k, v in layout["pos_confusion"].items():
                pos_confusion[str(k)] = str(v)

        qty_corrections = dict(_DEFAULT_QTY_CORRECTIONS)
        if isinstance(layout.get("qty_corrections"), dict):
            for k, v in layout["qty_corrections"].items():
                qty_corrections[str(k)] = int(v)

        return cls(
            enabled=bool(raw.get("enabled", True)),
            class_name=str(raw.get("class_name", cls.class_name)),
            stable_frames=int(raw.get("stable_frames", cls.stable_frames)),
            max_attempts=int(raw.get("max_attempts", cls.max_attempts)),
            enable_candidate_fallback=bool(mmr_cv.get("enable_candidate_fallback", False)),
            enable_full_image_fallback=bool(mmr_cv.get("enable_full_image_fallback", False)),
            enable_roi_refinement=bool(mmr_cv.get("enable_roi_refinement", False)),
            max_tilt_deg=float(layout.get("max_tilt_deg", cls.max_tilt_deg)),
            row_y_overlap=float(layout.get("row_y_overlap", cls.row_y_overlap)),
            qty_min_x_ratio=float(layout.get("qty_min_x_ratio", cls.qty_min_x_ratio)),
            qty_default_x_ratio=float(layout.get("qty_default_x_ratio", cls.qty_default_x_ratio)),
            qty_max_width_ratio=float(layout.get("qty_max_width_ratio", cls.qty_max_width_ratio)),
            header_max_y_ratio=float(layout.get("header_max_y_ratio", cls.header_max_y_ratio)),
            footer_min_y_ratio=float(layout.get("footer_min_y_ratio", cls.footer_min_y_ratio)),
            fuzzy_cutoff=float(layout.get("fuzzy_cutoff", cls.fuzzy_cutoff)),
            noise_keywords=tuple(kw) if kw else cls.noise_keywords,
            pos_lookup=pos_lookup,
            pos_confusion=pos_confusion,
            qty_corrections=qty_corrections,
        )


class StubParser:
    def __init__(
        self,
        menus: Sequence[str],
        cfg: Optional[StubTicketConfig] = None,
        folded: Optional[FoldedMatcher] = None,
        menu_index=None,
    ):
        self.cfg = cfg or StubTicketConfig()
        self.folded = folded or FoldedMatcher(menu_index)
        self.vocab = list(dict.fromkeys(menus or []))
        self._folded_map = self.folded.folded_map(self.vocab)
        self._folded_keys = list(self._folded_map.keys())
        self.last_qty_calls: int = 0
        self.last_qty_ms: float = 0.0
        self._cur_fallback_rects: List[Tuple[int, int, int, int]] = []

    def _predict_qty_timed(
        self,
        rec_pred,
        frame_crop: np.ndarray,
        cx: float,
        cy: float,
        h: float,
        w: float,
        is_flip: bool = False,
    ) -> Tuple[Optional[str], float]:
        t0 = time.perf_counter()
        try:
            text, score = rec_pred.predict_quantity(
                frame_crop, cx=cx, cy=cy, h=h, w=w, is_flip=is_flip
            )
        except TypeError:
            # 向下相容僅接收單一裁切圖參數的舊式或輕量測試替身
            text, score = rec_pred.predict_quantity(frame_crop)
        self.last_qty_calls += 1
        self.last_qty_ms += (time.perf_counter() - t0) * 1000.0
        return text, float(score or 0.0)

    def _clean_qty_str(self, raw_text: Optional[str]) -> Optional[int]:
        if not raw_text:
            return None
        s = _normalize_digits(str(raw_text))
        if s in self.cfg.qty_corrections:
            return self.cfg.qty_corrections[s]
        for k, v in self.cfg.qty_corrections.items():
            if k in s:
                return v
        m = _QTY_RE.match(s)
        if m:
            return int(m.group(1))
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits in self.cfg.qty_corrections:
            return self.cfg.qty_corrections[digits]
        if digits and 1 <= int(digits) <= 99:
            return int(digits)
        return None

    def parse(
        self,
        dt_boxes: Sequence,
        rec_res: Sequence[Tuple[str, float]],
        img: Optional[np.ndarray] = None,
        rec_pred=None,
        is_flip: bool = False,
    ) -> StubParse:
        self.last_qty_calls = 0
        self.last_qty_ms = 0.0
        self._cur_fallback_rects = []
        result = StubParse()

        raw_boxes = to_boxes(dt_boxes, rec_res)
        if not raw_boxes:
            return result

        boxes = filter_Rotated_boxes(raw_boxes, self.cfg.max_tilt_deg)
        if not boxes:
            return result

        if img is not None and img.size > 0:
            height, width = float(img.shape[0]), float(img.shape[1])
        else:
            width = max(b.x1 for b in boxes)
            height = max(b.y1 for b in boxes)

        rows = group_rows(boxes, self.cfg.row_y_overlap)
        result.raw_lines = [" ".join(b.text for b in r.boxes) for r in rows]

        header_cut, footer_cut, separator_x_end = self._locate_bands(rows, img, height)
        header_rows = [r for r in rows if r.cy <= header_cut]
        item_rows = [r for r in rows if header_cut < r.cy < footer_cut]

        # 若雙線或預設比例切分導致品項區為空，退回從非表頭列取品項
        if not item_rows and len(rows) > 1:
            header_rows = rows[:1]
            item_rows = rows[1:]

        self._parse_header(header_rows, width, result)
        self._parse_items(
            item_rows,
            width,
            img,
            rec_pred,
            result,
            separator_x_end=separator_x_end,
            is_flip=is_flip,
        )
        result.fallback_rects = list(self._cur_fallback_rects)
        return result

    def _locate_bands(
        self,
        rows: List[StubRow],
        img: Optional[np.ndarray],
        height: float,
    ) -> Tuple[float, float, Optional[float]]:
        spans = find_horizontal_line_spans(img)
        top_spans = [(y_c, x0, x1) for (y_c, x0, x1) in spans if 0.08 * height <= y_c <= 0.50 * height]
        bot_spans = [(y_c, x0, x1) for (y_c, x0, x1) in spans if 0.65 * height <= y_c <= 0.95 * height]

        separator_x_end: Optional[float] = None
        if top_spans:
            # 優先取上方有文字框的第一條水平分隔線，記錄其 Y 座標與右邊界（尾端 x1）
            valid_top = [sp for sp in top_spans if any(r.cy < sp[0] for r in rows)] or top_spans
            first_sep = valid_top[0]
            header_cut = float(first_sep[0])
            separator_x_end = float(first_sep[2])
        else:
            header_cut = height * self.cfg.header_max_y_ratio
            for r in rows:
                joined = "".join(b.text for b in r.boxes)
                if _TIME_RE.search(joined) and r.y1 < height * 0.45:
                    header_cut = max(header_cut, r.y1 + 2.0)
                    break

        footer_cut = float(bot_spans[0][0]) if bot_spans else height * self.cfg.footer_min_y_ratio
        return header_cut, footer_cut, separator_x_end

    def _parse_header(self, rows: List[StubRow], width: float, out: StubParse) -> None:
        """
        解析小傳票表頭（時間、桌號 table_no、傳票單號 ticket_no）：
        1. 逐列（由上而下）檢查時間框與右側數字框。
        2. 每一列若有多個數字框（如中間黑底反白『●伝票No●』被誤讀成 '80'，最右側為 '397'），
           只取該列最右側（cx 最大）的數字框，自動過濾中間標籤誤讀與左側文字雜訊（如 '枚7番号'）。
        3. 優先將「與時間框同一水平列」的最右側數字指定為 ticket_no（order_no），
           而位於時間列上方的右側數字指定為 table_no。
        """
        time_row_idx: Optional[int] = None
        row_rightmost_nums: List[Tuple[int, str]] = []

        for r_idx, r in enumerate(rows):
            # 若整列僅為外帶服務標記（如 'TO'），略過不計入號碼列
            joined_plain = "".join(b.text.strip() for b in r.boxes)
            if _SERVICE_MARK_PREFIX_RE.fullmatch(joined_plain):
                continue

            row_nums: List[Tuple[float, str]] = []
            for b in r.boxes:
                t = b.text.strip()
                if not t:
                    continue
                m = _TIME_RE.search(t)
                if m and out.time_str is None:
                    out.time_str = f"{int(m.group(1)):02d}:{m.group(2)}"
                    time_row_idx = r_idx
                    rest = (t[:m.start()] + " " + t[m.end():]).strip()
                    for tok in re.findall(r"\d+", rest):
                        row_nums.append((b.cx + 1.0, tok))
                    continue

                cleaned = _normalize_digits(t)
                if _PURE_DIGITS_RE.match(cleaned):
                    row_nums.append((b.cx, cleaned))
                elif b.cx >= width * 0.45:
                    # 僅當框位於右半部且以 2~4 碼數字結尾（如 'No.381'）時才擷取，排除左側 '枚7番号' 類雜訊
                    m_tail = re.search(r"(\d{2,4})$", t)
                    if m_tail:
                        row_nums.append((b.cx, m_tail.group(1)))

            if row_nums:
                # 同一水平列只取最右側（cx 最大）的數字，濾除中間黑底標籤（如 '80'）
                rightmost_tok = max(row_nums, key=lambda x: x[0])[1]
                row_rightmost_nums.append((r_idx, rightmost_tok))

        if not row_rightmost_nums:
            return

        # 優先規則：與時間（如 16:50 / 12:12）同一水平列的最右側數字即為 ticket_no (order_no)
        if time_row_idx is not None:
            same_row_hits = [tok for r_idx, tok in row_rightmost_nums if r_idx == time_row_idx]
            upper_row_hits = [tok for r_idx, tok in row_rightmost_nums if r_idx < time_row_idx]
            lower_row_hits = [tok for r_idx, tok in row_rightmost_nums if r_idx > time_row_idx]
            if same_row_hits:
                out.ticket_no = same_row_hits[-1]
                if upper_row_hits:
                    out.table_no = upper_row_hits[0]
                elif lower_row_hits:
                    out.table_no = lower_row_hits[0]
                return

        # 次要規則（時間漏讀或未同列時）：依垂直列由上而下，上列為 table_no、下列為 ticket_no
        if len(row_rightmost_nums) == 1:
            tok = row_rightmost_nums[0][1]
            if len(tok) >= 2:
                out.ticket_no = tok
            else:
                out.table_no = tok
        else:
            out.table_no = row_rightmost_nums[0][1]
            out.ticket_no = row_rightmost_nums[-1][1]

    def _is_noise_row(self, text: str) -> bool:
        if not text:
            return True
        if _TIME_RE.search(text):
            return True
        return any(kw in text for kw in self.cfg.noise_keywords)

    def _strip_item_prefix(self, text: str) -> str:
        """剝離品名前綴殘留的 1~4 碼不完整 POS 數字與外帶服務標記（如 '961TO 黒カレー' -> '黒カレー'）。"""
        s = re.sub(r"^[\d\s\-:：.・]+", "", text or "").strip()
        s = _SERVICE_MARK_PREFIX_RE.sub("", s).strip()
        s = re.sub(r"^[\s\-:：.・]+", "", s).strip()
        return s

    def _parse_items(
        self,
        rows: List[StubRow],
        width: float,
        img: Optional[np.ndarray],
        rec_pred,
        out: StubParse,
        separator_x_end: Optional[float] = None,
        is_flip: bool = False,
    ) -> None:
        img_w = int(img.shape[1]) if (img is not None and img.size > 0) else int(width)
        img_h = int(img.shape[0]) if (img is not None and img.size > 0) else 0

        all_box_hs = [b.h for r in rows for b in r.boxes if b.h > 4]
        ref_box_h = float(np.median(all_box_hs)) if all_box_hs else 24.0

        aligned_cx = compute_qty_column_x(
            rows,
            img_w,
            qty_min_x_ratio=self.cfg.qty_min_x_ratio,
            default_x_ratio=self.cfg.qty_default_x_ratio,
            separator_x_end=separator_x_end,
            ref_box_h=ref_box_h,
        )

        incomplete_rows: List[Tuple[StubRow, List[Box], Optional[Box], Optional[str], Optional[str], str, float]] = []

        for r in rows:
            body_boxes, qty_box = split_row_columns(
                r, width, self.cfg.qty_min_x_ratio, self.cfg.qty_max_width_ratio
            )
            body_text = " ".join(b.text.strip() for b in body_boxes if b.text.strip()).strip()
            if not body_text:
                continue

            pos_code: Optional[str] = None
            name_text = body_text

            # 1. 提取 4 碼 POS 代碼
            m = _POS_CODE_RE.search(body_text)
            if m and m.start() <= 6:
                pos_code = m.group(1)
                name_text = (body_text[:m.start()] + body_text[m.end():]).strip()
            elif body_boxes:
                first_norm = _normalize_digits(body_boxes[0].text)
                if re.fullmatch(r"\d{4}", first_norm):
                    pos_code = first_norm
                    name_text = " ".join(b.text.strip() for b in body_boxes[1:]).strip()

            if pos_code and pos_code in self.cfg.pos_confusion:
                pos_code = self.cfg.pos_confusion[pos_code]

            if pos_code is None and self._is_noise_row(body_text):
                continue

            clean_name = self._strip_item_prefix(name_text)
            if not clean_name and pos_code is None:
                continue

            matched_name, score = self._match_name(clean_name or name_text)

            # 2. 若 4 碼 POS 漏讀（如 '961TO 黒カレー' 僅讀出 3 碼），用比對出的標準品名反查 4 碼 POS
            if not pos_code and matched_name and matched_name in self.cfg.pos_lookup:
                pos_code = self.cfg.pos_lookup[matched_name]

            if matched_name is None and pos_code is None:
                continue

            # 3. 讀取數量（先由 CV 墨跡或第一條水平線右尾端對齊定位唯一 cx，再呼叫 1 次 predict_quantity）
            qty, qty_src = self._read_qty(
                r, body_boxes, qty_box, img, rec_pred,
                aligned_cx=aligned_cx, img_w=img_w, img_h=img_h,
                separator_x_end=separator_x_end,
                is_flip=is_flip,
                has_valid_item=bool(matched_name or pos_code),
            )

            if matched_name or pos_code:
                out.items.append(
                    StubItem(
                        pos_code=pos_code,
                        name=matched_name,
                        raw_name=clean_name or body_text,
                        quantity=qty,
                        score=score,
                        qty_source=qty_src,
                    )
                )
            else:
                incomplete_rows.append((r, body_boxes, qty_box, pos_code, matched_name, clean_name or body_text, score))

        # 挽救不完整列（若有任何漏網之魚）
        for r, body_boxes, qty_box, pos_code, matched_name, raw_name, score in incomplete_rows:
            if not matched_name and len(raw_name) >= 2:
                matched_name, score = self._match_name(raw_name)
            if not pos_code and matched_name and matched_name in self.cfg.pos_lookup:
                pos_code = self.cfg.pos_lookup[matched_name]
            if matched_name or pos_code:
                qty, qty_src = self._read_qty(
                    r, body_boxes, qty_box, img, rec_pred,
                    aligned_cx=aligned_cx, img_w=img_w, img_h=img_h,
                    separator_x_end=separator_x_end,
                    is_flip=is_flip,
                    has_valid_item=True,
                )
                out.items.append(
                    StubItem(
                        pos_code=pos_code,
                        name=matched_name,
                        raw_name=raw_name,
                        quantity=qty,
                        score=score,
                        qty_source=qty_src,
                    )
                )

    def _match_name(self, raw_name: str) -> Tuple[Optional[str], float]:
        if not raw_name or not self._folded_keys:
            return None, 0.0
        query = self.folded.fold(normalize_text(raw_name, ignore_space=True))
        if not query:
            return None, 0.0
        if query in self._folded_map:
            return self._folded_map[query], 100.0
        hit = process.extractOne(
            query,
            self._folded_keys,
            scorer=fuzz.WRatio,
            score_cutoff=self.cfg.fuzzy_cutoff,
        )
        if not hit:
            return None, 0.0
        key, score, _ = hit
        return self._folded_map[key], float(score)

    def _read_qty(
        self,
        row: StubRow,
        body_boxes: List[Box],
        qty_box: Optional[Box],
        img: Optional[np.ndarray],
        rec_pred,
        aligned_cx: float = 0.0,
        img_w: int = 0,
        img_h: int = 0,
        separator_x_end: Optional[float] = None,
        is_flip: bool = False,
        has_valid_item: bool = False,
    ) -> Tuple[Optional[int], str]:
        # Step 1. 直接從 det 數量框的 rec 文字解析
        if qty_box is not None:
            q = self._clean_qty_str(qty_box.text)
            if q is not None:
                return q, "det"

        # Step 2. 先由 CV 墨跡偵測或水平線右尾端垂直對齊決定唯一的 target_cx，再呼叫 1 次 predict_quantity
        ref_boxes = body_boxes if body_boxes else row.boxes
        by0 = min(b.y0 for b in ref_boxes) if ref_boxes else row.y0
        by1 = max(b.y1 for b in ref_boxes) if ref_boxes else row.y1
        body_cy = (by0 + by1) / 2.0
        body_h = max(16.0, float(by1 - by0))
        size_w = max(body_h * 1.2, 26.0)

        target_cx: Optional[float] = None
        src_tag = "aligned_col"
        fallback_rect: Optional[Tuple[int, int, int, int]] = None

        if qty_box is not None:
            target_cx = float(qty_box.cx)
            src_tag = "rec_box"
            fallback_rect = (int(qty_box.x0), int(qty_box.y0), int(qty_box.x1), int(qty_box.y1))
        else:
            # 2a. 在該列右側水平帶執行 CV 墨跡團塊偵測（右界不超過第一條水平線尾端 separator_x_end + 8）
            if img is not None and img.size > 0:
                h_img, w_img = img.shape[:2]
                pad_y = max(2, int(row.h * 0.15))
                ry0 = max(0, int(round(by0)) - pad_y)
                ry1 = min(h_img, int(round(by1)) + pad_y)
                band = img[ry0:ry1, 0:w_img]
                max_x_bound = int(round(separator_x_end)) + 8 if separator_x_end else None
                ink = find_rightmost_ink(
                    band,
                    min_x_ratio=max(0.65, self.cfg.qty_min_x_ratio),
                    max_x=max_x_bound,
                )
                if ink is not None:
                    ix0, iy0, ix1, iy1 = ink
                    target_cx = (ix0 + ix1) / 2.0
                    src_tag = "cv_ink"

            # 2b. 若 CV 未檢出墨跡，採用以第一條水平線右尾端為基準算出的垂直對齊線 aligned_cx
            if target_cx is None and aligned_cx > 0:
                target_cx = float(aligned_cx)
                src_tag = "aligned_col"

            if target_cx is not None and img_w > 0 and img_h > 0:
                fallback_rect = build_aligned_qty_rect(by0, by1, target_cx, img_w, img_h)
                if fallback_rect is not None and fallback_rect not in self._cur_fallback_rects:
                    self._cur_fallback_rects.append(fallback_rect)

        if (
            target_cx is not None
            and rec_pred is not None
            and img is not None
            and img.size > 0
        ):
            text, _ = self._predict_qty_timed(
                rec_pred,
                img,
                cx=target_cx - QTY_REC_X_OFFSET,
                cy=body_cy,
                h=body_h,
                w=size_w,
                is_flip=is_flip,
            )
            q = self._clean_qty_str(text)
            if q is not None:
                if (
                    qty_box is not None
                    and fallback_rect is not None
                    and fallback_rect not in self._cur_fallback_rects
                ):
                    self._cur_fallback_rects.append(fallback_rect)
                return q, src_tag

        # Step 3. 若該列已確認為有效品項，數量欄漏檢時保底為 1
        if has_valid_item:
            return 1, "default_1"
        return None, "missing"
