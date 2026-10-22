"""
ticket_layout.py
================
票據版面分析：把 OCR 的 (文字, 分數, 四點框) 轉成「每一列的品名 + 該列數量」。

刻意做成純幾何 / 純字串邏輯，不依賴任何模型或硬體，因此可以在開發機
完整單元測試（見 unit_test/test_ticket_layout.py）。

為什麼需要這一層
----------------
舊 OrderParser 用「與列中心 Y 最近的純數字框」當數量，完全沒有 X 方向的
欄位限制，所以同一列的「金額」欄數字（100 / 0）或訂單編號都可能被當成
數量。實測 images/order_ticket 的票據版面固定為：

    | (窄空白) |      品名      | 数量 | 金額 |

以表格寬為單位，欄界固定落在 [0, 0.105, 0.576, 0.680, 0.994]
（99 張票據實測，見 images/tool/README.md）。也就是說「数量」欄只佔
表格寬的 0.104，而「金額」欄中心在 0.837 —— 兩者相距 0.209 表格寬。
只要用「数量」表頭自己的 X 範圍當閘門（左右各外擴 0.3 倍表頭寬，
閘門半寬約 0.073 表格寬），金額欄就會被穩定排除。

數量取不到時一律回報 None（未確認），不預設 1。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from logic.text_utils import MatchDecision, match_restricted, normalize_text

# 票據編號：00821-01。保留前導零與連字號，整串輸出。
# 嚴格式優先（實際票據就是 5 位-2 位），避免抓到日期 2026-08-26 的前半段。
ORDER_NUMBER_STRICT_RE = re.compile(r"(?<!\d)(\d{5})\s*[-‐-―ー]\s*(\d{2})(?!\d)")
ORDER_NUMBER_LOOSE_RE = re.compile(r"(?<!\d)(\d{4,6})\s*[-‐-―ー]\s*(\d{1,3})(?!\d)")
# 日期樣式，用來排除；票據抬頭印 2026-08-26，會誤中寬鬆式。
DATE_LIKE_RE = re.compile(r"\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}")
# 純數量：1~99。刻意不接受 0 與 3 位數，避免抓到金額 (100) 與 0。
QUANTITY_RE = re.compile(r"^([1-9]\d?)$")
# 前綴數量：吉野家版面把數量印在品名左邊，OCR 常合併成 "2味噌ラーメン"。
LEADING_QTY_RE = re.compile(r"^([1-9]\d?)\s*(\D.*)$")
# 「数量」表頭常見誤辨識
QTY_HEADER_ALIASES = ("数量", "数", "枚", "數量", "牧量", "教量", "数昌", "数圭")
# 表格表頭列的字樣（品名 / 数量 / 金額 及常見誤辨識）。
# 整列只要命中其中一個，就當表頭列排除，不當品項也不報成「無法辨識的列」。
TABLE_HEADER_TOKENS = QTY_HEADER_ALIASES + (
    "品名", "品各", "品石", "品名称",
    "金額", "金顔", "金額円", "金梁", "全額",
)


@dataclass
class TextItem:
    """一個 OCR 文字框。quad 必須是已擺正（去斜）的座標。"""

    text: str
    score: float
    quad: np.ndarray
    orig_quad: Optional[np.ndarray] = None
    index: int = -1

    x0: float = field(init=False)
    x1: float = field(init=False)
    y0: float = field(init=False)
    y1: float = field(init=False)
    cx: float = field(init=False)
    cy: float = field(init=False)
    w: float = field(init=False)
    h: float = field(init=False)

    def __post_init__(self):
        q = np.asarray(self.quad, dtype=float).reshape(-1, 2)
        self.x0, self.x1 = float(q[:, 0].min()), float(q[:, 0].max())
        self.y0, self.y1 = float(q[:, 1].min()), float(q[:, 1].max())
        self.cx = (self.x0 + self.x1) / 2.0
        self.cy = (self.y0 + self.y1) / 2.0
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        if self.orig_quad is None:
            self.orig_quad = self.quad

    @property
    def clean(self) -> str:
        return self.text.strip()


@dataclass
class TicketRow:
    """一列的解析結果。quantity 為 None 代表未確認，呼叫端不得當成 1。"""

    row_index: int
    cy: float
    height: float
    name: Optional[str]
    name_raw: str
    name_score: float
    name_margin: float
    name_reason: str
    quantity: Optional[int]
    quantity_raw: Optional[str]
    quantity_source: str  # "qty_column" | "rec_crop" | "unconfirmed"

    @property
    def is_confirmed(self) -> bool:
        return self.name is not None and self.quantity is not None


@dataclass
class QtyColumn:
    """「数量」欄的水平閘門。"""

    x0: float
    x1: float
    header_cx: float
    header_cy: float
    header_w: float
    source: str

    def contains(self, cx: float) -> bool:
        return self.x0 <= cx <= self.x1


# ---------------------------------------------------------------------------
# 幾何
# ---------------------------------------------------------------------------

def build_items(
    dt_boxes: Sequence,
    rec_res: Sequence[Tuple[str, float]],
    orig_boxes: Optional[Sequence] = None,
    min_score: float = 0.0,
) -> List[TextItem]:
    """把 dt_boxes + rec_res 打包成 TextItem，長度不一致時取較短者。"""
    items: List[TextItem] = []
    n = min(len(dt_boxes), len(rec_res))
    for i in range(n):
        text, score = rec_res[i][0], float(rec_res[i][1])
        if score < min_score:
            continue
        oq = orig_boxes[i] if orig_boxes is not None and i < len(orig_boxes) else None
        items.append(TextItem(text=text, score=score, quad=dt_boxes[i],
                              orig_quad=oq, index=i))
    return items


def vertical_overlap_ratio(a: TextItem, b: TextItem) -> float:
    """兩框在 Y 軸的重疊量，除以較矮者的高度。"""
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    if inter <= 0:
        return 0.0
    shorter = max(1e-6, min(a.h, b.h))
    return inter / shorter


def group_rows(items: Sequence[TextItem], min_overlap: float = 0.45) -> List[List[TextItem]]:
    """依 Y 軸重疊率分列。

    比舊做法（|cy - avg_cy| < avg_h * 0.5）穩健：重疊率對字高差異
    （例如同列的大字品名與小字數量）不敏感。
    """
    ordered = sorted(items, key=lambda it: it.cy)
    rows: List[List[TextItem]] = []
    for it in ordered:
        placed = False
        for row in rows:
            if any(vertical_overlap_ratio(it, other) >= min_overlap for other in row):
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])
    for row in rows:
        row.sort(key=lambda t: t.cx)
    rows.sort(key=lambda r: np.mean([t.cy for t in r]))
    return rows


def find_qty_column(
    items: Sequence[TextItem],
    expand: float = 0.30,
    min_header_score: float = 55.0,
) -> Optional[QtyColumn]:
    """從「数量」表頭取出數量欄的水平閘門。

    找不到表頭時回傳 None —— 呼叫端應把該票所有數量標為未確認，
    而不是改用其他猜測方式。
    """
    best = None
    for it in items:
        clean = normalize_text(it.clean, ignore_space=True)
        if not clean or len(clean) > 4:
            continue
        d = match_restricted(clean, list(QTY_HEADER_ALIASES),
                             min_score=min_header_score, min_margin=0.0)
        if not d.accepted:
            continue
        if best is None or d.score > best[0]:
            best = (d.score, it)
    if best is None:
        return None
    it = best[1]
    pad = expand * max(it.w, 1.0)
    return QtyColumn(x0=it.x0 - pad, x1=it.x1 + pad,
                     header_cx=it.cx, header_cy=it.cy,
                     header_w=it.w, source="header")


def find_order_number(items: Sequence[TextItem]) -> Tuple[Optional[str], Optional[TextItem]]:
    """找票據編號，整串輸出並保留前導零。

    舊做法是 re.search(r'\\d{3}$')，只取結尾三位數，會把 '00821-01' 變成
    '-01' 抓不到、或抓到別的三位數。這裡要求完整的 數字-數字 結構，
    並選最靠上方（cy 最小）的一個，因為票據頂端與底部都會印編號。
    """
    strict, loose = [], []
    for it in items:
        txt = it.clean.replace(" ", "")
        # 抬頭的日期 2026-08-26 會誤中寬鬆式，先整段排除
        if DATE_LIKE_RE.search(txt):
            continue
        m = ORDER_NUMBER_STRICT_RE.search(txt)
        if m:
            strict.append((it.cy, f"{m.group(1)}-{m.group(2)}", it))
            continue
        m = ORDER_NUMBER_LOOSE_RE.search(txt)
        if m:
            loose.append((it.cy, f"{m.group(1)}-{m.group(2)}", it))
    hits = strict or loose
    if not hits:
        return None, None
    # 票據頂端與底部都會印編號，取最靠上方的那個
    hits.sort(key=lambda t: t[0])
    return hits[0][1], hits[0][2]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def is_table_header_row(row: Sequence[TextItem], min_score: float = 70.0) -> bool:
    """整列是否為表格表頭（品名 / 数量 / 金額）。"""
    for t in row:
        clean = normalize_text(t.clean, ignore_space=True)
        if not clean or len(clean) > 4:
            continue
        d = match_restricted(clean, list(TABLE_HEADER_TOKENS),
                             min_score=min_score, min_margin=0.0)
        if d.accepted:
            return True
    return False


def parse_rows(
    items: Sequence[TextItem],
    vocab: Sequence[str],
    qty_col: Optional[QtyColumn],
    header_names: Sequence[str] = ("数量",),
    min_score: float = 62.0,
    min_margin: float = 8.0,
    row_min_overlap: float = 0.45,
    exclude_above_cy: Optional[float] = None,
    allow_leading_digit: bool = True,
    names_with_digits: Sequence[str] = (),
) -> List[TicketRow]:
    """把文字框分列、逐列比對品名、逐列在數量欄內取數量。

    exclude_above_cy: 表格開始前（票據抬頭、日期、編號）的區域一律不當品名列。

    數量來源優先序（越前面越可信）：
        1. "qty_column"   落在「数量」欄 X 閘門內的獨立數字框。
        2. "leading_digit" 品名文字前綴的數字（吉野家版面：數量印在品名左邊，
           OCR 會合併成 "2味噌ラーメン"）。只在沒有數量欄資訊時採用，
           且品名本身不得含數字（例如 "18切牛舌" 不適用）。
        3. 呼叫端的裁圖數字辨識 fallback（"rec_crop"）。
        4. 都取不到 -> None，quantity_source = "unconfirmed"。
    """
    names_with_digits = set(names_with_digits)
    name_vocab = [v for v in vocab if v not in header_names]

    usable = [it for it in items
              if exclude_above_cy is None or it.cy > exclude_above_cy]
    rows = group_rows(usable, min_overlap=row_min_overlap)

    out: List[TicketRow] = []
    for ridx, row in enumerate(rows):
        row_cy = float(np.mean([t.cy for t in row]))
        row_h = float(np.mean([t.h for t in row]))

        # 品名欄 = 數量欄左側的文字；沒有數量欄資訊時退回整列
        if qty_col is not None:
            name_parts = [t for t in row if t.cx < qty_col.x0]
            qty_parts = [t for t in row if qty_col.contains(t.cx)]
        else:
            name_parts = list(row)
            qty_parts = []

        if is_table_header_row(row):
            continue

        merged_raw = "".join(t.clean for t in sorted(name_parts, key=lambda t: t.cx))
        if not merged_raw:
            continue

        decision: MatchDecision = match_restricted(
            merged_raw, name_vocab, min_score=min_score, min_margin=min_margin
        )

        # 若整列文字以數字開頭，去掉前綴數字後可能配得更好
        lead_m = LEADING_QTY_RE.match(normalize_text(merged_raw, ignore_space=True))
        leading_digit = int(lead_m.group(1)) if lead_m else None
        if lead_m:
            alt = match_restricted(lead_m.group(2), name_vocab,
                                   min_score=min_score, min_margin=min_margin)
            if (alt.accepted and not decision.accepted) or alt.score > decision.score:
                decision = alt

        # 表頭列（品名/数量/金額）不是品項
        if decision.accepted and decision.name in header_names:
            continue

        quantity: Optional[int] = None
        quantity_raw: Optional[str] = None
        source = "unconfirmed"
        for t in sorted(qty_parts, key=lambda t: abs(t.cy - row_cy)):
            m = QUANTITY_RE.match(normalize_text(t.clean, ignore_space=True))
            if m:
                quantity = int(m.group(1))
                quantity_raw = t.clean
                source = "qty_column"
                break

        # 沒有數量欄資訊時才採用前綴數字（見上方優先序說明）
        if (quantity is None and allow_leading_digit and qty_col is None
                and leading_digit is not None
                and decision.accepted
                and decision.name not in names_with_digits):
            quantity = leading_digit
            quantity_raw = str(leading_digit)
            source = "leading_digit"

        out.append(TicketRow(
            row_index=ridx,
            cy=row_cy,
            height=row_h,
            name=decision.name if decision.accepted else None,
            name_raw=merged_raw,
            name_score=decision.score,
            name_margin=decision.margin,
            name_reason=decision.reason,
            quantity=quantity,
            quantity_raw=quantity_raw,
            quantity_source=source,
        ))
    return out


def to_legacy_items(rows: Sequence[TicketRow]) -> List[Tuple[str, Optional[int]]]:
    """轉成舊的 (name, quantity) 列表；未確認數量為 None。

    同一品名出現在不同列時各自保留一筆，不合併。
    """
    return [(r.name, r.quantity) for r in rows if r.name is not None]
