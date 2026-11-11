"""
stub_layout.py
==============
客戶聯（stub / small_order_ticket）幾何排版：文字框過濾、列分群、欄位切分與裁切輔助。
OpenCV 視覺補強偵測（水平分隔線、右側漏檢墨跡、垂直對齊數量欄）由 logic.stub_cv_det 提供。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import numpy as np

from logic.yoshinoya.stub_cv_det import (
    find_horizontal_lines,
    find_rightmost_ink,
    compute_qty_column_x,
    build_aligned_qty_rect,
)


@dataclass
class Box:
    idx: int
    poly: np.ndarray
    text: str
    conf: float
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    angle_deg: float = 0.0

    def __post_init__(self):
        pts = np.asarray(self.poly, dtype=np.float32).reshape(-1, 2)
        self.poly = pts
        self.x0 = float(pts[:, 0].min())
        self.y0 = float(pts[:, 1].min())
        self.x1 = float(pts[:, 0].max())
        self.y1 = float(pts[:, 1].max())
        self.cx = 0.5 * (self.x0 + self.x1)
        self.cy = 0.5 * (self.y0 + self.y1)
        self.w = self.x1 - self.x0
        self.h = self.y1 - self.y0
        if len(pts) >= 2:
            dx = float(pts[1, 0] - pts[0, 0])
            dy = float(pts[1, 1] - pts[0, 1])
            ang = math.degrees(math.atan2(dy, dx))
            if ang > 90:
                ang -= 180
            elif ang < -90:
                ang += 180
            self.angle_deg = ang


@dataclass
class StubRow:
    boxes: List[Box] = field(default_factory=list)

    @property
    def cy(self) -> float:
        return sum(b.cy for b in self.boxes) / len(self.boxes) if self.boxes else 0.0

    @property
    def y0(self) -> float:
        return min((b.y0 for b in self.boxes), default=0.0)

    @property
    def y1(self) -> float:
        return max((b.y1 for b in self.boxes), default=0.0)

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def x0(self) -> float:
        return min((b.x0 for b in self.boxes), default=0.0)

    @property
    def x1(self) -> float:
        return max((b.x1 for b in self.boxes), default=0.0)


def to_boxes(dt_boxes: Sequence, rec_res: Sequence[Tuple[str, float]]) -> List[Box]:
    if dt_boxes is None or rec_res is None:
        return []
    n = min(len(dt_boxes), len(rec_res))
    out: List[Box] = []
    for i in range(n):
        text, conf = rec_res[i]
        out.append(Box(idx=i, poly=dt_boxes[i], text=text or "", conf=float(conf)))
    return out


def filter_Rotated_boxes(boxes: List[Box], max_tilt_deg: float = 25.0) -> List[Box]:
    """過濾垂直側欄或嚴重歪斜的文字框。"""
    kept: List[Box] = []
    for b in boxes:
        if abs(b.angle_deg) > max_tilt_deg and b.h > b.w * 1.6 and len(b.text.strip()) >= 2:
            continue
        kept.append(b)
    return kept


def group_rows(boxes: List[Box], y_overlap_ratio: float = 0.45) -> List[StubRow]:
    if not boxes:
        return []
    ordered = sorted(boxes, key=lambda b: (b.cy, b.x0))
    rows: List[StubRow] = []
    for b in ordered:
        placed = False
        for row in rows:
            ref_h = max(1.0, min(b.h, row.h))
            overlap = min(b.y1, row.y1) - max(b.y0, row.y0)
            cy_diff = abs(b.cy - row.cy)
            if overlap / ref_h >= y_overlap_ratio or cy_diff <= ref_h * 0.55:
                row.boxes.append(b)
                placed = True
                break
        if not placed:
            rows.append(StubRow(boxes=[b]))

    for row in rows:
        row.boxes.sort(key=lambda b: b.x0)
    rows.sort(key=lambda r: r.cy)
    return rows


def split_row_columns(
    row: StubRow,
    ticket_width: float,
    qty_min_x_ratio: float = 0.55,
    qty_max_width_ratio: float = 0.25,
) -> Tuple[List[Box], Optional[Box]]:
    """將一列拆為 (左側品名/POS 框清單, 右側數量框)。"""
    if not row.boxes or ticket_width <= 0:
        return list(row.boxes), None
    ordered = sorted(row.boxes, key=lambda b: b.x0)
    if len(ordered) == 1:
        b = ordered[0]
        if b.cx >= ticket_width * qty_min_x_ratio and b.w <= ticket_width * qty_max_width_ratio:
            return [], b
        return ordered, None

    rightmost = ordered[-1]
    if (
        rightmost.cx >= ticket_width * qty_min_x_ratio
        and rightmost.w <= ticket_width * qty_max_width_ratio
    ):
        return ordered[:-1], rightmost
    return ordered, None


def crop_box(img: np.ndarray, box: Box, pad: int = 2) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    x0 = max(0, int(round(box.x0)) - pad)
    y0 = max(0, int(round(box.y0)) - pad)
    x1 = min(w, int(round(box.x1)) + pad)
    y1 = min(h, int(round(box.y1)) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]


def crop_rect(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    pad: int = 2,
) -> Optional[np.ndarray]:
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    x0, y0, x1, y1 = rect
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img[y0:y1, x0:x1]
