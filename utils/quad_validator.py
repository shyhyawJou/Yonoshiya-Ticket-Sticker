"""
quad_validator.py

用於偵測與修復 det 模型輸出的畸形四邊形 bbox（菱形 / 漏斗形，
即四點順序錯亂 self-intersecting quadrilateral）。

使用場景：
    在 OrderParser.parse() 裡，corrected_dt_boxes = skew.correct_boxes(flip_dt)
    這一步之後（或甚至在 skew 校正之前），先跑一次 validate_and_fix_boxes()，
    確保後續所有 cy / x_coords / h 的幾何運算都建立在正確排序的四點上。
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple

Point = Tuple[float, float]
Quad = List[Point]  # 4 個點，理想順序為 [top-left, top-right, bottom-right, bottom-left]


def is_convex_ordered(quad: Quad) -> bool:
    """
    檢查四點是否按照一致方向（順時針或逆時針）排列且不自相交。
    做法：依序取三個連續點算邊向量的叉積，若四個叉積符號不一致，
    代表這是一個菱形/漏斗形（bowtie），順序有問題。
    """
    pts = np.asarray(quad, dtype=np.float64)
    if pts.shape != (4, 2):
        return False

    signs = []
    for i in range(4):
        p0, p1, p2 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross == 0:
            # 三點共線，視為退化框，交由外層面積/寬高檢查處理
            continue
        signs.append(np.sign(cross))

    if not signs:
        return False
    return len(set(signs)) == 1


def order_points_tl_tr_br_bl(quad: Quad) -> Quad:
    """
    不依賴原始輸入順序，直接用座標 sum / diff 規則重新排序成
    [top-left, top-right, bottom-right, bottom-left]。

    對於文字 OCR 框（通常只有小角度旋轉，不會是 45 度以上的極端旋轉）
    這個規則非常穩定：
        - top-left     : x + y 最小
        - bottom-right : x + y 最大
        - top-right    : y - x 最小 (即 x - y 最大)
        - bottom-left  : y - x 最大
    """
    pts = np.asarray(quad, dtype=np.float64)
    s = pts.sum(axis=1)          # x + y
    diff = pts[:, 1] - pts[:, 0]  # y - x

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return [tuple(tl), tuple(tr), tuple(br), tuple(bl)]


def quad_area(quad: Quad) -> float:
    """Shoelace 公式算面積，可用來過濾面積過小/退化的框。"""
    pts = np.asarray(quad, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def validate_and_fix_box(
    quad: Quad,
    min_area: float = 4.0,
) -> Tuple[Quad, bool, bool]:
    """
    對單個框做驗證與修復。

    Returns:
        fixed_quad: 修復後的四點（若原本就正確則原樣返回，只是保證型別一致）
        was_malformed: 原始框是否偵測到順序錯亂 / 自相交
        is_degenerate: 修復後仍然面積過小或無效（建議直接丟棄此框，
                       不要再拿去算 cy / h，會污染整列的合併結果）
    """
    was_malformed = not is_convex_ordered(quad)

    fixed = order_points_tl_tr_br_bl(quad) if was_malformed else list(quad)

    area = quad_area(fixed)
    is_degenerate = area < min_area

    return fixed, was_malformed, is_degenerate


def box_angle_deg(quad: Quad) -> float:
    """
    計算框的主軸角度（取 top 邊 quad[1] - quad[0] 的方向）。
    因為矩形轉 90 度看起來一樣，角度統一映射到 [-45, 45] 區間，
    這樣「橫的」跟「豎的」正常文字框都會落在接近 0 度附近，方便比較。
    """
    pts = np.asarray(quad, dtype=np.float64)
    dx, dy = pts[1] - pts[0]
    angle = np.degrees(np.arctan2(dy, dx))
    angle = angle % 90.0
    if angle > 45.0:
        angle -= 90.0
    return angle


def find_angle_outliers(
    quads: List[Quad],
    max_dev_deg: float = 20.0,
) -> List[int]:
    """
    找出角度明顯偏離同一批框「主流角度」的框 index。

    用 median（而非 mean）當基準，避免少數異常框把平均值拉走，
    導致真正的異常值反而看起來正常。

    這個檢查抓的是「四點順序完全正確、但形狀本身是菱形/怪異四邊形」
    的情況 —— is_convex_ordered() 對這種框無能為力，因為菱形本身
    在數學上就是合法的凸四邊形。
    """
    if len(quads) < 3:
        # 樣本太少，median 沒有統計意義，不做這項檢查
        return []

    angles = [box_angle_deg(q) for q in quads]
    median_angle = float(np.median(angles))

    outlier_indices = []
    for i, a in enumerate(angles):
        # 處理角度環繞問題（例如 median=44, a=-44 其實只差 2 度）
        dev = abs(a - median_angle)
        dev = min(dev, 90.0 - dev)
        if dev > max_dev_deg:
            outlier_indices.append(i)

    return outlier_indices


def validate_and_fix_boxes(
    quads: List[Quad],
    min_area: float = 4.0,
    log_fn=None,
) -> Tuple[List[Quad], List[int]]:
    """
    批次處理版本，回傳 (修復後的框列表, 被判定為退化/建議丟棄的 index 列表)。

    log_fn: 可傳入 logger.warning 之類的函式，用來記錄哪些框被修正/丟棄，
            方便你在 debug 圖上把這些框標成紅色，一眼看出哪張單子出問題。
    """
    fixed_boxes = []
    bad_indices = []

    for i, q in enumerate(quads):
        fixed, was_malformed, is_degenerate = validate_and_fix_box(q, min_area=min_area)
        fixed_boxes.append(fixed)

        if was_malformed and log_fn:
            log_fn(f"[bbox 順序修復] index={i} 原始四點順序錯亂(菱形/漏斗)，已重新排序: {q} -> {fixed}")

        if is_degenerate:
            bad_indices.append(i)
            if log_fn:
                log_fn(f"[bbox 退化] index={i} 面積過小/無效，建議丟棄: {fixed}")

    return fixed_boxes, bad_indices