"""
geometry.py
===========
純幾何運算工具，完全不依賴 tray / ticket / sticker 等業務概念。

這裡的所有函式輸入都是 8 點多邊形 (PolygonXYXY) 或由此衍生的座標，
不應該在這個檔案裡加入任何跟 tray 生命週期、狀態機、OCR 有關的邏輯。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from shapely.geometry import Point, Polygon

PolygonXYXY = Tuple[float, float, float, float, float, float, float, float]
ObbXYWHR = Tuple[float, float, float, float, float]


def _polygon_from_tuple(p: PolygonXYXY) -> Polygon:
    """將 8點元組 (4頂點) 轉換為 Shapely Polygon 物件"""
    return Polygon([(p[i], p[i + 1]) for i in range(0, 8, 2)])


def polygon_area(p: PolygonXYXY) -> float:
    """計算 4頂點多邊形 的面積"""
    if not p:
        return 0.0
    return _polygon_from_tuple(p).area


def inter_area_poly_poly(p1: PolygonXYXY, p2: PolygonXYXY) -> float:
    """計算兩個 4頂點多邊形 之間的交集面積"""
    if not p1 or not p2:
        return 0.0
    poly1 = _polygon_from_tuple(p1)
    poly2 = _polygon_from_tuple(p2)
    return poly1.intersection(poly2).area


def iou_poly_poly(p1: PolygonXYXY, p2: PolygonXYXY) -> float:
    """計算兩個 4頂點多邊形 之間的 IOU"""
    ia = inter_area_poly_poly(p1, p2)
    if ia <= 0.0:
        return 0.0

    poly1_a = polygon_area(p1)
    poly2_a = polygon_area(p2)
    union_a = poly1_a + poly2_a - ia
    return ia / (union_a + 1e-9) if union_a > 0 else 0.0


def get_polygon_centroid(p: PolygonXYXY) -> Tuple[float, float]:
    """計算 4頂點多邊形 的中心點 (質心)"""
    poly = _polygon_from_tuple(p)
    return poly.centroid.x, poly.centroid.y


def is_center_in_polygon(center: Tuple[float, float], p: PolygonXYXY) -> bool:
    """檢查一個點是否在 4頂點多邊形 內部"""
    point = Point(center)
    poly = _polygon_from_tuple(p)
    return poly.contains(point)


def shrink_rect(r: PolygonXYXY, factor: float) -> PolygonXYXY:
    """將矩形依中心點等比例縮小，factor 為縮小比例 (0~1)"""
    pts = np.array(r).reshape(4, 2)
    centroid = np.mean(pts, axis=0)
    scale = 1.0 - factor
    shrunk_pts = centroid + (pts - centroid) * scale
    return tuple(shrunk_pts.flatten())