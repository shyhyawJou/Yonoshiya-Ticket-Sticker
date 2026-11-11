"""
stub_cv_det.py
==============
客戶聯（stub / small_order_ticket）專用的 OpenCV 補強偵測模組。

包含兩個階段的 CV 補強：
1. MMR 物件偵測階段（單據邊界緊貼修正與漏檢候選框推算）：
   - estimate_document_roi       : 利用亮度分佈與直線角度緊密鎖定白紙單據 ROI (cx, cy, w, h, r)
   - refine_stub_detection_roi   : 對已偵測到的 stub 框進行白紙邊界緊貼修正
   - generate_stub_candidates    : 當僅偵測到主傳票（order_ticket）而漏檢 stub 時，推算頂部候選框
   - enhance_stub_detections     : MMR 階段統一呼叫入口（支援 Detection 物件與 list 兩種格式）

2. OCR 文字/數量欄偵測階段（分隔線、漏檢細長數字墨跡、垂直對齊數量欄補框）：
   - find_horizontal_lines       : 形態學水平開運算尋找表頭與品項區之間的水平分隔線 Y 座標
   - find_rightmost_ink          : 連通元件分析尋找品項列右側被 OCR det 漏框的細長數字墨跡團塊
   - compute_qty_column_x        : 統計各列右側框 X 中心中位數，決定整張單據的數量欄垂直對齊基準線
   - build_aligned_qty_rect      : 依列高與數量欄垂直基準線產生正方形補救框（fallback_rect）
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import cv2
import numpy as np

from utils.schema import Detection


def xywhr_to_corners(cx: float, cy: float, w: float, h: float, r: float) -> np.ndarray:
    """將旋轉框 (cx, cy, w, h, r) 轉為 shape=(4, 2) 的四角點座標陣列。"""
    cos_v, sin_v = float(np.cos(r)), float(np.sin(r))
    vx1, vy1 = (w / 2.0) * cos_v, (w / 2.0) * sin_v
    vx2, vy2 = -(h / 2.0) * sin_v, (h / 2.0) * cos_v
    return np.array(
        [
            [cx + vx1 + vx2, cy + vy1 + vy2],
            [cx + vx1 - vx2, cy + vy1 - vy2],
            [cx - vx1 - vx2, cy - vy1 - vy2],
            [cx - vx1 + vx2, cy - vy1 + vy2],
        ],
        dtype=np.float32,
    )


def xywhr_to_xyxy8(
    cx: float, cy: float, w: float, h: float, r: float
) -> Tuple[float, float, float, float, float, float, float, float]:
    """將旋轉框 (cx, cy, w, h, r) 轉為 Detection.xyxy 所需的 8-tuple 四角點座標。"""
    cos_v, sin_v = float(np.cos(r)), float(np.sin(r))
    vx1, vy1 = (w / 2.0) * cos_v, (w / 2.0) * sin_v
    vx2, vy2 = -(h / 2.0) * sin_v, (h / 2.0) * cos_v
    return (
        float(cx + vx1 + vx2), float(cy + vy1 + vy2),
        float(cx + vx1 - vx2), float(cy + vy1 - vy2),
        float(cx - vx1 - vx2), float(cy - vy1 - vy2),
        float(cx - vx1 + vx2), float(cy - vy1 + vy2),
    )


# ==============================================================================
# 一、MMR 階段 CV 補強：白紙單據邊界緊貼修正 & 傳票頂部候選框推算
# ==============================================================================
def estimate_document_roi(img: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    針對包含單據的影像，自動緊貼鎖定白紙單據邊界，剔除黑底托盤、木紋桌面與底部反光。
    回傳旋轉框格式：(cx, cy, w, h, r)，其中 r 為弧度。
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. 取得初步白紙輪廓以估計單據傾斜角度 r
    _, thresh = cv2.threshold(blur, 155, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    r_rad = 0.0
    if contours:
        valid_cnts = [c for c in contours if cv2.contourArea(c) > (w * h * 0.03)]
        target_cnt = max(valid_cnts, key=cv2.contourArea) if valid_cnts else max(contours, key=cv2.contourArea)
        if cv2.contourArea(target_cnt) > (w * h * 0.01):
            rect = cv2.minAreaRect(target_cnt)
            (_, _), (rw, rh), angle = rect
            if rw > rh:
                rw, rh = rh, rw
                angle += 90.0
            if angle > 45.0:
                angle -= 90.0
            elif angle < -45.0:
                angle += 90.0

            # 用直線偵測微調傾斜角（比外輪廓更準確）
            edges = cv2.Canny(blur, 50, 150)
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=60, minLineLength=w // 6, maxLineGap=20
            )
            angles = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    if -15.0 <= deg <= 15.0:
                        angles.append(deg)
                    elif 75.0 <= abs(deg) <= 105.0:
                        deg_v = deg - 90.0 if deg > 0 else deg + 90.0
                        if -15.0 <= deg_v <= 15.0:
                            angles.append(deg_v)
            if angles:
                angle = float(np.median(angles))
            r_rad = float(np.deg2rad(angle))

    # 2. 在轉正的灰階圖中，精準分析白紙單據的左、右、上、下邊界
    #    只取中間 15%~72% 的高度來決定左右邊界（避開底部托盤邊框反光）
    deg_corr = float(np.rad2deg(r_rad))
    M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg_corr, 1.0)
    rot_gray = cv2.warpAffine(
        blur, M_rot, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )

    y_mid0, y_mid1 = int(h * 0.15), int(h * 0.72)
    col_means = np.mean(rot_gray[y_mid0:y_mid1, :], axis=0)
    white_cols = np.where(col_means > 165.0)[0]

    if len(white_cols) > w * 0.15:
        x0 = int(white_cols[0])
        x1 = int(white_cols[-1])
        pad_x = max(6, int((x1 - x0) * 0.02))
        x0 = max(0, x0 - pad_x)
        x1 = min(w, x1 + pad_x)

        row_means = np.mean(rot_gray[:, x0:x1], axis=1)
        white_rows = np.where(row_means > 155.0)[0]
        if len(white_rows) > h * 0.20:
            y0 = int(white_rows[0])
            # 尋找從頂部往下連續白紙的終點（遇到黑底托盤暗區 < 130 超過 12 px 即停止）
            y1 = int(white_rows[-1])
            dark_streak = 0
            for y_idx in range(max(y0 + int(h * 0.25), int(h * 0.50)), h):
                if row_means[y_idx] < 130.0:
                    dark_streak += 1
                    if dark_streak >= 12:
                        y1 = y_idx - dark_streak
                        break
                else:
                    dark_streak = 0

            pad_y = max(6, int((y1 - y0) * 0.015))
            y0 = max(0, y0 - pad_y)
            y1 = min(h, y1 + pad_y)

            rw = float(x1 - x0)
            rh = float(y1 - y0)
            rcx_rot = (x0 + x1) / 2.0
            rcy_rot = (y0 + y1) / 2.0
            M_inv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -deg_corr, 1.0)
            orig_pt = M_inv @ np.array([rcx_rot, rcy_rot, 1.0])
            return float(orig_pt[0]), float(orig_pt[1]), rw, rh, r_rad

    # 保底回傳全圖微縮
    return w / 2.0, h / 2.0, w * 0.90, h * 0.90, 0.0


def refine_stub_detection_roi(
    img: np.ndarray,
    det_box: Sequence[float],
    expand_ratio: float = 1.15,
) -> List[float]:
    """
    對 MMR 偵測到的 stub 框 [cx, cy, w, h, r, conf, cls_id] 進行白紙邊界緊貼修正。
    若裁切區內可找到合理的白紙邊界，回傳修正後的 [cx, cy, w, h, r, conf, cls_id]；
    否則回傳原框。
    """
    if img is None or img.size == 0 or len(det_box) < 7:
        return list(det_box)

    img_h, img_w = img.shape[:2]
    cx, cy, bw, bh, r, conf, cls_id = [float(v) for v in det_box[:7]]
    if bw < 20 or bh < 20:
        return list(det_box)

    half_w = (bw * expand_ratio) / 2.0
    half_h = (bh * expand_ratio) / 2.0
    x0 = max(0, int(round(cx - half_w)))
    y0 = max(0, int(round(cy - half_h)))
    x1 = min(img_w, int(round(cx + half_w)))
    y1 = min(img_h, int(round(cy + half_h)))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return list(det_box)

    patch = img[y0:y1, x0:x1]
    rcx, rcy, rw, rh, rr = estimate_document_roi(patch)
    # 若修正後的面積太小（低於原框 35%），視為無效，保留原框
    if rw * rh < (bw * bh * 0.35):
        return list(det_box)

    new_cx = float(x0) + rcx
    new_cy = float(y0) + rcy
    new_r = rr if abs(rr) > 1e-3 else r
    return [new_cx, new_cy, float(rw), float(rh), float(new_r), conf, float(int(cls_id))]


def generate_stub_candidates(
    img: np.ndarray,
    detections: list,
    ticket_cls_id: int = 4,
    stub_cls_id: int = 9,
) -> list:
    """
    當 MMR 只偵測到主傳票（order_ticket, cls 4）而漏檢客戶聯（small_order_ticket, cls 9）時，
    從主傳票框的頂部區域估算 small_order_ticket 候選框。
    """
    has_stub = any(int(d[6]) == stub_cls_id for d in detections if len(d) >= 7)
    if has_stub:
        return []

    tickets = [d for d in detections if len(d) >= 7 and int(d[6]) == ticket_cls_id]
    candidates = []
    for t in tickets:
        cx, cy, w, h, r, conf, _ = t[:7]
        long_side = max(w, h)
        short_side = min(w, h)
        if long_side < 60:
            continue
        stub_h = long_side * 0.42
        stub_w = short_side * 1.02
        offset = (long_side - stub_h) * 0.35
        new_cy = max(stub_h / 2.0, cy - offset)
        candidates.append([
            float(cx), float(new_cy), float(stub_w), float(stub_h),
            float(r), 0.55, float(stub_cls_id)
        ])
    return candidates


def enhance_stub_detections(
    img: Optional[np.ndarray],
    detections: list,
    stub_cls_id: int = 9,
    ticket_cls_id: int = 4,
    stub_cls_name: str = "small_order_ticket",
    ticket_cls_name: str = "order_ticket",
    enable_candidate_fallback: bool = False,
    enable_full_image_fallback: bool = False,
    enable_roi_refinement: bool = False,
) -> list:
    """
    MMR 階段的 stub CV 補強統一入口（同時支援 Detection 物件清單與 [cx, cy, w, h, r, conf, cls_id] 陣列）：
    1. 若開啟 enable_roi_refinement，對已偵測到的 small_order_ticket 框做白紙邊界緊貼修正。
    2. 若畫面沒有偵測到 small_order_ticket：
       - 若開啟 enable_candidate_fallback 且有主傳票，從主傳票頂部推算 stub 候選框。
       - 若開啟 enable_full_image_fallback 且完全無任何偵測框，對全圖用 estimate_document_roi 鎖定白紙單據。
    回傳更新後的 detections 清單。
    """
    if img is None or img.size == 0:
        return list(detections) if detections else []
    if not (enable_candidate_fallback or enable_full_image_fallback or enable_roi_refinement):
        return list(detections) if detections else []

    is_det_obj = bool(detections and isinstance(detections[0], Detection))
    if is_det_obj or not detections:
        out_objs: List[Detection] = list(detections or [])
        has_stub = any(d.cls_name == stub_cls_name or d.cls_id == stub_cls_id for d in out_objs)

        if has_stub and enable_roi_refinement:
            refined: List[Detection] = []
            for d in out_objs:
                if d.cls_name == stub_cls_name or d.cls_id == stub_cls_id:
                    cx, cy, w, h, r = d.xywhr
                    raw_box = [cx, cy, w, h, r, d.conf, d.cls_id]
                    ncx, ncy, nw, nh, nr, nconf, ncls = refine_stub_detection_roi(img, raw_box)
                    refined.append(
                        Detection(
                            xyxy=xywhr_to_xyxy8(ncx, ncy, nw, nh, nr),
                            xywhr=(ncx, ncy, nw, nh, nr),
                            cls_id=int(ncls),
                            cls_name=d.cls_name,
                            conf=float(nconf),
                            frame_id=d.frame_id,
                        )
                    )
                else:
                    refined.append(d)
            return refined

        if not has_stub and enable_candidate_fallback:
            raw_tickets = [
                [*d.xywhr, d.conf, ticket_cls_id]
                for d in out_objs
                if d.cls_name == ticket_cls_name or d.cls_id == ticket_cls_id
            ]
            cands = generate_stub_candidates(
                img, raw_tickets, ticket_cls_id=ticket_cls_id, stub_cls_id=stub_cls_id
            )
            for c in cands:
                ncx, ncy, nw, nh, nr, nconf, ncls = c
                out_objs.append(
                    Detection(
                        xyxy=xywhr_to_xyxy8(ncx, ncy, nw, nh, nr),
                        xywhr=(ncx, ncy, nw, nh, nr),
                        cls_id=int(ncls),
                        cls_name=stub_cls_name,
                        conf=float(nconf),
                    )
                )
            if cands:
                has_stub = True

        if not has_stub and not out_objs and enable_full_image_fallback:
            ncx, ncy, nw, nh, nr = estimate_document_roi(img)
            out_objs.append(
                Detection(
                    xyxy=xywhr_to_xyxy8(ncx, ncy, nw, nh, nr),
                    xywhr=(ncx, ncy, nw, nh, nr),
                    cls_id=stub_cls_id,
                    cls_name=stub_cls_name,
                    conf=0.95,
                )
            )
        return out_objs

    # 原始 [cx, cy, w, h, r, conf, cls_id] 格式
    out = [list(d) for d in (detections or [])]
    has_stub = any(len(d) >= 7 and int(d[6]) == stub_cls_id for d in out)

    if has_stub and enable_roi_refinement:
        for i, d in enumerate(out):
            if len(d) >= 7 and int(d[6]) == stub_cls_id:
                out[i] = refine_stub_detection_roi(img, d)
        return out

    if not has_stub and enable_candidate_fallback:
        cands = generate_stub_candidates(
            img, out, ticket_cls_id=ticket_cls_id, stub_cls_id=stub_cls_id
        )
        if cands:
            out.extend(cands)
            has_stub = True

    if not has_stub and not out and enable_full_image_fallback:
        cx, cy, rw, rh, rr = estimate_document_roi(img)
        out.append([cx, cy, rw, rh, rr, 0.95, float(stub_cls_id)])

    return out


# ==============================================================================
# 二、OCR 階段 CV 補強：水平分隔線、漏檢細長數字墨跡團塊、垂直對齊數量欄補框
# ==============================================================================
HLineSpan = Tuple[float, int, int]  # (y_center, x_start, x_end)


def find_horizontal_line_spans(
    img: Optional[np.ndarray],
    min_width_ratio: float = 0.38,
) -> List[HLineSpan]:
    """
    偵測客戶聯影像中的水平分隔黑線（如表頭與品項區之間的貫穿黑線），
    回傳 (y_center, x_start, x_end) 清單（依 y_center 由上而下排序），
    其中 x_end 即為該水平線的右邊界（尾端）。
    """
    if img is None or img.size == 0:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    if w < 40 or h < 40:
        return []

    inv = 255 - gray
    bw = cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10
    )
    klen = max(20, int(w * min_width_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    num, _, stats, centroids = cv2.connectedComponentsWithStats(horiz)
    spans: List[HLineSpan] = []
    for i in range(1, num):
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        if cw >= klen and ch <= max(8, int(h * 0.04)):
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            x1 = x0 + cw
            y_c = float(centroids[i][1])
            spans.append((y_c, x0, x1))
    spans.sort(key=lambda t: t[0])
    return spans


def find_horizontal_lines(
    img: Optional[np.ndarray],
    min_width_ratio: float = 0.38,
) -> List[float]:
    """
    偵測客戶聯影像中的水平分隔黑線，回傳 Y 座標清單（由上而下）。
    """
    return [y for y, _, _ in find_horizontal_line_spans(img, min_width_ratio=min_width_ratio)]


def find_rightmost_ink(
    crop: Optional[np.ndarray],
    min_x_ratio: float = 0.72,
    max_x: Optional[int] = None,
) -> Optional[Tuple[int, int, int, int]]:
    """
    當 OCR det 漏框某列的右側細長數字（如 '1'）時，在該列高度帶的右側（x >= W * min_x_ratio）
    利用 adaptiveThreshold + connectedComponentsWithStats 尋找最右側的墨跡團塊 (x0, y0, x1, y1)。
    若提供 max_x（例如第一條水平線的右邊界尾端），則搜尋範圍不超過 max_x，避免誤抓裁切邊緣黑影。
    """
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape[:2]
    x_off = int(w * min_x_ratio)
    x_lim = min(w, int(max_x)) if (max_x is not None and max_x > x_off + 8) else w
    if x_lim - x_off < 8 or h < 8:
        return None

    roi = gray[:, x_off:x_lim]
    roi_w = x_lim - x_off
    bw = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    # 垂直膨脹讓斷裂的點陣字或細線連成一個連通塊
    bw = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3)))

    num, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    cands = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        # 字高約佔列高 30%~95%，且不可寬扁（排除底線或雜訊）
        if ch < h * 0.30 or ch > h * 0.95:
            continue
        if cw > ch * 1.3 or area < 8:
            continue
        # 排除緊貼最右邊界的裁切黑邊
        if x + cw >= roi_w - 2 and cw <= 3:
            continue
        cands.append((x + cw, x, y, cw, ch))

    if not cands:
        return None
    _, x, y, cw, ch = max(cands, key=lambda t: t[0])
    return (x_off + x, y, x_off + x + cw, y + ch)


def compute_qty_column_x(
    item_rows: list,
    img_width: int,
    qty_min_x_ratio: float = 0.55,
    default_x_ratio: float = 0.87,
    separator_x_end: Optional[float] = None,
    ref_box_h: float = 24.0,
) -> float:
    """
    計算整張客戶聯的「數量欄垂直對齊基準線 X 座標」：
    1. 收集所有品項列中位於右側（cx >= img_width * qty_min_x_ratio）且寬度合理的文字框中心 X，
       若至少有一列具備右側數量框，取其中位數。
    2. 若無已知數量框，優先以「第一條水平線的右邊界（尾端 separator_x_end）」為右界基準，
       往內縮半個補救框寬（half_w），使數量補救框右緣對齊水平線尾端，不受圖片裁切寬度偏差影響。
    3. 若連水平線亦未檢出，才退回預設比例 img_width * default_x_ratio。
    """
    if img_width <= 0:
        return 0.0

    min_cx = img_width * qty_min_x_ratio
    max_box_w = img_width * 0.25
    detected_qty_xs: List[float] = []

    for row in item_rows:
        boxes = getattr(row, "boxes", None) or []
        if not boxes:
            continue
        rightmost = max(boxes, key=lambda b: b.cx)
        if rightmost.cx >= min_cx and rightmost.w <= max_box_w:
            detected_qty_xs.append(float(rightmost.cx))

    if detected_qty_xs:
        return float(np.median(detected_qty_xs))

    if separator_x_end is not None and separator_x_end > 0:
        eff_h = max(16.0, float(ref_box_h))
        half_w = float(max(10, int(eff_h * 0.65)))
        return float(max(half_w, min(float(img_width) - half_w, float(separator_x_end) - half_w)))

    return float(img_width * default_x_ratio)


def build_aligned_qty_rect(
    row_y0: float,
    row_y1: float,
    aligned_cx: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    利用品項列的上下邊界 (row_y0, row_y1) 與數量欄垂直對齊基準線 aligned_cx，
    推算該列數量欄的正方形補救框座標 (x0, y0, x1, y1)。
    """
    if img_w <= 0 or img_h <= 0 or aligned_cx <= 0:
        return None

    body_cy = (row_y0 + row_y1) / 2.0
    body_h = max(16.0, float(row_y1 - row_y0))
    half_h = max(10, int(body_h * 0.65))
    half_w = half_h  # 保持接近正方形，有利於單字元數字辨識

    rx0 = max(0, int(round(aligned_cx)) - half_w)
    rx1 = min(img_w, int(round(aligned_cx)) + half_w)
    ry0 = max(0, int(round(body_cy)) - half_h)
    ry1 = min(img_h, int(round(body_cy)) + half_h)

    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None
    return (rx0, ry0, rx1, ry1)
