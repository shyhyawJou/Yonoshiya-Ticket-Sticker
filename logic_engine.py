from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto
# from thefuzz import fuzz, process
from rapidfuzz import process, fuzz
import re
import time
import base64
import traceback
import cv2
import copy
import numpy as np
from collections import defaultdict, deque, Counter
from shapely.geometry import Polygon, Point
from loguru import logger

from schema import Detection
from config import Config
from mqtt_bus import MqttBus
from ocr_engine import StandaloneRecDLA
from mmr_engine import Rotated_RTMDET
from csv_writer import CsvWriter
import unicodedata

PolygonXYXY = Tuple[float, float, float, float, float, float, float, float]
ObbXYWHR = Tuple[float, float, float, float, float]

# Text normalization
def normalize_text(text: str, lower: bool = True, ignore_space: bool = True) -> str:
    """NFC-normalize unicode and strip leading/trailing whitespace."""
    # text = unicodedata.normalize("NFC", text).strip()
    text = unicodedata.normalize("NFKC", text).strip()
    if lower:
        text = text.lower()
    if ignore_space:
        text = text.replace(" ", "")
    return text

# ---------- geometry helpers ----------
def _polygon_from_tuple(p: PolygonXYXY) -> Polygon:
    """將 8點元組 (4頂點) 轉換為 Shapely Polygon 物件"""
    return Polygon([(p[i], p[i+1]) for i in range(0, 8, 2)])

def polygon_area(p: PolygonXYXY) -> float:
    """計算 4頂點多邊形 的面積"""
    if not p: return 0.0
    return _polygon_from_tuple(p).area

def inter_area_poly_poly(p1: PolygonXYXY, p2: PolygonXYXY) -> float:
    """計算兩個 4頂點多邊形 之間的交集面積"""
    if not p1 or not p2: return 0.0
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

# ------------------------------------------------------------
class TrayState(Enum):
    WAITING_TICKET = auto()      # 1. 等待訂單放入並穩定
    CHECKING_TICKET = auto()     # 2. 訂單已穩定，正在跑 OCR
    WAITING_STICKERS = auto()    # 3. 訂單已解析，等待餐點(貼紙)放入並穩定
    CHECKING_STICKERS = auto()   # 4. 餐點貼紙已穩定，正在跑 OCR
    COMPLETED = auto()           # 5. 所有品項核對完成

# ------------------------------------------------------------
@dataclass
class TrackedItem:
    bbox: PolygonXYXY
    xywhr: ObbXYWHR
    stable_frames: int = 0
    is_ocr_busy: bool = False # 標記這個物件是否正在跑 OCR
    is_checked: bool = False  # 標記這個貼紙是否已經核對成功，避免重複辨識

@dataclass
class Tray:
    id: str
    rect: PolygonXYXY
    xywhr: ObbXYWHR
    ticket_crop: np.ndarray
    state: TrayState = TrayState.WAITING_TICKET
    start_time_str: str = ''

    missing_count: int = 0
    drift_count: int = 0
    
    expected_items: List[str] = field(default_factory=list) 
    checked_items: List[str] = field(default_factory=list)  
    
    ticket: Optional[TrackedItem] = None
    stickers: List[TrackedItem] = field(default_factory=list)

# ------------------------------------------------------------
class LogicEngine:
    def __init__(self, cfg: Config, bus: MqttBus, mmr: Rotated_RTMDET, rec_path: str, dict_path: str):
        self.cfg = cfg
        self.bus = bus
        self.mmr = mmr
        # self.menus_ticket = cfg.menus_ticket
        # self.menus_sticker = cfg.menus_sticker
        # self.special_cases: Dict[str, List[str]] = {
        #     sc.name: sc.sub_items for sc in cfg.menus_mapping
        # }
        #### 2026/06/24 by Chris
        self.menus_ticket = [normalize_text(m, ignore_space=True) for m in cfg.menus_ticket]
        self.menus_sticker = [normalize_text(m, ignore_space=True) for m in cfg.menus_sticker]
        self.menus_with_digits = {
            m for m in self.menus_ticket
            if re.search(r'[1-9]', m)
        }
        self.special_cases: Dict[str, List[str]] = {
            normalize_text(sc.name, ignore_space=True): [
                normalize_text(sub, ignore_space=True) for sub in sc.sub_items
            ]
            for sc in cfg.menus_mapping
        }
        self.rec_pred = StandaloneRecDLA(
            rec_path=rec_path,
            dict_path=dict_path
        )


        # === 參數讀取設定 ===
        self.N = int(cfg.stability.n_settle_frame)
        self.K = int(cfg.stability.k_container_fail)
        self.K_new = int(cfg.stability.k_container_new)
        self.tray_missing_frame = int(cfg.stability.tray_missing_frame)
        self.roi_strink = cfg.placement.roi_strink
        self.iou_assign = cfg.placement.iou_assign
        self.iou_candidate = cfg.placement.iou_candidate
        self.drift_iou_thresh = cfg.placement.drift_iou_thresh

        self.trays: Dict[str, dict] = {}
        self.tray_candidates: List[dict] = []
        self.tray_id_counter: int = 0

        self.csv = CsvWriter(log_dir="/mnt/reserved/csv_uploaded")


    def reset(self, tray_id: str):
        """硬重置：清除所有餐盤，從頭開始偵測"""
        if tray_id not in self.trays:
            logger.warning(f"[RESET-A] 找不到餐盤 ID: {tray_id}，略過")
            return
        del self.trays[tray_id]
        logger.info(f"[RESET-A] 已刪除餐盤 {tray_id}")
    
    def reset_all(self):
        """硬重置：清除所有餐盤，從頭開始偵測"""
        self.trays.clear()
        logger.info(f"[RESET-all] 已刪除 all 餐盤")

    def reset_tray_states(self, tray_id: str):
        """軟重置：保留餐盤位置，每個餐盤回到 WAITING_TICKET"""
        if tray_id not in self.trays:
            logger.warning(f"[RESET-B] 找不到餐盤 ID: {tray_id}，略過")
            return

        tray = self.trays[tray_id]
        tray.state = TrayState.WAITING_TICKET  
        tray.missing_count = 0
        tray.drift_count = 0
        tray.expected_items = []
        tray.checked_items = []
        tray.ticket = None
        tray.stickers = []
        logger.info(f"[RESET-B] 已重置餐盤 {tray_id} 狀態")


    def remove_tray(self, tray_id: str, ts_utc: str) -> bool:
        if tray_id in self.trays:
            del self.trays[tray_id]
            logger.info(f"偵測餐盤移除，ID: {tray_id}")
            self.bus.publish_system({
                "ts": ts_utc,
                "type": "TRAY_REMOVED",
                "msg": {"tray_id": tray_id}
            })
            return True
        else:
            logger.warning(f"收到完成指令，但找不到盤 ID: {tray_id}")
            return False


    def _shrink_rect(self, r: PolygonXYXY, factor: float) -> PolygonXYXY:

        pts = np.array(r).reshape(4, 2)
        centroid = np.mean(pts, axis=0)
        scale = 1.0 - factor
        shrunk_pts = centroid + (pts - centroid) * scale

        return tuple(shrunk_pts.flatten())
    def _fuzzy_match_with_hints(
        self,
        text: str,
        candidates: List[str],
        special_prefixes: List[str] = ["ねぎ"],  # 需要特殊處理的前綴清單
        prefix_len: int = 2,
        score_threshold: int = 35
    ) -> Optional[Tuple[str, int]]:

        text = normalize_text(text, ignore_space=True)
        text_len = len(text)
        prefix = text[:prefix_len]

        # 只有在前綴命中 special_prefixes 時才走特殊邏輯
        if any(text.startswith(p) for p in special_prefixes):
            p1 = [c for c in candidates if len(c) == text_len and c.startswith(prefix)]
            if len(p1) == 1:
                logger.info(f"[前綴唯一] '{text}' → '{p1[0]}'")
                return p1[0], 100
            if len(p1) > 1:
                result = process.extractOne(text, p1, scorer=fuzz.ratio)
                if result and result[1] > score_threshold:
                    logger.info(f"[前綴+字數] '{text}' → '{result[0]}' score={result[1]}")
                    return result[0], result[1]
            # 特殊前綴但比不到，也 fallback 下去

        # 原本邏輯，其他品項完全不受影響
        result = process.extractOne(text, candidates, scorer=fuzz.token_set_ratio)
        if result and result[1] > 40:
            return result[0], result[1]

        return None

    def _parse_order_text(self, frame_crop: np.ndarray, dt_boxes: list, rec_res: list, is_flip: bool) -> List:
        parsed_items = []
        order_number = None
        nn_x = None

        flip_res = list(reversed(rec_res)) if is_flip else rec_res
        flip_dt = list(reversed(dt_boxes)) if is_flip else dt_boxes

        # 1. 找出 takeout_y / takeout_x / nn_x，同時收集純數字框
        takeout_y = None
        takeout_x = None
        digit_boxes = []  # (cy, digit_value) 純數字框，用於 menus_with_digits 的 fallback

        for dt_box, (text, score) in zip(flip_dt, flip_res):
            if score < 0.5:
                continue

            cy = np.mean([pt[1] for pt in dt_box])

            # 收集純數字框
            if re.fullmatch(r'[1-9]', text.strip()):
                digit_boxes.append((cy, int(text.strip())))

            # 判斷訂單編號
            if order_number is None:
                match_order_number = re.search(r'\d{3}$', text.strip())
                if match_order_number:
                    order_number = match_order_number.group()
                    x_coords = [pt[0] for pt in dt_box]
                    if not is_flip:
                        takeout_x = (80, max(x_coords) - 60)
                        takeout_y = np.mean([pt[1] for pt in dt_box]) + 100
                    else:
                        takeout_x = (min(x_coords) + 60, frame_crop.shape[1] - 80)
                        takeout_y = np.mean([pt[1] for pt in dt_box]) - 100

            if text == "枚":
                text = "数"
            nn_match = process.extractOne(text, ["数"], scorer=fuzz.token_set_ratio)
            if nn_match and nn_match[1] > 60:
                nn_x = np.mean([pt[0] for pt in dt_box])
                x_coords = [pt[0] for pt in dt_box]
                nn_w = max(x_coords) - min(x_coords)  # 數/枚 框的寬度

            take_out_match = process.extractOne(text, ["ティクアウト"], scorer=fuzz.token_set_ratio)
            if take_out_match and take_out_match[1] > 60:
                takeout_y = np.mean([pt[1] for pt in dt_box])
                x_coords = [pt[0] for pt in dt_box]
                takeout_x = (min(x_coords), max(x_coords))
                break

        # 2. 過濾有效文字區塊
        valid_items = []
        for dt_box, (text, score) in zip(flip_dt, flip_res):
            min_score = 0.3 if text.startswith("ねぎ") else 0.5
            if score < min_score:
                continue

            cy = np.mean([pt[1] for pt in dt_box])
            cx = np.mean([pt[0] for pt in dt_box])
            h = abs(min(dt_box[3][1], dt_box[2][1]) - max(dt_box[0][1], dt_box[1][1]))

            if takeout_y is not None and takeout_x is not None:
                if not is_flip:
                    if cy < takeout_y:
                        continue
                    if cx < (takeout_x[0] - 30) or cx > (takeout_x[1] + 40):
                        continue
                else:
                    if cy > takeout_y:
                        continue
                    if cx < (takeout_x[0] - 40) or cx > (takeout_x[1] + 30):
                        continue

            valid_items.append({
                'text': text,
                'score': score,
                'dt_box': dt_box,
                'cx': cx,
                'cy': cy,
                'h': h
            })

        # 3. 將同個高度的文字合併成同一列
        rows = []
        for item in valid_items:
            placed = False
            for row in rows:
                avg_cy = sum(r['cy'] for r in row) / len(row)
                avg_h = sum(r['h'] for r in row) / len(row)
                if abs(item['cy'] - avg_cy) < avg_h * 0.5:
                    row.append(item)
                    placed = True
                    break
            if not placed:
                rows.append([item])

        # 4. 合併內容並與 menus_ticket 比對，判斷數量
        for row in rows:
            row.sort(key=lambda x: x['cx'], reverse=is_flip)
            merged_text = "".join([item['text'] for item in row])
            row_cy = np.mean([item['cy'] for item in row])
            avg_h = np.mean([item['h'] for item in row])

            # 嘗試拆出前綴數字
            prefix_m = re.match(r'^([1-9])(.+)', merged_text.strip())
            leading_digit = int(prefix_m.group(1)) if prefix_m else None

            # 永遠用完整字串做 fuzzy match
            match_result = self._fuzzy_match_with_hints(merged_text, self.menus_ticket)

            if match_result:
                matched_str, match_score = match_result
                if match_score > 40:
                    match_num = 1  # 預設

                    if leading_digit is not None and matched_str not in self.menus_with_digits:
                        # 菜名不含數字，前綴數字直接可信
                        match_num = leading_digit
                        logger.info(f"[數量判斷-方法一 前綴數字]: '{merged_text}' → 數量={match_num}")
                    else:
                        # 菜名含數字（或沒有前綴），用 digit_boxes 找同行最近的純數字框
                        if digit_boxes:
                            nearest_cy, nearest_val = min(digit_boxes, key=lambda x: abs(x[0] - row_cy))
                            if abs(nearest_cy - row_cy) < avg_h * 0.8:
                                match_num = nearest_val
                                logger.info(f"[數量判斷-方法二 digit_boxes]: '{merged_text}' → 最近純數字框 cy={nearest_cy:.1f} val={nearest_val} → 數量={match_num}")
                            
                        if match_num == 1:
                            menu_item = None
                            for item in row:
                                item_match = self._fuzzy_match_with_hints(item['text'], self.menus_ticket)
                                if item_match and item_match[0] == matched_str:
                                    menu_item = item
                                    break
                            if menu_item is None:
                                menu_item = max(row, key=lambda x: x['h'])
                                logger.info(f"[數量判斷-方法三 rec_pred]: '{merged_text}' → 找不到菜名框，使用最大框")
                            else:
                                logger.info(f"[數量判斷-方法三 rec_pred]: '{merged_text}' → 找到菜名框 cy={menu_item['cy']:.1f} h={menu_item['h']:.1f}")

                            if nn_x is not None:
                                alone_res, _ = self.rec_pred.predict_quantity(
                                    frame_crop,
                                    cx=nn_x,
                                    cy=menu_item['cy'],
                                    h=menu_item['h'],
                                    w=nn_w,
                                    is_flip=is_flip
                                )
                                m2 = re.search(r'[1-9]', alone_res)
                                if m2:
                                    match_num = int(m2.group())
                                    logger.info(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 數量={match_num}")
                                else:
                                    logger.warning(f"[數量判斷-方法三 rec_pred]: rec_pred 辨識結果='{alone_res}' → 無法取得數量，預設=1")
                            else:
                                logger.warning(f"[數量判斷-方法三 rec_pred]: nn_x 為 None，無法定位數量欄位，預設=1")

                    logger.info(f"[訂單餐點確認]: OCR合併結果 -> '{merged_text}' 對應 -> '{matched_str}' ... 數量: {match_num}")
                    parsed_items.append((matched_str, match_num))

        logger.info(f"[訂單解析結果]: {parsed_items}, 編號: {order_number}")
        return parsed_items, order_number

    ##### 2026/06/24 by Chris
    def _match_sticker_with_ticket(self, rec_res: list, dt_boxes: list, is_flip: bool, expected_items: list, checked_items: list) -> str:

        if not expected_items:
            return None, "UNRECOGNIZED"
            
        remaining_items = copy.deepcopy(expected_items) if expected_items else []
        for item in checked_items:
            if item in remaining_items:
                remaining_items.remove(item)         

        if not remaining_items:
            return None, "UNRECOGNIZED"

        # 收集所有 rec_res 中分數夠高的候選比對結果，最後選最佳
        best_matched_str = None
        best_match_score = 0
        best_match_status = "UNRECOGNIZED"
        last_text = ""

        if dt_boxes:
            # ====== [已註解] 原本挑選高度最高的邏輯 (適用於未轉正或依大小篩選) ======
            # def box_height(box):
            #     ys = [pt[1] for pt in box]
            #     return max(ys) - min(ys)
            # top_idx = max(range(len(dt_boxes)), key=lambda i: box_height(dt_boxes[i]))
            # ====================================================================

            # ====== [新邏輯] 貼紙轉正後，挑選 y 軸最小的（最靠上方的文字） ======
            def box_top_y(box):
                # 取得該 box 四個點中最小的 y 值（即該文字方塊的頂部邊界）
                return min([pt[1] for pt in box])
            
            # 尋找頂部 y 軸最小（最靠近圖片上方）的 box 索引
            top_idx = min(range(len(dt_boxes)), key=lambda i: box_top_y(dt_boxes[i]))
            rec_res = [rec_res[top_idx]] if top_idx < len(rec_res) else rec_res

        for text, score in rec_res:
            if score < 0.5:
                continue
            last_text = normalize_text(text)
            
            match_result = self._fuzzy_match_with_hints(text, self.menus_sticker)
            if match_result:
                matched_str, match_score = match_result
                if match_score > 60 and match_score > best_match_score:
                    best_match_score = match_score
                    best_matched_str = matched_str

        if best_matched_str is not None:
            if best_matched_str in remaining_items:
                logger.info(f"[餐點確認成功]: 最佳OCR {last_text} 對應 -> '{best_matched_str}' (score: {best_match_score})")
                return best_matched_str, "MATCHED"
            else:
                logger.warning(f"[錯誤餐點]: 最佳OCR {last_text}  對應 -> '{best_matched_str}'，不在該餐盤訂單中")
                return best_matched_str, "WRONG_ITEM"
        
        logger.warning(f"[餐點確認失敗]: OCR -> '{last_text}'")
        return None, "UNRECOGNIZED"


    def set_ocr_busy(self, tray_id: str, item_type: str, bbox: PolygonXYXY):
        """
        由 app.py 確認已將任務送給 OCR 後呼叫，正式鎖定狀態
        """
        if tray_id not in self.trays:
            return
            
        tray = self.trays[tray_id]
        
        if item_type == "ticket" and tray.state == TrayState.WAITING_TICKET:
            if tray.ticket:
                tray.ticket.is_ocr_busy = True
                tray.state = TrayState.CHECKING_TICKET
                
        elif item_type == "sticker" and tray.state == TrayState.WAITING_STICKERS:
            # 找出是哪一張貼紙被送出去了 (利用 bbox 的 IoU 來確認)
            for ts in tray.stickers:
                if iou_poly_poly(ts.bbox, bbox) > 0.1:
                    ts.is_ocr_busy = True
                    tray.state = TrayState.CHECKING_STICKERS
                    break
 

    def _encode_image(self, img: Optional[np.ndarray]) -> str:
        if img is None:
            return ""
        try:
            success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if success:
                return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            logger.error(f"Image encode error: {e}")
        return ""


    def update(self, frame: np.ndarray, detections: List[Detection]) -> List[dict]:
        """
        1. 追蹤 tray盤 (餐盤)
        2. 分配並追蹤盤內的 ticket (訂單) 與 sticker (貼紙)，計算穩定度
        3. 依據狀態機生成 OCR Task
        """
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        tasks: List[dict] = []
        tray_movements: dict = {}
        
        # --- 分類物件偵測結果 ---
        tray_dets: List[Detection] = []
        ticket_dets: List[Detection] = []
        sticker_dets: List[Detection] = []

        for d in detections:
            if d.cls_name == "tray":
                tray_dets.append(d)
            elif d.cls_name == "order_ticket":
                ticket_dets.append(d)
            elif d.cls_name == "order_sticker":
                sticker_dets.append(d)

        # ==========================================
        # 更新餐盤位置 (Tray Tracking)
        # ==========================================
    
        # 1-1. 追蹤既有餐盤
        matched_tray_indices = set()   
        for _, tray in self.trays.items():
            best_iou = 0.0
            best_idx = -1
            
            for idx, d in enumerate(tray_dets):
                if idx in matched_tray_indices:
                    continue
                val = iou_poly_poly(d.xyxy, tray.rect)
                if val > best_iou:
                    best_iou = val
                    best_idx = idx
            
            if best_idx != -1 and best_iou >= self.iou_assign:
                matched_tray_indices.add(best_idx)
                current_rect = tray_dets[best_idx].xyxy
                current_xywhr = tray_dets[best_idx].xywhr

                tray.missing_count = 0
                if best_iou < self.drift_iou_thresh:
                    tray.drift_count += 1
                else:
                    tray.drift_count = 0
                    tray.xywhr = current_xywhr
                
                if tray.drift_count >= self.K:

                    # 計算 tray 盤移動向量
                    # old_cx, old_cy = get_polygon_centroid(tray.rect)
                    # new_cx, new_cy = get_polygon_centroid(current_rect)
                    # dx = new_cx - old_cx
                    # dy = new_cy - old_cy

                    # 更新 tray 盤座標
                    tray.rect = current_rect           
                    tray.drift_count = 0      

                    # if dx != 0 or dy != 0:
                    #     tray_movements[tray.id] = (dx, dy)

                    # # tray 盤中的物品一起移動
                    # if tray.ticket:
                    #     x1, y1, x2, y2 = tray.ticket.bbox
                    #     tray.ticket.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                        
                    # for ts in tray.stickers:
                    #     x1, y1, x2, y2 = ts.bbox
                    #     ts.bbox = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            else:
                tray.missing_count += 1
                tray.drift_count = 0 
                      
        # 1-2. 處理未匹配的偵測 (可能是新盤)
        unmatched_tray_dets = [d for i, d in enumerate(tray_dets) if i not in matched_tray_indices]        
        next_candidates = []     
        for d in unmatched_tray_dets:
            is_overlap_existing = False
            new_shrunk_rect = self._shrink_rect(d.xyxy, self.roi_strink)
            for tray in self.trays.values():
                existing_shrunk_rect = self._shrink_rect(tray.rect, self.roi_strink)
                if iou_poly_poly(new_shrunk_rect, existing_shrunk_rect) > 0.1: 
                    is_overlap_existing = True
                    break
            
            if is_overlap_existing:
                continue

            matched_candidate = None
            for cand in self.tray_candidates:
                if iou_poly_poly(d.xyxy, cand['rect']) >= self.iou_candidate:
                    matched_candidate = cand
                    break 
            
            if matched_candidate:
                matched_candidate['count'] += 1
                matched_candidate['rect'] = d.xyxy 
                matched_candidate['xywhr'] = d.xywhr

                if matched_candidate['count'] >= self.K_new:
                    new_id = f"{self.tray_id_counter:03d}"
                    self.tray_id_counter += 1                  
                    self.trays[new_id] = Tray(
                        id=new_id,
                        rect=matched_candidate['rect'],
                        xywhr=matched_candidate['xywhr'],
                        start_time_str=ts_utc,
                        ticket_crop=None
                    )
                    logger.info(f"偵測到新餐盤，分配 ID: {new_id}")
                    rrect = matched_candidate['rect']
                    self.bus.publish_system({
                        "ts": ts_utc,
                        "type": "NEW_TRAY_DETECTED",
                        "msg": {"tray_id": new_id, "rect": [rrect[0].item(), rrect[1].item(), rrect[4].item(), rrect[5].item()]}
                    })
                else:
                    next_candidates.append(matched_candidate)
            else:
                next_candidates.append({'count': 1, 'rect': d.xyxy, 'xywhr': d.xywhr})
        
        self.tray_candidates = next_candidates

        # ==========================================
        # 處理訂單 (ticket) 和貼紙 (sticker)
        # ==========================================

        for tray_id, tray in self.trays.items():

            if tray.missing_count > 0:
                continue
  
            # --- 2-1. 處理訂單 (Ticket) ---
            tray_tickets = [d for d in ticket_dets if is_center_in_polygon(get_polygon_centroid(d.xyxy), tray.rect)]
            if tray_tickets:
                t_rect = tray_tickets[0].xyxy 
                t_xywhr = tray_tickets[0].xywhr
                if tray.ticket is None:
                    tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
                else:
                    if iou_poly_poly(t_rect, tray.ticket.bbox) > 0.7:
                        tray.ticket.stable_frames += 1
                        tray.ticket.bbox = t_rect  
                        tray.ticket.xywhr = t_xywhr  
                    else:
                        tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
            else:
                if tray.ticket and not tray.ticket.is_ocr_busy:
                    tray.ticket.stable_frames = 0
            
            # --- 2-2. 處理貼紙 (Stickers) ---
            tray_stickers = [d for d in sticker_dets if is_center_in_polygon(get_polygon_centroid(d.xyxy), tray.rect)]
            matched_sticker_indices = set()
            
            for d in tray_stickers:
                s_rect = d.xyxy
                s_xywhr = d.xywhr
                best_iou, best_idx = 0.0, -1
  
                for idx, ts in enumerate(tray.stickers):
                    if idx in matched_sticker_indices: 
                        continue
                    val = iou_poly_poly(s_rect, ts.bbox)
                    if val > best_iou:
                        best_iou, best_idx = val, idx

                if best_idx != -1 and best_iou > 0.1:
                    tray.stickers[best_idx].stable_frames += 1
                    tray.stickers[best_idx].bbox = s_rect
                    tray.stickers[best_idx].xywhr = s_xywhr
                    matched_sticker_indices.add(best_idx)                    
                else:
                    tray.stickers.append(TrackedItem(bbox=s_rect, xywhr=s_xywhr))
            
            # 對於沒偵測到的歷史貼紙，只要還沒過關且沒在跑 OCR，就讓穩定度歸零
            for idx, ts in enumerate(tray.stickers):
                if idx not in matched_sticker_indices:
                    if not ts.is_checked and not ts.is_ocr_busy:
                        ts.stable_frames = 0

        # ==========================================
        # 階段三：依據狀態機生成 OCR 任務
        # ==========================================
        
        trays_to_remove = []
        for tray_id, tray in self.trays.items():
 
            if tray.state == TrayState.WAITING_TICKET:
                if tray.ticket and tray.ticket.stable_frames >= self.N and not tray.ticket.is_ocr_busy:
                    tasks.append({
                        "tray_id": tray.id,
                        "type": "ticket",
                        "bbox": tray.ticket.bbox,
                        "xywhr": tray.ticket.xywhr
                    })

            elif tray.state == TrayState.WAITING_STICKERS:
                for ts in tray.stickers:
                    if ts.stable_frames >= self.N and not ts.is_checked and not ts.is_ocr_busy:
                        tasks.append({
                            "tray_id": tray.id,
                            "type": "sticker",
                            "bbox": ts.bbox,
                            "xywhr": ts.xywhr
                        })
                        break

            if tray.missing_count > self.tray_missing_frame:
                trays_to_remove.append(tray_id)

                # ======== 影像 ========
                final_tray_capture_b64 = ""
                ticket_capture_b64 = ""

                last_xywhr = tray.xywhr              
                if last_xywhr is not None:
                    cx, cy, w, h, r = last_xywhr
                    warped_img, _ = self.mmr.crop_by_angle(frame, cx, cy, w, h, r)
                    final_tray_capture_b64 = self._encode_image(warped_img)

                ticket_capture = tray.ticket_crop
                if ticket_capture is not None:
                    ticket_capture_b64 = self._encode_image(ticket_capture)

                expected_counts = Counter(tray.expected_items)
                expected_list = [{item: count} for item, count in expected_counts.items()]

                check_counts = Counter(tray.checked_items)
                check_list = [{item: count} for item, count in check_counts.items()]

                # 寫入 Log
                log_payload = {
                    "tray_id": tray_id,
                    "start_time": tray.start_time_str,
                    "end_time": ts_utc,
                    "expected_item": expected_list,
                    "expected_item_count": len(expected_list),
                    "checked_item": check_list,
                    "ticket_capture": ticket_capture_b64,
                    "final_tray_capture": final_tray_capture_b64
                }
                self.csv.log(log_payload)

        for tid in trays_to_remove:
            self.remove_tray(tid, ts_utc)
            
        return tasks, tray_movements


    def apply_ocr_result(self, tray_id: str, item_type: str, frame_crop: np.ndarray, dt_boxes: list, rec_res: list, is_flip: bool, task_bbox: PolygonXYXY):
        """
        1. 處理 OCR 結果
        2. 檢查 tray盤狀態
        3. 發布 mqtt 訊息
        """
        if tray_id not in self.trays:
            return
            
        tray = self.trays[tray_id]
        print(f"OCR res : {rec_res}")
        # --------------------------------------------------
        # 「訂單」的 OCR 結果
        # --------------------------------------------------
        if item_type == "ticket" and tray.state == TrayState.CHECKING_TICKET:

            if tray.ticket:
                tray.ticket.is_ocr_busy = False

            parsed_items, oreder_number,  = self._parse_order_text(frame_crop, dt_boxes, rec_res, is_flip) 
            
            if parsed_items:
                expanded_expected_items = []
                expected_items_list = []

                for item_name, item_num in parsed_items:
                    if item_name in self.special_cases:
                        sub_items = self.special_cases[item_name]
                        for sub_item in sub_items:
                            expanded_expected_items.extend([sub_item] * item_num)
                            expected_items_list.append({sub_item: item_num})
                    else:
                        expanded_expected_items.extend([item_name] * item_num)
                        expected_items_list.append({item_name: item_num})

                tray.expected_items = expanded_expected_items
                tray.state = TrayState.WAITING_STICKERS
                tray.ticket_crop = frame_crop
                oreder_number = "000" if oreder_number is None else oreder_number

                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TICKET_READY",
                    "expected_items": expected_items_list,
                    "order_number": oreder_number
                })
            else:
                if tray.ticket:
                    tray.ticket.stable_frames = 0
                tray.state = TrayState.WAITING_TICKET

            # logger.info(f"tray id: {tray_id}, tray state: {tray.state}")

        # --------------------------------------------------
        # 「貼紙」的 OCR 結果
        # --------------------------------------------------
        elif item_type == "sticker" and tray.state == TrayState.CHECKING_STICKERS:
            
            matched_item = None
            match_status = "UNRECOGNIZED"

            if len(rec_res) > 0:
                matched_item, match_status = self._match_sticker_with_ticket(rec_res, dt_boxes, is_flip, tray.expected_items, tray.checked_items)

            for ts in tray.stickers:
                if iou_poly_poly(ts.bbox, task_bbox) > 0.1: 
                    ts.is_ocr_busy = False
                    
                    if match_status == "MATCHED":
                        ts.is_checked = True # 標記核對成功，以後不會再送這張貼紙去 OCR
                    else:
                        ts.stable_frames = 0 # 核對失敗：穩定度歸零，準備重試
                    break

            if match_status == "MATCHED":
                tray.checked_items.append(matched_item)
                check_counts = Counter(tray.checked_items)
                items_list = [{item: count} for item, count in check_counts.items()]

                # 已確認正確品項
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "ITEM_CHECKED",
                    "items": items_list
                })
                
                if len(tray.checked_items) == len(tray.expected_items):
                    tray.state = TrayState.COMPLETED
                    # 完成所有品項
                    self.bus.publish_det_status({
                        "tray_id": tray_id,
                        "status": "TRAY_COMPLETED",
                        "items": items_list
                    })

            elif match_status == "WRONG_ITEM":
                # 錯誤品項
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "WRONG_ITEM_DETECTED",
                    "items": [{matched_item: 1}]
                })
      
            if tray.state != TrayState.COMPLETED:
                tray.state = TrayState.WAITING_STICKERS

            # logger.info(f"tray id: {tray_id}, tray state: {tray.state}")
