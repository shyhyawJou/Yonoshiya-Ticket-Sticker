"""
sticker_pending_buffer.py
=========================
「連續多幀確認」候選緩衝區。

用於 STABLE_CONFIRM_CLASSES 這幾類 sticker(目前是醬料包):
單幀偵測不直接建立追蹤物件,而是先進候選區,同一「品項」
(而非同一個原始 cls_name)連續匹配到 K 次以上才正式建立。

之所以按「品項」而不是「cls_name」分組,是因為同一包醬料
會因為 mmrotate 判斷正反面而在 front/back 兩個 cls_name 間
切換,如果按 cls_name 分組,一翻面就會被視為不同候選,導致
計數被打斷、永遠湊不齊門檻——這是先前造成畫面閃爍與遲遲無法
穩定建立的主因。

TrayTracker 與 SingleOrderTracker 都需要同一套累積規則,
故抽成獨立元件,避免兩邊各自維護一份幾乎相同、卻可能漂移
不同步的程式碼。

這裡刻意只處理「純追蹤層」的事(候選要不要升級成正式物件),
完全不知道 OCR、不知道訂單比對——跟 TrayTracker 本身的設計
原則一致。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from logic.geometry import PolygonXYXY, ObbXYWHR, iou_poly_poly
from logic.known_class_items import resolve_item_name


class StickerPendingBuffer:
    """
    封裝單一 tray 的 pending_stickers 累積/升級/淘汰邏輯。
    無狀態(不持有任何資料),每次呼叫時把 tray.pending_stickers
    當參數傳入傳出,維持跟 TrayTracker 現有風格一致(trays 資料
    本身仍歸呼叫端擁有)。
    """

    def __init__(self, k_new: int, iou_thresh: float = 0.1):
        self.k_new = k_new
        self.iou_thresh = iou_thresh

    def update(
        self,
        pending_list: List[dict],
        s_rect: PolygonXYXY,
        s_xywhr: ObbXYWHR,
        cls_name: str,
        matched_indices: Set[int],
    ) -> Optional[dict]:
        """
        嘗試將一個新偵測匹配進既有候選,或建立新候選。

        回傳:
            若這次匹配後候選數已達門檻,回傳應升級成正式 TrackedItem
            的 dict(bbox / xywhr / cls_name);否則回傳 None。
        """
        item_key = resolve_item_name(cls_name)
        best_idx, best_iou = -1, 0.0

        for idx, pending in enumerate(pending_list):
            if idx in matched_indices:
                continue
            if pending['item_key'] != item_key:
                continue
            val = iou_poly_poly(s_rect, pending['bbox'])
            if val > best_iou and val > self.iou_thresh:
                best_iou, best_idx = val, idx

        if best_idx != -1:
            pending = pending_list[best_idx]
            pending['bbox'] = s_rect
            pending['xywhr'] = s_xywhr
            pending['cls_name'] = cls_name  # 記錄最新一次看到的原始類別(front/back 皆可能)
            pending['count'] += 1
            matched_indices.add(best_idx)

            if pending['count'] >= self.k_new:
                return {'bbox': s_rect, 'xywhr': s_xywhr, 'cls_name': pending['cls_name']}
            return None

        pending_list.append({
            'bbox': s_rect,
            'xywhr': s_xywhr,
            'cls_name': cls_name,
            'item_key': item_key,
            'count': 1,
        })
        matched_indices.add(len(pending_list) - 1)
        return None

    def prune(self, pending_list: List[dict], matched_indices: Set[int]) -> List[dict]:
        """
        本幀收尾:
            1. 沒被匹配到的候選直接淘汰(中斷即重新計數,維持原行為)
            2. 已升級成正式物件的候選一併移除
        """
        survivors = [p for i, p in enumerate(pending_list) if i in matched_indices]
        return [p for p in survivors if p['count'] < self.k_new]