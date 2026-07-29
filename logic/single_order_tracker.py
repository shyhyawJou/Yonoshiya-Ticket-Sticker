from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from loguru import logger

from logic.geometry import iou_poly_poly, is_geometrically_frozen
from logic.models import Tray, TrackedItem, TrayState
from logic.known_class_items import STABLE_CONFIRM_CLASSES
from logic.sticker_pending_buffer import StickerPendingBuffer

SINGLE_ORDER_ID = "single_order"


class SingleOrderTracker:
    def __init__(
        self,
        bus,
        sticker_missing_frame: int,
        tray_missing_frame: int,
        frame_width: int,
        frame_height: int,
        ticket_leave_frame: int,
        tray_id: str = SINGLE_ORDER_ID,
        k_sticker_new: int = 3,  # 需連續幾幀確認才建立 STABLE_CONFIRM_CLASSES 這幾類貼紙
    ):
        self.bus = bus
        self.sticker_missing_frame = sticker_missing_frame
        self.tray_missing_frame = tray_missing_frame
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tray_id = tray_id

        self.ticket_leave_frame = ticket_leave_frame

        self.trays: Dict[str, Tray] = {}
        self._create_tray()

        # 需要「連續多幀確認」才建立的 sticker 類別，其餘類別偵測一幀就建立
        self.K_sticker_new = k_sticker_new
        self.STABLE_CONFIRM_CLASSES = STABLE_CONFIRM_CLASSES
        self.pending_buffer = StickerPendingBuffer(k_new=self.K_sticker_new)
    # ------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------
    def _create_tray(self):
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        default_xywhr = (
            self.frame_width / 2.0,
            self.frame_height / 2.0,
            float(self.frame_width),
            float(self.frame_height),
            0.0,
        )
        self.trays[self.tray_id] = Tray(
            id=self.tray_id,
            rect=None,
            xywhr=default_xywhr,
            start_time_str=ts_utc,
            ticket_crop=None,
        )

    # ------------------------------------------------------------
    # 生命週期操作（reset 系列，跟原本一致，只加了 order_number 清空）
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        self.reset_all()

    def reset_all(self):
        self.trays.clear()
        self._create_tray()
        logger.info("[RESET-all][single_order] 已重置訂單狀態")
        self._publish_reset_status()

    def reset_tray_states(self, tray_id: str):
        if self.tray_id not in self.trays:
            self._create_tray()
            self._publish_reset_status()
            return

        tray = self.trays[self.tray_id]
        tray.state = TrayState.WAITING_TICKET
        tray.missing_count = 0
        tray.drift_count = 0
        tray.expected_items = []
        tray.checked_items = []
        tray.ticket = None
        tray.stickers = []
        tray.order_number = None  # 避免舊訂單編號殘留到下一筆
        logger.info("[RESET-B][single_order] 已重置訂單狀態")
        self._publish_reset_status()

    def _publish_reset_status(self):
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self.bus.publish_system({
            "ts": ts_utc,
            "type": "TRAY_RESET",
            "msg": {"tray_id": self.tray_id}
        })
        self.bus.publish_det_status({
            "tray_id": self.tray_id,
            "status": "WAITING_TICKET",
        })

    def _publish_new_order_detected(self):
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        self.bus.publish_system({
            "ts": ts_utc,
            "type": "NEW_TRAY_DETECTED",
            "msg": {
                "tray_id": self.tray_id,
                "rect": [0, 0, self.frame_width, self.frame_height],
            }
        })

    def remove_tray(self, tray_id: str, ts_utc: str) -> bool:
        self.reset_tray_states(self.tray_id)
        self.bus.publish_system({
            "ts": ts_utc,
            "type": "TRAY_REMOVED",
            "msg": {"tray_id": self.tray_id}
        })
        return True
    
    def _filter_ticket_dets(self, ticket_dets: list) -> list:
        """
        決定哪些 ticket 偵測要被納入這筆訂單的判斷。
        SingleOrderTracker 預設不做任何過濾(整個畫面只有一張訂單);
        PresetRoiTracker 會覆寫這個方法,只採用中心點落在 ticket_roi
        內的偵測,藉此重用本類別「單一訂單追蹤」的所有其餘邏輯
        (貼紙不受限制、ticket_leave_frame 判斷離開...等),不需要
        另外複製一份幾乎相同的程式碼。
        """
        return ticket_dets

    # ------------------------------------------------------------
    # 每幀更新
    # ------------------------------------------------------------
    def update_tray_positions(self, tray_dets: list, ts_utc: str) -> None:
        pass

    def update_ticket_and_stickers(self, ticket_dets: list, sticker_dets: list) -> None:
        if self.tray_id not in self.trays:
            self._create_tray()
        tray = self.trays[self.tray_id]
        ticket_dets = self._filter_ticket_dets(ticket_dets)
        # --- ticket ---
        if ticket_dets:
            if len(ticket_dets) > 1:
                logger.warning(f"[single_order] 同時偵測到 {len(ticket_dets)} 張 ticket，僅採用第一張")

            t_rect = ticket_dets[0].xyxy
            t_xywhr = ticket_dets[0].xywhr
            if tray.ticket is None:
                tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
                self._publish_new_order_detected()
            else:
                if iou_poly_poly(t_rect, tray.ticket.bbox) > 0.7:
                    tray.ticket.stable_frames += 1
                    tray.ticket.bbox = t_rect
                    tray.ticket.xywhr = t_xywhr
                else:
                    tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
            tray.ticket.missing_count = 0
        else:
            if tray.ticket and not tray.ticket.is_ocr_busy:
                tray.ticket.stable_frames = 0
                tray.ticket.missing_count += 1

        # --- stickers（邏輯不變，省略） ---
        matched_sticker_indices = set()
        newly_created_indices = set()
        matched_pending_indices = set()

        for d in sticker_dets:
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
                ts_obj = tray.stickers[best_idx]
                if is_geometrically_frozen(ts_obj.bbox, s_rect):
                    ts_obj.geo_stable_frames += 1
                else:
                    ts_obj.geo_stable_frames = 0
                ts_obj.stable_frames += 1
                ts_obj.bbox = s_rect
                ts_obj.xywhr = s_xywhr
                matched_sticker_indices.add(best_idx)
                continue

            if d.cls_name in self.STABLE_CONFIRM_CLASSES:
                upgrade = self.pending_buffer.update(
                    tray.pending_stickers, s_rect, s_xywhr, d.cls_name, matched_pending_indices
                )
                if upgrade is not None:
                    tray.stickers.append(TrackedItem(
                        bbox=upgrade['bbox'], xywhr=upgrade['xywhr'], cls_name=upgrade['cls_name']
                    ))
                    newly_created_indices.add(len(tray.stickers) - 1)
            else:
                tray.stickers.append(TrackedItem(bbox=s_rect, xywhr=s_xywhr, cls_name=d.cls_name))
                newly_created_indices.add(len(tray.stickers) - 1)

        # 本幀沒被匹配到的候選：直接淘汰（中斷即重新計數, 已升級為正式 sticker 的候選，需從 pending_stickers 移除
        tray.pending_stickers = self.pending_buffer.prune(tray.pending_stickers, matched_pending_indices)

        detected_indices = matched_sticker_indices | newly_created_indices
        for idx, ts in enumerate(tray.stickers):
            if idx in detected_indices:
                ts.missing_count = 0
                ts.is_missing = False
                ts.has_notified_missing = False
                continue

            if not ts.is_checked and not ts.is_ocr_busy:
                ts.stable_frames = 0
            if not ts.is_ocr_busy:
                ts.missing_count += 1
                ts.geo_stable_frames = 0

            if ts.missing_count > self.sticker_missing_frame:
                ts.is_missing = True

        tray.stickers = [
            ts for ts in tray.stickers
            if not (ts.is_missing and ts.has_notified_missing)
        ]

        # ------------------------------------------------------------
        # 收尾觸發：只看 ticket 是否「真的離開」，不管 tray.state。
        # 無論訂單有沒有核對完成，ticket 離開就收尾 —— 這是你要的行為。
        # ------------------------------------------------------------
        if tray.ticket is not None and tray.ticket.missing_count > 0:
            ticket_left = tray.ticket.missing_count > self.ticket_leave_frame
        else:
            ticket_left = False

        if ticket_left:
            tray.missing_count = self.tray_missing_frame + 1
            logger.warning(f'傳票消失超過 {self.ticket_leave_frame} 幀, 觸發結束 !')
        else:
            tray.missing_count = 0