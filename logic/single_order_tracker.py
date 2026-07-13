"""
single_order_tracker.py
=========================
SingleOrderTracker 是 TrayTracker 的簡化版本，用於「畫面裡預設只會
處理一筆訂單」的場景：不偵測、不追蹤 tray 盤本身的位置，也沒有
「候選 tray 累積到門檻才新建」這種跟容器外型有關的邏輯。

它在建立時就直接生出唯一一筆固定的 Tray（id 固定為 SINGLE_ORDER_ID，
沿用前端原本 `tray_id` 這個 key，只是 value 先寫死），之後每一幀只需要
追蹤畫面中 ticket / sticker 的穩定度即可 —— 邏輯上跟 TrayTracker 的
`update_ticket_and_stickers` 幾乎一致，差別只在於不用 `tray.rect` 去
篩選「這個貼紙是不是屬於這個 tray」（反正整個畫面只有一筆訂單，
偵測到的全部都算它的）。

對外提供跟 TrayTracker 相同形狀的介面：
    - self.trays: Dict[str, Tray]
    - reset(tray_id) / reset_all() / reset_tray_states(tray_id)
    - remove_tray(tray_id, ts_utc)
    - update_tray_positions(tray_dets, ts_utc)   # no-op
    - update_ticket_and_stickers(ticket_dets, sticker_dets)

這樣 TrayStateMachine / LogicEngine 完全不需要知道底下到底是
TrayTracker 還是 SingleOrderTracker，就能正常運作。
"""

from __future__ import annotations

import time
from typing import Dict, List

from loguru import logger

from logic.geometry import iou_poly_poly
from logic.models import Tray, TrackedItem, TrayState

# 固定的訂單 key，沿用前端原本傳遞的 `tray_id` 欄位名稱，
# 只是這個 mode 下 value 永遠是這個寫死的字串。
# 如果之後跟前端討論出更適合的名稱，改這裡就好，其他地方不用動。
SINGLE_ORDER_ID = "single_order"


class SingleOrderTracker:
    def __init__(
        self,
        bus,
        sticker_missing_frame: int,
        tray_missing_frame: int,
        frame_width: int,
        frame_height: int,
        tray_id: str = SINGLE_ORDER_ID,
    ):
        self.bus = bus
        self.sticker_missing_frame = sticker_missing_frame
        # 沿用 tray 模式「多久沒偵測到算真的離開」的門檻，套用在 ticket 上，
        # 當作「這筆訂單結束、可以收尾」的其中一個判斷依據（見下方
        # update_ticket_and_stickers 尾端的收尾觸發邏輯）。
        self.tray_missing_frame = tray_missing_frame
        # single 模式沒有 tray 盤可以框，NEW_TRAY_DETECTED 的 rect 直接
        # 給整個畫面尺寸（見 _publish_new_order_detected）。
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tray_id = tray_id

        self.trays: Dict[str, Tray] = {}
        self._create_tray()

    # ------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------
    def _create_tray(self):
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        # single 模式沒有 tray 盤的旋轉框可以用，這裡直接給「整個畫面」
        # 的 xywhr（角度 0）當替代值，讓 LogicEngine._finalize_missing_trays
        # 裡用 crop_by_angle 擷取最後畫面的邏輯可以正常運作 —— 擷取到的
        # 就是訂單完成/失敗當下的整個畫面，而不是 None 導致整段被跳過。
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
    # 生命週期操作：對應 TrayTracker 的 reset 系列，介面相容
    # 單一訂單模式下沒有「這個 tray 消失了 / 換了個新的」的概念，
    # 所以 tray_id 參數只是為了介面相容才保留，內容一律忽略，
    # 一律針對固定的 self.tray_id 操作。
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        """單一訂單模式沒有『刪除某個 tray』，等同整個硬重置"""
        self.reset_all()

    def reset_all(self):
        self.trays.clear()
        self._create_tray()
        logger.info("[RESET-all][single_order] 已重置訂單狀態")
        self._publish_reset_status()

    def reset_tray_states(self, tray_id: str):
        """軟重置：回到 WAITING_TICKET，準備接下一筆訂單"""
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
        logger.info("[RESET-B][single_order] 已重置訂單狀態")
        self._publish_reset_status()

    def _publish_reset_status(self):
        """
        原本 TrayTracker 的 reset 系列本身也不 publish 任何訊息——tray 模式
        下前端「感覺得到」重置，是因為實體 tray 盤還在鏡頭前，之後會被
        update_tray_positions 重新累積候選、門檻到了再發一次
        NEW_TRAY_DETECTED。single 模式沒有這條「重新被偵測到」的路，
        所以這裡要主動補一個通知，讓前端知道系統已經清空、正在重新
        等待下一張 ticket，不會卡在原本畫面上不動。
        """
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
        """
        tray 模式下，前端是靠 NEW_TRAY_DETECTED（tray_id + rect）這個
        訊號來開始追蹤/顯示一個新 tray；single 模式因為沒有 tray 盤
        偵測，這個起點訊號整個不見了，前端會沒有觸發點可以接後續動作。

        這裡在「第一次鎖定這筆訂單的 ticket」時，補發跟 tray 模式相同
        格式的 NEW_TRAY_DETECTED：tray_id 固定、rect 直接給整個畫面
        尺寸（因為 single 模式下沒有 tray 盤可以框，這個 mode 的
        「容器」概念上就是整個鏡頭畫面）。
        """
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
        """
        單一訂單模式不會真的把 tray 從字典移除（否則下一幀就沒地方
        累積新訂單了），這裡改成軟重置，讓介面對 app.py / LogicEngine
        仍然相容。
        """
        self.reset_tray_states(self.tray_id)
        self.bus.publish_system({
            "ts": ts_utc,
            "type": "TRAY_REMOVED",
            "msg": {"tray_id": self.tray_id}
        })
        return True

    # ------------------------------------------------------------
    # 每幀更新
    # ------------------------------------------------------------
    def update_tray_positions(self, tray_dets: list, ts_utc: str) -> None:
        """單一訂單模式不追蹤 tray 盤位置，這裡刻意什麼都不做"""
        pass

    def update_ticket_and_stickers(self, ticket_dets: list, sticker_dets: list) -> None:
        """
        跟 TrayTracker 同名方法邏輯一致，只是不用 tray.rect 篩選
        —— 畫面上偵測到的 ticket / sticker 全部視為同一筆訂單的。
        """
        if self.tray_id not in self.trays:
            self._create_tray()
        tray = self.trays[self.tray_id]

        # --- ticket ---
        # 假設同時只會有一張 ticket；若偵測到多張（誤判/雜訊），先取第一張。
        if ticket_dets:
            if len(ticket_dets) > 1:
                logger.warning(f"[single_order] 同時偵測到 {len(ticket_dets)} 張 ticket，僅採用第一張")

            t_rect = ticket_dets[0].xyxy
            t_xywhr = ticket_dets[0].xywhr
            if tray.ticket is None:
                tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
                # 這筆訂單第一次被鎖定 ticket -> 通知前端可以開始後續動作
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

        # --- stickers ---
        matched_sticker_indices = set()
        newly_created_indices = set()  # 本幀剛建立的 entry，不該被視為「消失」

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
                tray.stickers[best_idx].stable_frames += 1
                tray.stickers[best_idx].bbox = s_rect
                tray.stickers[best_idx].xywhr = s_xywhr
                matched_sticker_indices.add(best_idx)
            else:
                tray.stickers.append(TrackedItem(bbox=s_rect, xywhr=s_xywhr))
                newly_created_indices.add(len(tray.stickers) - 1)

        # 邏輯跟 TrayTracker 一致：沒被偵測到的貼紙才累加 missing_count，
        # 到門檻後標記 is_missing，交給 TrayStateMachine.handle_missing_items
        # 處理業務判斷（扣品項 / 發 MQTT），這裡不重複做那件事。
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

            if ts.missing_count > self.sticker_missing_frame:
                ts.is_missing = True

        tray.stickers = [
            ts for ts in tray.stickers
            if not (ts.is_missing and ts.has_notified_missing)
        ]

        # ------------------------------------------------------------
        # 收尾觸發：single 模式沒有「tray 盤物理消失」這件事可以拿來判斷
        # 要不要截圖+CSV+收尾，所以改用下面兩個時機頂替，滿足其一就把
        # tray.missing_count 直接設成超過門檻的值——這樣完全複用既有的
        # LogicEngine._finalize_missing_trays（screenshot + CSV log +
        # remove_tray）邏輯，不用在 LogicEngine 另外開一條新流程：
        #   1) 這筆訂單已經核對完成 (TrayState.COMPLETED)
        #   2) ticket 已經離開畫面超過門檻 (訂單被拿走/收走了)
        # 其餘情況維持 0，代表這筆訂單還在進行中，先不要收尾。
        # ------------------------------------------------------------
        ticket_left = tray.ticket is not None and tray.ticket.missing_count > self.tray_missing_frame
        if tray.state == TrayState.COMPLETED or ticket_left:
            tray.missing_count = self.tray_missing_frame + 1
        else:
            tray.missing_count = 0