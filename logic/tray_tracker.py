"""
tray_tracker.py
================
TrayTracker 擁有並管理 `trays` 字典本身的生命週期：
    - 追蹤既有 tray 的位置（IoU 比對、drift 平滑）
    - 從未匹配的偵測結果中累積候選，門檻到了才建立新 tray
    - 追蹤 tray 內的 ticket / sticker 位置與穩定度 (stable_frames)
    - reset / remove 等硬重置操作

刻意設計成「只管資料本身該長什麼樣子，不管這些資料接下來要拿去做
什麼」：不知道 OCR、不知道狀態機、不知道 MQTT 訊息格式（除了 tray
建立/移除這兩個屬於「tray 生命週期」本身的事件）。

未來如果要換一套追蹤演算法（例如換偵測模型、換座標系統、甚至不再用
IoU 而改用其他跟蹤方式），理論上只需要替換這個檔案，
TrayStateMachine / OrderParser / StickerMatcher 完全不用動。
"""

from __future__ import annotations

from typing import Dict, List

from loguru import logger

from logic.geometry import get_polygon_centroid, is_center_in_polygon, iou_poly_poly, shrink_rect
from logic.models import Tray, TrackedItem


class TrayTracker:
    def __init__(
        self,
        bus,
        iou_assign: float,
        iou_candidate: float,
        drift_iou_thresh: float,
        k_container_fail: int,
        k_container_new: int,
        roi_shrink: float,
        sticker_missing_frame: int,
    ):
        self.bus = bus
        self.iou_assign = iou_assign
        self.iou_candidate = iou_candidate
        self.drift_iou_thresh = drift_iou_thresh
        self.K = k_container_fail
        self.K_new = k_container_new
        self.roi_shrink = roi_shrink
        self.sticker_missing_frame = sticker_missing_frame

        self.trays: Dict[str, Tray] = {}
        self.tray_candidates: List[dict] = []
        self.tray_id_counter: int = 0

    # ------------------------------------------------------------
    # 生命週期操作
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        """硬重置：清除指定餐盤"""
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

        # 延遲 import 避免與 models 之間不必要的耦合順序問題
        from logic.models import TrayState

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

    # ------------------------------------------------------------
    # 每幀更新
    # ------------------------------------------------------------
    def update_tray_positions(self, tray_dets: list, ts_utc: str) -> None:
        """
        1-1. 追蹤既有餐盤（IoU 比對 + drift 平滑）
        1-2. 處理未匹配的偵測結果（可能是新盤），累積候選並在門檻到達時建立新 tray
        """
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
                    tray.rect = current_rect
                    tray.drift_count = 0
            else:
                tray.missing_count += 1
                tray.drift_count = 0

        unmatched_tray_dets = [d for i, d in enumerate(tray_dets) if i not in matched_tray_indices]
        next_candidates = []
        for d in unmatched_tray_dets:
            is_overlap_existing = False
            new_shrunk_rect = shrink_rect(d.xyxy, self.roi_shrink)
            for tray in self.trays.values():
                existing_shrunk_rect = shrink_rect(tray.rect, self.roi_shrink)
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

    def update_ticket_and_stickers(self, ticket_dets: list, sticker_dets: list) -> None:
        """
        2-1. 更新每個 tray 的 ticket 追蹤與穩定度
        2-2. 更新每個 tray 的 sticker 追蹤與穩定度
        """
        for tray_id, tray in self.trays.items():

            if tray.missing_count > 0:
                continue

            # --- ticket ---
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

            # --- stickers ---
            tray_stickers = [d for d in sticker_dets if is_center_in_polygon(get_polygon_centroid(d.xyxy), tray.rect)]
            matched_sticker_indices = set()
            newly_created_indices = set()  # 本幀剛建立的 entry，不該被視為「消失」

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
                    newly_created_indices.add(len(tray.stickers) - 1)

            # 這一幀有被偵測到的貼紙（含既有匹配到的、含剛建立的）：消失計數歸零
            # 沒被偵測到的：
            #   - 還沒核對過、也沒在跑 OCR 的，穩定度先歸零（原本就有的行為）
            #   - 只要沒在跑 OCR（避免結果回來時對不到 bbox），消失計數 +1
            #     累積到門檻後，這筆追蹤紀錄會被視為「真的離開了」而移除，
            #     即使它是 is_checked=True 也一樣 —— 這是修正「貼紙拿走又放回
            #     被誤判成兩個餐點」的關鍵：已核對的貼紙不該永遠佔用一個
            #     幽靈 bbox，導致它真正離開又回來時，被系統當成第二個新物件。
            # --- 尋找這幀沒被匹配到的貼紙，累加 missing_count ---
            detected_indices = matched_sticker_indices | newly_created_indices
            for idx, ts in enumerate(tray.stickers):
                if idx in detected_indices:
                    ts.missing_count = 0
                    # 如果原本被判消失又出現了（例如短暫遮擋），可以救回來
                    ts.is_missing = False 
                    ts.has_notified_missing = False
                    continue

                if not ts.is_checked and not ts.is_ocr_busy:
                    ts.stable_frames = 0
                if not ts.is_ocr_busy:
                    ts.missing_count += 1
                
                # 門檻到了，標記為消失，交由 StateMachine 處理業務與發送 MQTT
                if ts.missing_count > self.sticker_missing_frame:
                    ts.is_missing = True

            # 【優化重點】：只移除「已經通知過前端消失」的鬼魂紀錄
            # 沒過期的、或是「剛過期但 StateMachine 還沒發通知」的都要保留
            tray.stickers = [
                ts for ts in tray.stickers 
                if not (ts.is_missing and ts.has_notified_missing)
            ]