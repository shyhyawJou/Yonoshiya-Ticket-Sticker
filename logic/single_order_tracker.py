from __future__ import annotations

import time
from typing import Dict, List

import numpy as np
from loguru import logger

from logic.geometry import iou_poly_poly, is_center_in_polygon, PolygonXYXY, get_polygon_centroid, center_shift_ratio
from logic.models import Tray, TrackedItem, TrayState
from logic.known_class_items import STABLE_CONFIRM_CLASSES
from logic.sticker_pending_buffer import StickerPendingBuffer

from utils.clip_video import get_now_iso_str

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
        ticket_exclude_roi_polygon: Optional[PolygonXYXY] = None,
        finalize_mode: str = "ticket_leave",   # "ticket_leave" | "ticket_mat"
        ticket_mat_confirm_frames: int = 10,
        still_shift_ratio: float = 0.0,   # >0 才啟用靜止判定，見 _advance_stable
    ):
        self.bus = bus
        self.still_shift_ratio = still_shift_ratio
        self.sticker_missing_frame = sticker_missing_frame
        self.tray_missing_frame = tray_missing_frame
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tray_id = tray_id

        self.ticket_leave_frame = ticket_leave_frame
        self.ticket_exclude_roi_polygon = ticket_exclude_roi_polygon

        # 超時收尾後的「等傳票離開」閘門（見 hold_tickets_until_clear）。
        # 連續消失幾幀才算離開，沿用 ticket_leave_frame 的定義。
        self._holding_tickets = False
        self._ticket_clear_count = 0

        self.trays: Dict[str, Tray] = {}
        self._create_tray()

        # 需要「連續多幀確認」才建立的 sticker 類別，其餘類別偵測一幀就建立
        self.K_sticker_new = k_sticker_new
        self.STABLE_CONFIRM_CLASSES = STABLE_CONFIRM_CLASSES
        self.pending_buffer = StickerPendingBuffer(k_new=self.K_sticker_new)

        # 新增 ticket mat 相關參數, 最後判定有兩種方法, 一個是 ticket (物件遮住會被誤判), 一個是 ticket mat (實驗中)
        self.ticket_mat_confirm_frames = ticket_mat_confirm_frames
        self.finalize_mode = finalize_mode

        finalize_checkers = {
            "ticket_leave": self._check_finalize_ticket_leave,
            "ticket_mat": self._check_finalize_ticket_mat,
        }
        if finalize_mode not in finalize_checkers:
            raise ValueError(f"unknown finalize_mode: {finalize_mode}")
        self._finalize_checker = finalize_checkers[finalize_mode]
    # ------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------
    def _create_tray(self):
        # ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        ts_local = get_now_iso_str(time.time(), utc=False)
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
            start_time_str=ts_local,
            ticket_crop=None,
        )

    # ------------------------------------------------------------
    # 生命週期操作（reset 系列）
    #
    # 重置一律「換一個全新的 Tray」，不逐欄清：一筆訂單的所有狀態都掛在
    # Tray 上，預設值就是「新訂單長什麼樣子」。以前逐欄清的版本，每次 Tray
    # 加欄位都要記得來這裡補，漏過 record_session_id（CSV 指到上一單的影片）
    # 和 order_number（舊單號殘留）。
    #
    # single 模式只有一筆訂單、tray 也不是實體餐盤，所以 a / b / all 三種
    # reset 結果相同，只差 log 標籤。誰會呼叫這些、各自還做了什麼，見
    # LogicEngine 的「訂單生命週期」說明。
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        self.reset_all()

    def reset_all(self):
        self._start_new_order(log_tag="RESET-all", reason="reset_all")

    def reset_tray_states(self, tray_id: str):
        # 收尾（remove_tray）也會經過這裡；超時收尾的接單閘門是在那之後才由
        # LogicEngine 關上的，所以這裡無條件打開不會吃掉它
        self._start_new_order(log_tag="RESET-B", reason="reset_tray_states")

    def _start_new_order(self, log_tag: str, reason: str) -> None:
        # 人工 reset 的意思就是「重新偵測」，桌上那張票要能馬上被讀
        self._release_ticket_hold(reason=reason)
        self.trays.clear()
        self._create_tray()   # start_time_str 也在這裡刷新成現在
        logger.info(f"[{log_tag}][single_order] 已重置訂單狀態")
        self._publish_reset_status()

    # ------------------------------------------------------------
    # 超時收尾後的「等傳票離開」閘門
    # ------------------------------------------------------------
    def hold_tickets_until_clear(self, tray_id: str) -> None:
        """
        訂單超時收尾後由 LogicEngine 呼叫：在 ROI 裡連續 ticket_leave_frame 幀
        都沒有傳票之前，忽略所有傳票偵測。人工 reset（a / b / all）會直接放行。

        超時的時候那張傳票多半還放在 ROI 裡。不擋的話，下一幀它就會被重新
        鎖定、OCR、開出一筆一模一樣的訂單，再過 timeout_sec 又寫一列重複的
        CSV。正常收尾（護貝板露出）本來就要求 ROI 裡沒有傳票，所以不需要這道閘門。
        """
        self._holding_tickets = True
        self._ticket_clear_count = 0
        logger.warning(
            f"[order_timeout] tray={tray_id} 超時收尾後暫停接單，"
            f"等 ROI 內傳票連續消失 {self.ticket_leave_frame} 幀再重新偵測"
        )

    def _release_ticket_hold(self, reason: str) -> None:
        if self._holding_tickets:
            self._holding_tickets = False
            self._ticket_clear_count = 0
            logger.info(f"[order_timeout] 恢復接單（{reason}）")

    def _apply_ticket_hold(self, ticket_dets: list) -> list:
        """閘門關著時吞掉這一幀的傳票偵測，並計算 ROI 已連續淨空幾幀"""
        if not self._holding_tickets:
            return ticket_dets
        if ticket_dets:
            self._ticket_clear_count = 0
        else:
            self._ticket_clear_count += 1
            if self._ticket_clear_count >= self.ticket_leave_frame:
                self._release_ticket_hold(
                    reason=f"ROI 內傳票已連續消失 {self._ticket_clear_count} 幀"
                )
        return []

    def _publish_reset_status(self):
        # ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        ts_local = get_now_iso_str(time.time(), utc=False)
        self.bus.publish_system({
            "ts": ts_local,
            "type": "TRAY_RESET",
            "msg": {"tray_id": self.tray_id}
        })
        self.bus.publish_det_status({
            "tray_id": self.tray_id,
            "status": "WAITING_TICKET",
        })

    def _publish_new_order_detected(self):
        # ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        ts_local = get_now_iso_str(time.time(), utc=False)
        self.bus.publish_system({
            "ts": ts_local,
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
        if self.ticket_exclude_roi_polygon is None:
            return ticket_dets

        # 如果 ticket 在 ROI 之外才做辨識
        return [
            d for d in ticket_dets
            if not is_center_in_polygon(
                get_polygon_centroid(d.xyxy), self.ticket_exclude_roi_polygon
            )
        ]
    
    def _filter_ticket_mat_dets(self, ticket_mat_dets: list) -> list:
        """
        預設不過濾,整個畫面只看有沒有偵測到完整露出的 ticket_mat。
        若子類別需要限制 mat 偵測的判斷範圍(例如多 ROI 場景),
        覆寫此方法即可,沿用跟 _filter_ticket_dets 一樣的擴充點模式。
        """
        return ticket_mat_dets
    
    # ------------------------------------------------------------
    # 判斷是否要重製的方法
    # ------------------------------------------------------------
    def _check_finalize_ticket_leave(self, tray: Tray, ticket_dets: list, ticket_mat_dets: list) -> bool:
        """舊邏輯:ticket 連續 missing 超過門檻視為離開"""
        if tray.ticket is not None and tray.ticket.missing_count > 0:
            ticket_left = tray.ticket.missing_count > self.ticket_leave_frame
        else:
            ticket_left = False
        if ticket_left:
            logger.warning(f'傳票消失超過 {self.ticket_leave_frame} 幀, 觸發結束 !')
        return ticket_left

    def _check_finalize_ticket_mat(self, tray: Tray, ticket_dets: list, ticket_mat_dets: list) -> bool:
        """
        新邏輯:用 ticket_mat 完整露出與否判斷一輪訂單是否真的結束,
        避免核對過程中人手遮擋 ticket 造成誤判。

        狀態機:
          idle(order_session_active=False)
            -> mat 不可見 且 ROI 內有 ticket -> 進入 active
          active(order_session_active=True)
            -> mat 連續 N 幀重新完整可見 且 ROI 內已無 ticket -> 觸發一次收尾, 回到 idle

        idle 狀態下絕不觸發收尾, 避免反覆誤清空造成當掉。
        """
        mat_visible_now = len(ticket_mat_dets) > 0
        ticket_present_now = len(ticket_dets) > 0

        if mat_visible_now:
            tray.ticket_mat_stable_frames += 1
        else:
            tray.ticket_mat_stable_frames = 0

        if not tray.order_session_active:
            if not mat_visible_now and ticket_present_now:
                tray.order_session_active = True
                logger.info(f"[ticket_mat] tray={tray.id} mat 被遮擋且偵測到 ticket, 訂單流程開始")
            return False

        if tray.ticket_mat_stable_frames >= self.ticket_mat_confirm_frames and not ticket_present_now:
            logger.warning(
                f"[ticket_mat] tray={tray.id} mat 連續 {tray.ticket_mat_stable_frames} 幀重新完整露出, "
                f"且 ROI 內無殘留 ticket, 觸發結束!"
            )
            tray.order_session_active = False
            tray.ticket_mat_stable_frames = 0
            return True

        return False
    
    def _is_ticket_leave_detection_suspended(self, tray: Tray) -> bool:
        """
        finalize_mode == "ticket_mat" 且訂單進行中(order_session_active=True)時,
        候選(第二張)ticket 暫時不可見屬於人手正常操作,不代表真的離開,
        暫停「真的離開」判定,避免誤觸發 TrayStateMachine 那邊的扣除/清除通知。
        真正的清除,交給 mat 出現時的一次性收尾(整批 reset)處理。

        finalize_mode == "ticket_leave" 或 idle 狀態下,維持原本即時判斷行為。
        """
        return self.finalize_mode == "ticket_mat" and tray.order_session_active

    # ------------------------------------------------------------
    # 每幀更新
    # ------------------------------------------------------------
    def update_tray_positions(self, tray_dets: list, ts_utc: str) -> None:
        pass

    def _advance_stable(self, item: TrackedItem, new_bbox: PolygonXYXY, label: str) -> None:
        """這一幀又對上了同一個物件：穩定幀數 +1。

        still_shift_ratio > 0 時多一個條件：跟上一幀比，中心位移超過框短邊的
        這個比例就算「還在動」，計數歸零。原本只要對得上就 +1，紙被手慢慢推著
        走也會一路數到 n_settle_frame，OCR 就拍到糊掉的字。
        必須在呼叫端把 item.bbox 換成新框之前呼叫。
        """
        if self.still_shift_ratio > 0:
            shift = center_shift_ratio(item.bbox, new_bbox)
            if shift > self.still_shift_ratio:
                if item.stable_frames > 0:
                    # 逐次明細在 DEBUG；次數記在 item 上，併進 [OCR#n ■] 那行的「移動重置 N 次」
                    item.still_resets += 1
                    logger.debug(
                        f"[still] {label} 累積 {item.stable_frames} 幀後移動"
                        f"（位移 {shift:.1%} > {self.still_shift_ratio:.1%}），穩定幀數歸零"
                    )
                item.stable_frames = 0
                return
        item.stable_frames += 1

    def update_ticket_and_stickers(self, ticket_dets: list, sticker_dets: list, ticket_mat_dets: list = None) -> None:
        if self.tray_id not in self.trays:
            self._create_tray()
        tray = self.trays[self.tray_id]
        ticket_dets = self._apply_ticket_hold(self._filter_ticket_dets(ticket_dets))
        ticket_mat_dets = self._filter_ticket_mat_dets(ticket_mat_dets or [])
        # --- ticket ---
        primary_seen_this_frame = False
        matched_extra_tickets = set()  # 存放本幀有被偵測配到的 extra_ticket 物件 id

        for d in ticket_dets:
            t_rect = d.xyxy
            t_xywhr = d.xywhr

            if tray.ticket is not None and iou_poly_poly(t_rect, tray.ticket.bbox) > 0.4:
                self._advance_stable(tray.ticket, t_rect, "ticket")
                tray.ticket.bbox = t_rect
                tray.ticket.xywhr = t_xywhr
                tray.ticket.missing_count = 0
                primary_seen_this_frame = True
                continue

            best_iou, best_et = 0.0, None
            for et in tray.extra_tickets:
                if id(et) in matched_extra_tickets:
                    continue
                val = iou_poly_poly(t_rect, et.bbox)
                if val > best_iou:
                    best_iou, best_et = val, et

            if best_et is not None and best_iou > 0.4:
                self._advance_stable(best_et, t_rect, "extra_ticket")
                best_et.bbox = t_rect
                best_et.xywhr = t_xywhr
                best_et.missing_count = 0
                matched_extra_tickets.add(id(best_et))
                continue

            if tray.ticket is None:
                tray.ticket = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
                self._publish_new_order_detected()
                primary_seen_this_frame = True
            else:
                new_et = TrackedItem(bbox=t_rect, xywhr=t_xywhr)
                tray.extra_tickets.append(new_et)
                matched_extra_tickets.add(id(new_et))
                logger.info("[single_order] 偵測到疑似第二張同單號訂單，開始追蹤穩定度")

        # --- 記錄這一幀「同時被看到」的傳票 ---
        # 一張紙同一時間只會有一個偵測框，所以兩張票曾在同一幀一起出現，就代表
        # 桌上確實是兩張紙。TrayStateMachine 用它分辨「同號的另一張（長單分頁）」
        # 跟「已知那張被拿起來放到別處」—— 後者從頭到尾都不會跟本尊同時出現。
        # 必須在下面「主票頂替」之前做，這裡的 tray.ticket 還是本幀實際配到的那張。
        seen_now = [et for et in tray.extra_tickets if id(et) in matched_extra_tickets]
        if primary_seen_this_frame and tray.ticket is not None:
            seen_now.append(tray.ticket)
        if len(seen_now) > 1:
            for item in seen_now:
                item.seen_with.update(id(other) for other in seen_now if other is not item)

        # --- 主 ticket 本幀沒看到：嘗試用「已確認同單號」且「本幀有出現」的候選頂替 ---
        if tray.ticket is not None and not primary_seen_this_frame:
            promote_candidate = None
            # ticket mat mode 不需要考慮主副 ticket 的問題, 只有 ticket level 要判斷是否 ticket 離開要清除菜單
            if self.finalize_mode != "ticket_mat":
                promote_candidate = next(
                    (et for et in tray.extra_tickets if et.is_ticket_merged and id(et) in matched_extra_tickets),
                    None
                )
            if promote_candidate is not None:
                logger.info("[single_order] 主 ticket 消失，改以已確認同單號的第二張 ticket 接手追蹤")
                old_primary = tray.ticket
                tray.extra_tickets.remove(promote_candidate)
                tray.ticket = promote_candidate
                # 把被頂替下來的舊主 ticket 降級，繼續當作候選追蹤,
                # 而不是直接丟棄——它當初貢獻的品項(打底那份)還留在
                # tray.expected_items 裡，必須繼續追蹤它的存在感，
                # 之後才能:
                #   1) 真的消失時，正確扣回這份貢獻
                #   2) 重新出現時，被辨識成「已處理過」而不會重新送 OCR、重複疊加
                if old_primary.contributed_display:
                    old_primary.is_ticket_merged = True
                    tray.extra_tickets.append(old_primary)

            elif not tray.ticket.is_ocr_busy:
                tray.ticket.stable_frames = 0
                tray.ticket.missing_count += 1

        # --- 其餘候選 ticket 本幀沒被匹配到 ---
        suspend_ticket_leave = self._is_ticket_leave_detection_suspended(tray)
        for et in tray.extra_tickets:
            if id(et) in matched_extra_tickets:
                continue
            if not et.is_ocr_busy:
                et.stable_frames = 0
                et.missing_count += 1
            if not suspend_ticket_leave and et.missing_count > self.sticker_missing_frame:
                et.is_missing = True   # 標記「真的離開」

        tray.extra_tickets = [
            et for et in tray.extra_tickets
            if not (et.is_missing and et.has_notified_missing)   # 兩段式收尾
        ]

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
                self._advance_stable(ts_obj, s_rect, f"sticker({ts_obj.cls_name})")
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

            if ts.missing_count > self.sticker_missing_frame:
                ts.is_missing = True

        tray.stickers = [
            ts for ts in tray.stickers
            if not (ts.is_missing and ts.has_notified_missing)
        ]

        # ------------------------------------------------------------
        # 收尾觸發:判斷方法可透過 finalize_mode 抽換("ticket_leave" / "ticket_mat"),
        # 不管哪種方法, 觸發後對外的訊號都一樣是 tray.missing_count = tray_missing_frame + 1,
        # 下游 LogicEngine._finalize_missing_trays 完全不用改。
        # ------------------------------------------------------------
        finalize_triggered = self._finalize_checker(tray, ticket_dets, ticket_mat_dets)
        if finalize_triggered:
            tray.missing_count = self.tray_missing_frame + 1
        else:
            tray.missing_count = 0