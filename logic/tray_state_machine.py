"""
tray_state_machine.py
=======================
TrayStateMachine 操作 TrayTracker 手上的 `trays` 字典，但不擁有它
（每個方法都把 `trays` 當參數傳進來，而不是自己 new 一份）。

職責：
    - 依 stable_frames 決定要不要送出 OCR task (generate_tasks)
    - 收到 OCR 結果後，呼叫 OrderParser / StickerMatcher 做業務判斷，
      更新 tray.state，並發布對應的 MQTT 狀態事件 (apply_ocr_result)
    - 標記某個 ticket/sticker 已送出 OCR，避免重複送單 (set_ocr_busy)

這一層是「tray 的生命週期事件」跟「訂單/貼紙比對的業務結果」之間的
膠水邏輯，會隨著流程設計調整而變動的機率，通常比底下的追蹤演算法
(TrayTracker) 或比對演算法 (OrderParser/StickerMatcher) 更頻繁。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np
import time
from loguru import logger

from logic.geometry import PolygonXYXY, iou_poly_poly
from logic.models import Tray, TrayState
from logic.order_parser import OrderParser
from logic.sticker_matcher import StickerMatcher
from logic.single_order_tracker import SINGLE_ORDER_ID
from logic.known_class_items import is_known_class_item, resolve_item_name



class TrayStateMachine:
    def __init__(
        self,
        bus,
        order_parser: OrderParser,
        sticker_matcher: StickerMatcher,
        special_cases: Dict[str, List[str]],
        n_settle_frame: int,
        wrong_item_margin: int = 10,
        duplicate_content_similarity_threshold: float = 0.9,
        on_recording_start=None,
    ):
        self.bus = bus
        self.order_parser = order_parser
        self.sticker_matcher = sticker_matcher
        self.special_cases = special_cases
        self.N = n_settle_frame
        self.wrong_item_margin = wrong_item_margin

        # 新增：判斷「同單號第二張候選」是否為追丟重偵測(而非真的第二張票)的相似度門檻
        self.duplicate_content_similarity_threshold = duplicate_content_similarity_threshold

        # 只有「開始錄影」這個事件天然屬於狀態機自己的判斷（票證確認）；
        # 「結束錄影」則不管訂單完成/reset 都收斂在 LogicEngine 那層，
        # 狀態機不需要知道 reset 這件事。
        self.on_recording_start = on_recording_start or (lambda reason="": None)

    def set_ocr_busy(self, trays: Dict[str, Tray], tray_id: str, item_type: str, bbox: PolygonXYXY):
        """由呼叫端確認已將任務送給 OCR 後呼叫，正式鎖定狀態"""
        if tray_id not in trays:
            return

        tray = trays[tray_id]

        if item_type == "ticket" and tray.state == TrayState.WAITING_TICKET:
            if tray.ticket:
                tray.ticket.is_ocr_busy = True
                tray.state = TrayState.CHECKING_TICKET

        elif item_type == "ticket_extra":
            for et in tray.extra_tickets:
                if iou_poly_poly(et.bbox, bbox) > 0.1 and not et.is_ticket_merged:
                    et.is_ocr_busy = True
                    break

        elif item_type == "sticker" and tray.state == TrayState.WAITING_STICKERS:
            for ts in tray.stickers:
                if iou_poly_poly(ts.bbox, bbox) > 0.1:
                    ts.is_ocr_busy = True
                    tray.state = TrayState.CHECKING_STICKERS
                    break

    def generate_tasks(self, trays: Dict[str, Tray]) -> List[dict]:
        """依目前每個 tray 的狀態與穩定度，決定要送出哪些 OCR task"""
        tasks: List[dict] = []

        for tray_id, tray in trays.items():
            if tray.state == TrayState.WAITING_TICKET:
                if tray.ticket and tray.ticket.stable_frames >= self.N and not tray.ticket.is_ocr_busy:
                    tasks.append({
                        "tray_id": tray.id,
                        "type": "ticket",
                        "bbox": tray.ticket.bbox,
                        "xywhr": tray.ticket.xywhr,
                        "cls_name": tray.ticket.cls_name 
                    })

            elif tray.state == TrayState.WAITING_STICKERS:
                for ts in tray.stickers:
                    if is_known_class_item(ts.cls_name):
                        continue  # 醬料包不需要 OCR,交給 resolve_known_class_items 處理
                    if ts.stable_frames >= self.N and not ts.is_checked and not ts.is_ocr_busy:
                        tasks.append({
                            "tray_id": tray.id,
                            "type": "sticker",
                            "bbox": ts.bbox,
                            "xywhr": ts.xywhr,
                            "cls_name": ts.cls_name
                        })
                        break
            # 候選(第二張)ticket：不管目前狀態是 WAITING_STICKERS / CHECKING_STICKERS / COMPLETED
            # 都要繼續看有沒有穩定夠帧、還沒處理過的候選，送去 OCR 判斷單號
            if tray.state in (TrayState.WAITING_STICKERS, TrayState.CHECKING_STICKERS, TrayState.COMPLETED):
                for et in tray.extra_tickets:
                    if et.stable_frames >= self.N and not et.is_ocr_busy and not et.is_ticket_merged:
                        tasks.append({
                            "tray_id": tray.id, "type": "ticket_extra",
                            "bbox": et.bbox, "xywhr": et.xywhr, "cls_name": et.cls_name,
                        })
                        break

        return tasks

    def _calc_multiset_similarity(self, items_a: List[str], items_b: List[str]) -> float:
        """
        計算兩份「已展開品項清單」(例如 expanded_expected_items,一個品項出現幾次
        就代表數量幾份)的相似度,採 multiset 版 Jaccard:
            similarity = |交集(取較小數量)| / |聯集(取較大數量)|

        任一邊為空視為 0 相似(避免「空 vs 空」或「有 vs 無」被誤判成高度相似)。
        """
        if not items_a or not items_b:
            return 0.0

        counter_a, counter_b = Counter(items_a), Counter(items_b)
        all_keys = set(counter_a) | set(counter_b)

        intersection = sum(min(counter_a[k], counter_b[k]) for k in all_keys)
        union = sum(max(counter_a[k], counter_b[k]) for k in all_keys)

        return intersection / union if union else 0.0
    
    def _find_stale_tracking_anchor(self, tray, exclude):
        """
        在「已確認屬於這筆訂單」的既有追蹤物件中(tray.ticket 本身，
        以及已合併過的其他候選 extra_ticket)，找出目前 missing_count
        最高(代表 SingleOrderTracker 本幀沒追到它)的那一個，
        視為被遮擋/快速移動追丟、bbox 卡在原地不動的幽靈物件。

        找到後，這個 anchor 會被 target 的新 bbox 接管位置，
        取代掉暫時冒出來的候選 extra_ticket，避免同一張票
        留下兩個 TrackedItem 互相打架。
        """
        candidates = []
        if tray.ticket is not None and tray.ticket is not exclude:
            candidates.append(tray.ticket)
        for et in tray.extra_tickets:
            if et is exclude or not et.is_ticket_merged:
                continue
            candidates.append(et)

        if not candidates:
            return None

        return max(candidates, key=lambda item: item.missing_count)

    def _flatten_contributed_display(self, contributed_display: List[dict]) -> List[str]:
        """把 [{"品項名": 數量}, ...] 攤平成 ["品項名", "品項名", ...]，方便跟
        expanded_expected_items 做同一種格式的相似度比較。"""
        flat = []
        for d in contributed_display:
            for item_name, count in d.items():
                flat.extend([item_name] * count)
        return flat

    def _find_duplicate_content_candidate(self, tray, new_items: List[str], exclude):
        """
        在「已確認屬於這筆訂單」的既有追蹤物件(tray.ticket 本身，以及
        已合併過的其他候選 extra_ticket)中，逐一比對「新解析出來的內容」
        跟「該物件自己當初貢獻的那一份 contributed_display」的相似度，
        取相似度最高者。

        注意：基準是各自的「單一貢獻」，不是 tray.expected_items 整份
        累積總表——否則被拿去比較的分母會隨著訂單疊加越來越大，導致
        同一張票重複偵測到的內容永遠被稀釋、怎麼樣都達不到門檻。
        """
        candidates = []
        if tray.ticket is not None and tray.ticket is not exclude:
            candidates.append(tray.ticket)
        for et in tray.extra_tickets:
            if et is exclude or not et.is_ticket_merged:
                continue
            candidates.append(et)

        best_candidate, best_similarity = None, 0.0
        for c in candidates:
            contributed_flat = self._flatten_contributed_display(getattr(c, "contributed_display", []) or [])
            similarity = self._calc_multiset_similarity(new_items, contributed_flat)
            if similarity > best_similarity:
                best_candidate, best_similarity = c, similarity

        return best_candidate, best_similarity

    def apply_ocr_result(
        self,
        trays: Dict[str, Tray],
        tray_id: str,
        item_type: str,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool,
        task_bbox: PolygonXYXY,
        last_order_number: str = None,
    ):
        """
        1. 呼叫 OrderParser / StickerMatcher 處理 OCR 結果
        2. 更新 tray 狀態
        3. 發布 mqtt 訊息
        """
        if tray_id not in trays:
            return

        tray = trays[tray_id]
        logger.info(f"[tray state machine] OCR res : \n {rec_res} \n")

        if item_type == "ticket" and tray.state == TrayState.CHECKING_TICKET:
            self._apply_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, last_order_number)

        elif item_type == "ticket_extra":
            self._apply_extra_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, task_bbox)

        elif item_type == "sticker" and tray.state == TrayState.CHECKING_STICKERS:
            self._apply_sticker_result(tray, tray_id, dt_boxes, rec_res, is_flip, task_bbox)

    # ------------------------------------------------------------
    # 內部細節
    # ------------------------------------------------------------
    def _expand_parsed_items(self, parsed_items):
        """把 OrderParser 解析出的品項，展開 special_cases（套餐拆子品項）。"""
        expanded_expected_items = []
        expected_items_list = []
        for item_name, item_num in parsed_items:
            if item_name in self.special_cases:
                for sub_item in self.special_cases[item_name]:
                    expanded_expected_items.extend([sub_item] * item_num)
                    expected_items_list.append({sub_item: item_num})
            else:
                expanded_expected_items.extend([item_name] * item_num)
                expected_items_list.append({item_name: item_num})
        return expanded_expected_items, expected_items_list

    def _apply_ticket_result(self, tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, last_order_number=None):
        if tray.ticket:
            tray.ticket.is_ocr_busy = False

        parsed_items, order_number = self.order_parser.parse(frame_crop, dt_boxes, rec_res, is_flip)

        if parsed_items:
            expanded_expected_items, expected_items_list = self._expand_parsed_items(parsed_items)

            tray.expected_items = expanded_expected_items
            tray.expected_items_display = expected_items_list 
            tray.state = TrayState.WAITING_STICKERS
            tray.ticket_crop = frame_crop
            order_number = "000" if order_number is None else order_number
            tray.order_number = order_number

            # # 這裡要用複製(list(...))，不能直接指到同一個 list 物件，
            # 否則之後 tray.expected_items_display.extend(...) 會連帶
            # 污染 tray.ticket.contributed_display，造成扣除時範圍算錯
            tray.ticket.contributed_display = list(expected_items_list)
            tray.ticket.is_ticket_merged = True

            self.bus.publish_det_status({
                "tray_id": tray_id, "status": "TICKET_READY",
                "expected_items": expected_items_list, "order_number": order_number,
            })
            self.on_recording_start(reason=f"ticket_confirmed:{tray_id}") # 開始錄影
        else:
            if tray.ticket:
                tray.ticket.stable_frames = 0
            tray.state = TrayState.WAITING_TICKET

    def _apply_extra_ticket_result(self, tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, task_bbox):
        target = None
        for et in tray.extra_tickets:
            if iou_poly_poly(et.bbox, task_bbox) > 0.1:
                et.is_ocr_busy = False
                target = et
                break
        if target is None:
            return

        if target.is_ticket_merged:
            logger.info(
                f"[tray_state_machine] tray={tray_id} 候選 ticket 早已合併過(is_ticket_merged=True)，"
                f"視為遲到的重複 OCR 結果，忽略內容、僅確認狀態"
            )
            target.missing_count = 0
            target.is_missing = False
            target.has_notified_missing = False
            return

        parsed_items, order_number = self.order_parser.parse(frame_crop, dt_boxes, rec_res, is_flip)

        if order_number is None or order_number != tray.order_number:
            logger.warning(f"[tray_state_machine] 第二張 ticket 單號 {order_number} 與目前訂單 {tray.order_number} 不同，忽略合併")
            tray.extra_tickets.remove(target)  # 直接淘汰，避免每幀重送 OCR
            return

        expanded_expected_items, expected_items_list = self._expand_parsed_items(parsed_items)

        # --- 改為跟「既有各貢獻者各自的內容」比對，而非跟訂單累積總表比對 ---
        dup_candidate, similarity = self._find_duplicate_content_candidate(
            tray, expanded_expected_items, exclude=target
        )
        logger.info(
            f"[tray_state_machine] 候選 ticket 單號 {order_number} 內容相似度={similarity:.2f} "
            f"(比對對象={'tray.ticket' if dup_candidate is tray.ticket else ('extra_ticket' if dup_candidate else 'None')}, "
            f"門檻={self.duplicate_content_similarity_threshold})"
        )

        if dup_candidate is not None and similarity >= self.duplicate_content_similarity_threshold:
            anchor = self._find_stale_tracking_anchor(tray, exclude=target)
            # 優先用「內容比對出的那個候選」當接管對象；找不到才退回用 missing_count 找
            anchor = dup_candidate if dup_candidate.missing_count > 0 else (anchor or dup_candidate)

            logger.info(
                f"[tray_state_machine] tray={tray_id} 判定為追丟重偵測(同一張票)，"
                f"位置由 target(bbox={target.bbox}) 接管回原本卡住的追蹤物件，並移除暫時候選"
            )
            anchor.bbox = target.bbox
            anchor.xywhr = target.xywhr
            anchor.missing_count = 0
            anchor.is_missing = False
            anchor.has_notified_missing = False
            tray.extra_tickets.remove(target)
            return

        # --- 相似度不足門檻 → 視為真正的第二張同單號票，維持原本疊加邏輯 ---

        tray.expected_items.extend(expanded_expected_items) # ex: ["菜單名"]
        tray.expected_items_display.extend(expected_items_list) # ex: [{"菜單名": 數量}]
        target.contributed_display = expected_items_list

        target.is_ticket_merged = True  # 保留在 extra_tickets 裡，供之後「主 ticket 消失頂替」使用

        if tray.state == TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS  # 加了新品項，不再算完成

        logger.info(f"[tray_state_machine] ticket 單號 {order_number} 品項追加合併完成, 並發送資訊: {tray.expected_items_display}")
        # 直接複用 TICKET_READY，整包覆寫 expected_items，前端不用新增事件處理
        self.bus.publish_det_status({
            "tray_id": tray_id,
            "status": "TICKET_READY",
            "expected_items": tray.expected_items_display,
            "order_number": tray.order_number,
        })

    def _apply_sticker_result(self, tray: Tray, tray_id: str, dt_boxes, rec_res, is_flip, task_bbox):
        matched_item = None
        match_status = "UNRECOGNIZED"

        if len(rec_res) > 0:
            matched_item, match_status = self.sticker_matcher.match(
                rec_res, dt_boxes, is_flip, tray.expected_items, tray.checked_items
            )

        target_ts = None
        for ts in tray.stickers:
            if iou_poly_poly(ts.bbox, task_bbox) > 0.1:
                ts.is_ocr_busy = False
                target_ts = ts
                break

        self._settle_sticker_match(tray, tray_id, target_ts, matched_item, match_status)

    def _settle_sticker_match(self, tray: Tray, tray_id: str, ts, matched_item, match_status):
        """
        比對結果的共用收尾:不管身份是 OCR 得來或分類器直讀得來,
        「核對成功/失敗後該做什麼」都是同一套業務邏輯。
        """
        if ts is not None:
            if match_status == "MATCHED":
                ts.is_checked = True
                ts.item_name = matched_item
            else:
                ts.stable_frames = 0
                if match_status == "WRONG_ITEM":
                    ts.is_wrong_item = True
                    ts.item_name = matched_item
                    ts.last_wrong_notify_ts = time.time()

        if match_status == "MATCHED":
            tray.checked_items.append(matched_item)
            check_counts = Counter(tray.checked_items)
            items_list = [{item: count} for item, count in check_counts.items()]
            self.bus.publish_det_status({
                "tray_id": tray_id,
                "status": "ITEM_CHECKED",
                "items": items_list
            })

            has_unresolved_wrong_item = any(t.is_wrong_item for t in tray.stickers)
            if len(tray.checked_items) == len(tray.expected_items) and not has_unresolved_wrong_item:
                tray.state = TrayState.COMPLETED
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TRAY_COMPLETED",
                    "items": items_list
                })

        elif match_status == "WRONG_ITEM":
            self.bus.publish_det_status({
                "tray_id": tray_id,
                "status": "WRONG_ITEM_DETECTED",
                "items": [{matched_item: 1}]
            })

        if tray.state != TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS

    def reopen_sticker_for_recheck(self, tray: Tray, tray_id: str, ts) -> None:
        """
        貼紙已核對過,但像素內容比對確認已被替換(例如被疊上新餐點)。
        清空核對狀態、扣回原本記帳的品項,讓它重新走一次穩定度累積,
        交給既有的 generate_tasks 重新送 OCR——處理方式跟「全新貼紙
        出現」完全一樣,不需要另外寫一套辨識流程。
        """
        if ts.is_checked and ts.item_name and ts.item_name in tray.checked_items:
            tray.checked_items.remove(ts.item_name)
            ### 暫時不發送減少
            # check_counts = Counter(tray.checked_items)
            # items_list = [{item: count} for item, count in check_counts.items()]
            # self.bus.publish_det_status({
            #     "tray_id": tray_id,
            #     "status": "ITEM_CHECKED",
            #     "items": [{ts.item_name: -1}]
            # })

        ts.is_checked = False
        ts.item_name = None
        ts.stable_frames = 0
        ts.content_ref_image = None
        ts.content_check_counter = 0
        ts.content_mismatch_streak = 0

        if tray.state == TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS

        logger.info(f"[內容變更偵測] tray={tray_id} 貼紙內容疑似已被替換或物件堆疊,重新排入辨識佇列")
    
    def resolve_known_class_items(self, trays: Dict[str, Tray]) -> None:
        """
        處理「類別即身份」的品項(醬料包等):穩定度到門檻後直接跟訂單比對,
        不經過 PaddleOCR,避免不必要的 OCR 呼叫與畫面閃爍。

        WRONG_ITEM 使用比一般門檻更高的穩定度(self.N + margin),因為
        誤發 WRONG_ITEM 警訊的代價(前端跳出不必要的警示)遠高於晚一點
        才判定成功的代價;而 mmrotate 分類抖動造成的誤判通常撐不過這個
        延長門檻,會在達標前就因為 missing_count 超標被 tracker 清掉。
        """
        for tray_id, tray in trays.items():
            if tray.state != TrayState.WAITING_STICKERS:
                continue

            for ts in tray.stickers:
                item_name = resolve_item_name(ts.cls_name)
                if item_name is None or ts.is_checked:
                    continue

                remaining = Counter(tray.expected_items) - Counter(tray.checked_items)
                will_match = remaining.get(item_name, 0) > 0
                required_frames = self.N if will_match else (self.N + self.wrong_item_margin)

                if item_name is None or ts.is_checked:
                    continue
                if ts.stable_frames < required_frames:
                    continue

                matched_item, match_status = self.sticker_matcher.match_known_item(
                    item_name, tray.expected_items, tray.checked_items
                )
                logger.info(f"[tray state machine] 已知物件 {matched_item}, 匹配狀況: {match_status}")
                self._settle_sticker_match(tray, tray_id, ts, matched_item, match_status) 

    def resolve_wrong_item_removal(self, trays: Dict[str, Tray]) -> None:
        """
        每幀檢查:被標記為 WRONG_ITEM 的貼紙,一旦被 tracker 確認「真的離開」
        (is_missing=True),就解除警示狀態,並重新檢查該 tray 是否可以轉為 COMPLETED
        ——因為湊滿核對數量的當下,可能正好被這個尚未移除的警示卡住過。
        """
        for tray_id, tray in trays.items():
            resolved_any = False
            for ts in tray.stickers:
                if ts.is_wrong_item and ts.is_missing and not ts.has_notified_missing:
                    ts.has_notified_missing = True  # 讓 tracker 下一幀清掉這筆幽靈紀錄
                    resolved_any = True
                    # self.bus.publish_det_status({
                    #     "tray_id": tray_id,
                    #     "status": "WRONG_ITEM_RESOLVED",
                    #     "items": [],
                    # })
                    logger.info(f"[tray_state_machine] 錯誤菜單已移除 tray={tray_id}")

            if not resolved_any:
                continue

            has_unresolved_wrong_item = any(t.is_wrong_item and not t.has_notified_missing for t in tray.stickers)
            if (tray.state == TrayState.WAITING_STICKERS
                    and tray.expected_items
                    and len(tray.checked_items) == len(tray.expected_items)
                    and not has_unresolved_wrong_item):
                tray.state = TrayState.COMPLETED
                check_counts = Counter(tray.checked_items)
                items_list = [{item: count} for item, count in check_counts.items()]
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TRAY_COMPLETED",
                    "items": items_list
                })

    def handle_missing_items(self, trays: Dict[str, Tray]) -> None:
        """
        每幀呼叫：檢查是否有貼紙確認消失，發送資訊給前端，並將對應已核對品項扣除
        """
        for tray_id, tray in trays.items():
            # 收集這一個 tray 裡面，這一幀剛確認消失、且還沒通知外部的貼紙, wrong_item 交給後面的 resolve_wrong_item_removal 處理
            missing_stickers = [
                ts for ts in tray.stickers
                if ts.is_missing and not ts.has_notified_missing and not ts.is_wrong_item
            ]
            
            if not missing_stickers:
                continue
                
            # 假設要做比較細緻的業務處理：
            # 如果這張貼紙當初是核對成功的，它代表某個餐點，消失了要從 checked_items 扣除
            # (這裡需要 StickerMatcher 在 match 成功時，把品項名稱綁定在 TrackedItem 上，
            #  例如 ts.item_name = matched_item。假設你有做這個擴充：)
            
            for ts in missing_stickers:
                ts.has_notified_missing = True # 標記已通知，下一幀 Tracker 就會洗掉它
                
                # 舉例：如果該貼紙有綁定品項名稱，且之前是核對過的
                if ts.is_checked and hasattr(ts, 'item_name') and ts.item_name:
                    item_name = ts.item_name
                    
                    # 從記憶中移除該品項
                    if item_name in tray.checked_items:
                        tray.checked_items.remove(item_name)
                    
                    logger.info(f"[tray_state_machine] 貼紙消失 tray={tray_id}, 品項={item_name} 離開，通知前端扣減")
                    
                    # 發送 MQTT 給前端：塞入你說的數量 -1
                    self.bus.publish_det_status({
                        "tray_id": tray_id,
                        "status": "ITEM_CHECKED",  # 或者保持 ITEM_CHECKED 但數量給負數
                        "items": [{item_name: -1}] # 滿足前端直接加總的邏輯
                    })
                    
                    # 既然餐點少拿了，狀態要從 COMPLETED 退回 WAITING_STICKERS
                    if tray.state == TrayState.COMPLETED:
                        tray.state = TrayState.WAITING_STICKERS

    def heartbeat_wrong_items(self, trays: Dict[str, Tray], heartbeat_interval: float = 1.5) -> None:
        """
        每幀呼叫:對於目前處於 WRONG_ITEM 警示中、且尚未被判定為真的離開
        (is_missing)的貼紙,以實際時間為準定期重發通知,不受判定週期
        (stable_frames 重新累積)影響,確保前端不會因為判定間隔過長
        而誤判成「已經沒有了」。
        """
        now = time.time()
        for tray_id, tray in trays.items():
            for ts in tray.stickers:
                if not ts.is_wrong_item or ts.is_missing:
                    continue
                if now - ts.last_wrong_notify_ts < heartbeat_interval:
                    continue

                item_name = ts.item_name or resolve_item_name(ts.cls_name) or "unknown"
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "WRONG_ITEM_DETECTED",
                    "items": [{item_name: 1}]
                })
                ts.last_wrong_notify_ts = now
    
    def resolve_missing_extra_tickets(self, trays: Dict[str, Tray]) -> None:
        """
        每幀檢查:候選(第二張)ticket 一旦被 tracker 判定「真的離開」(is_missing=True)
        且尚未通知過:
        - 如果它之前合併成功過(is_ticket_merged=True)，代表當初已經把品項
            加進 tray.expected_items / expected_items_display，現在紙離開了
            就要把當初加進去的那份扣回來——避免「隱藏菜單」：這張紙下次又被
            放回來重新穩定、重新 OCR，會被當成全新候選重新走一次合併流程，
            不會因為殘留的舊資料造成無限疊加。
        - 扣除後重新發送一次 TICKET_READY，讓前端畫面同步移除這批品項。
        - 標記 has_notified_missing=True，讓 tracker 下一幀清掉這筆候選。
        """
        for tray_id, tray in trays.items():
            for et in tray.extra_tickets:
                if not et.is_missing or et.has_notified_missing:
                    continue

                et.has_notified_missing = True  # 讓 tracker 下一幀清掉這筆候選

                if not et.is_ticket_merged or not et.contributed_display:
                    continue  # 從沒合併成功過，沒有品項要扣

                # 把「這張候選當初貢獻的顯示清單」現場展開回攤平清單
                contributed_flat = []
                for d in et.contributed_display:
                    for item_name, count in d.items():
                        contributed_flat.extend([item_name] * count)

                # 攤平清單：用 Counter 精準扣除這張紙當初貢獻的數量，
                # 避免誤扣其他張紙貢獻的同名品項
                removed_counter = Counter(contributed_flat)
                new_expected_items = []
                for item in tray.expected_items:
                    if removed_counter.get(item, 0) > 0:
                        removed_counter[item] -= 1
                        continue
                    new_expected_items.append(item)
                tray.expected_items = new_expected_items

                # 顯示清單：用物件身份(id)精準移除當初 append 進去的那幾筆 dict，
                # 不影響其他來源（同名但不同來源的品項不會被誤刪）
                contributed_ids = {id(d) for d in et.contributed_display}
                tray.expected_items_display = [
                    d for d in tray.expected_items_display if id(d) not in contributed_ids
                ]

                logger.warning(
                    f"[tray_state_machine] tray={tray_id} 候選 ticket 離開，"
                    f"扣回當初合併的品項: {et.contributed_display}"
                )

                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TICKET_READY",
                    "expected_items": tray.expected_items_display,
                    "order_number": tray.order_number,
                })

                # 扣除後可能剛好湊滿核對數量，補一次 COMPLETED 判斷
                has_unresolved_wrong_item = any(t.is_wrong_item for t in tray.stickers)
                if (tray.expected_items
                        and tray.state != TrayState.COMPLETED
                        and len(tray.checked_items) == len(tray.expected_items)
                        and not has_unresolved_wrong_item):
                    tray.state = TrayState.COMPLETED
                    check_counts = Counter(tray.checked_items)
                    items_list = [{item: count} for item, count in check_counts.items()]
                    self.bus.publish_det_status({
                        "tray_id": tray_id,
                        "status": "TRAY_COMPLETED",
                        "items": items_list
                    })