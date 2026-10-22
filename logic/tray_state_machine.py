"""
tray_state_machine.py
=======================
TrayStateMachine 操作 TrayTracker 手上的 `trays` 字典，但不擁有它
（每個方法都把 `trays` 當參數傳進來，而不是自己 new 一份）。

職責：
    - 依 stable_frames 決定要不要送出 OCR task、同時只送哪一個 (generate_tasks)
    - 標記某個 ticket/sticker 已送出 OCR，避免重複送單 (set_ocr_busy)
    - 收到 OCR 結果後，依物件類別交給對應的 reader 解讀並套用 (apply_ocr_result)
    - 每幀的生命週期檢查：貼紙離開、錯誤品項移除、警示心跳、第二張傳票離開、醬料包直讀

各類別的結果怎麼解讀、怎麼套用到訂單，在 logic/readers/：
    readers/ticket.py   TicketReader   大傳票（主票開單、第二張傳票合併）
    readers/sticker.py  StickerReader  餐點貼紙（比對、錯誤品項確認、完成判定）
傳票文字的解析器依品牌放在 logic/yoshinoya/、logic/szm/（見 readers/base.py）。
傳票品項怎麼換算成待核對的貼紙（兩個 reader 共用）在 logic/order_requirements.py。

這一層是「tray 的生命週期事件」跟「訂單/貼紙比對的業務結果」之間的
膠水邏輯，會隨著流程設計調整而變動的機率，通常比底下的追蹤演算法
(TrayTracker) 或比對演算法 (OrderParser/StickerMatcher) 更頻繁。
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import time
from loguru import logger

from logic.geometry import PolygonXYXY
from logic.models import Tray, TrayState
from logic.order_requirements import OrderRequirements
from logic.readers.base import TicketParser, find_task_item
from logic.readers.sticker import StickerReader, blocks_completion
from logic.readers.sticker_matcher import StickerMatcher
from logic.readers.ticket import TicketReader


class TrayStateMachine:
    def __init__(
        self,
        bus,
        order_parser: TicketParser,
        sticker_matcher: StickerMatcher,
        menu_variants: Dict[str, List[List[str]]],
        n_settle_frame: int,
        menu_distinct: List[List[str]] = None,
        wrong_item_margin: int = 10,
        on_recording_start=None,
        object_class_items=None,
        modifier_knowledge=None,
        max_item_qty: int = 0,
        wrong_item_confirm: int = 0,
        wrong_item_strong_score: float = 0.0,
        wrong_item_strong_conf: float = 0.9,
    ):
        from logic.known_class_items import CLASS_TO_ITEM_NAME
        self.object_class_items = dict(CLASS_TO_ITEM_NAME if object_class_items is None else object_class_items)
        self.bus = bus
        self.order_parser = order_parser
        self.sticker_matcher = sticker_matcher
        # 訂單需求（傳票品項 -> 待核對的貼紙組合、完成判定），兩個 reader 共用同一份
        self.requirements = OrderRequirements(menu_variants, menu_distinct, modifier_knowledge)

        self.N = n_settle_frame
        self.wrong_item_margin = wrong_item_margin

        # 只有「開始錄影」這個事件天然屬於狀態機自己的判斷（票證確認）；
        # 「結束錄影」則不管訂單完成/reset 都收斂在 LogicEngine 那層，
        # 狀態機不需要知道 reset 這件事。
        self.on_recording_start = on_recording_start or (lambda reason="": None)

        # 各類別 OCR 結果的解讀與套用（見 logic/readers/）
        self.ticket_reader = TicketReader(
            bus, order_parser, self.requirements,
            max_item_qty=max_item_qty, on_recording_start=self.on_recording_start,
        )
        self.sticker_reader = StickerReader(
            bus, sticker_matcher, self.requirements,
            wrong_item_confirm=wrong_item_confirm,
            wrong_item_strong_score=wrong_item_strong_score,
            wrong_item_strong_conf=wrong_item_strong_conf,
        )

    def set_ocr_busy(self, trays: Dict[str, Tray], tray_id: str, item_type: str, bbox: PolygonXYXY,
                     target=None):
        """由呼叫端確認已將任務送給 OCR 後呼叫，正式鎖定狀態。target = task["target"]"""
        if tray_id not in trays:
            return

        tray = trays[tray_id]

        if item_type == "ticket" and tray.state == TrayState.WAITING_TICKET:
            if tray.ticket:
                tray.ticket.is_ocr_busy = True
                self._mark_ocr_sent(tray.ticket)
                tray.state = TrayState.CHECKING_TICKET

        elif item_type == "ticket_extra":
            et = find_task_item(tray.extra_tickets, bbox, target,
                                skip=lambda it: it.is_ticket_merged)
            if et is not None:
                et.is_ocr_busy = True
                self._mark_ocr_sent(et)

        elif item_type == "sticker" and tray.state == TrayState.WAITING_STICKERS:
            ts = find_task_item(tray.stickers, bbox, target)
            if ts is not None:
                ts.is_ocr_busy = True
                self._mark_ocr_sent(ts)
                tray.state = TrayState.CHECKING_STICKERS

    @staticmethod
    def _mark_ocr_sent(item) -> None:
        """計時 log 用：送出一次 OCR 後，讀取次數 +1、移動重置次數重新算"""
        item.ocr_reads += 1
        item.still_resets = 0

    @staticmethod
    def _task_trace(item, ready_mono: Optional[float] = None) -> dict:
        """跟著 OCR task 走的計時資訊，app.py 在 [OCR#n ■] 那行印出來。

        印的是「這一輪」：從這一輪開始等，到這次推論拿到結果。
          第 1 輪：起點是物件出現。ready_mono 是它最早「可以」送 OCR 的時間 ——
                   貼紙（和第二張傳票）要等主傳票確認（TICKET_READY）才會讀，比傳票
                   早放上桌的貼紙在那之前只是被追蹤，追蹤框可能已經好幾分鐘（上一單
                   收尾後就一直放在桌上），所以取兩者較晚的那個。
          第 2 輪起：起點是上一輪結果套用回來的時間（判錯後要重新等穩定）。
        """
        if item.ocr_reads > 0 and item.last_result_mono is not None:
            return {"cycle_start_mono": item.last_result_mono, "cycle_origin": "上一輪結果",
                    "still_resets": item.still_resets, "read_no": item.ocr_reads + 1}
        start = item.first_seen_mono
        early = ready_mono is not None and ready_mono > start
        return {
            "cycle_start_mono": ready_mono if early else start,
            "cycle_origin": "傳票確認" if early else "出現",
            "waited_before_ready": (ready_mono - start) if early else 0.0,
            "still_resets": item.still_resets,
            "read_no": item.ocr_reads + 1,
        }

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
                        "cls_name": tray.ticket.cls_name,
                        "trace": self._task_trace(tray.ticket),
                    })

            elif tray.state == TrayState.WAITING_STICKERS:
                for ts in tray.stickers:
                    if ts.cls_name in self.object_class_items:
                        continue  # 醬料包不需要 OCR,交給 resolve_known_class_items 處理
                    if ts.stable_frames >= self.N and not ts.is_checked and not ts.is_ocr_busy:
                        tasks.append({
                            "tray_id": tray.id,
                            "type": "sticker",
                            "bbox": ts.bbox,
                            "xywhr": ts.xywhr,
                            "cls_name": ts.cls_name,
                            "trace": self._task_trace(ts, tray.order_started_mono),
                            "target": ts,   # 標記 / 套用結果都認這個物件，見 readers/base.py::find_task_item
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
                            "trace": self._task_trace(et, tray.order_started_mono),
                            "target": et,
                        })
                        break

        return tasks

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
        target=None,
    ):
        """
        依物件類別交給對應的 reader：解讀 OCR 結果（OrderParser / StickerMatcher）、
        更新 tray 狀態、發布 mqtt 訊息。
        target = 送出時 task["target"]，結果只套用到這個物件（見 readers/base.py::find_task_item）
        """
        if tray_id not in trays:
            return

        tray = trays[tray_id]
        logger.info(f"[tray state machine] OCR res : \n {rec_res} \n")

        if item_type == "ticket" and tray.state == TrayState.CHECKING_TICKET:
            self.ticket_reader.apply_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip)

        elif item_type == "ticket_extra":
            self.ticket_reader.apply_extra_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip,
                                                         task_bbox, target)

        elif item_type == "sticker" and tray.state == TrayState.CHECKING_STICKERS:
            self.sticker_reader.apply_sticker_result(tray, tray_id, dt_boxes, rec_res, is_flip,
                                                     task_bbox, target)

    def resolve_effective_combo(self, tray, counts) -> Counter:
        """CSV / 報表用的「最能解釋現場」的組合假設，見 OrderRequirements.resolve_effective_combo"""
        return self.requirements.resolve_effective_combo(tray, counts)

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
                item_name = self.object_class_items.get(ts.cls_name)
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
                self.sticker_reader.settle(tray, tray_id, ts, matched_item, match_status) 

    def resolve_wrong_item_removal(self, trays: Dict[str, Tray]) -> None:
        """
        每幀檢查:被標記為 WRONG_ITEM 的貼紙,一旦被 tracker 確認「真的離開」
        (is_missing=True),就解除警示狀態,並重新檢查該 tray 是否可以轉為 COMPLETED
        ——因為湊滿核對數量的當下,可能正好被這個尚未移除的警示卡住過。

        完成進度必須用 confirmed_counts,不可以用 checked_items
        ------------------------------------------------------
        * confirmed_counts 是已送給前端的歷史最高確認數。品項一旦達標,即使
          實體之後短暫漏偵測、被遮擋或拿起再放回,完成進度仍然有效。
        * checked_items 只表示 tracker 當下仍看得到的已確認實體,會隨物件
          消失而減少,不適合用來判斷訂單是否曾經完成。

        2026-09-22 實測的卡住情境（傳票 519：牛皿FP4食/牛丼並/お新香/キムチ
        各 1,現場放了兩份牛丼並）:第一份牛丼並 MATCHED 進 confirmed_counts,
        第二份被標記 WRONG_ITEM;其餘品項陸續達標,訂單先被 wrong item 擋住。
        操作員移開多餘的那份時,tracker 有可能把「消失」歸給那個**已核對**的
        TrackedItem（兩張貼紙幾乎同位置,IoU 配對無法保證配到哪一個）,
        於是 handle_missing_items() 把牛丼並從 checked_items 扣掉。此時改看
        checked_items 就永遠湊不滿 -> 訂單永久卡在 WAITING_STICKERS。

        用 self.requirements.is_order_satisfied(tray, tray.confirmed_counts) 而不是自己
        重寫一份聯集比較,是為了跟 StickerReader.settle() 的一般完成路徑走
        **完全同一個判斷**:多重組合假設（同一個傳票名稱對到多個貼紙組合）
        只要任一版被湊齊就算完成。若在這裡改成比對 expected_items 聯集,
        多變體訂單會回頭去等一份永遠不會來的品項。
        """
        for tray_id, tray in trays.items():
            resolved_any = False
            for ts in tray.stickers:
                if blocks_completion(ts) and ts.is_missing and not ts.has_notified_missing:
                    ts.has_notified_missing = True  # 讓 tracker 下一幀清掉這筆幽靈紀錄
                    resolved_any = True
                    if ts.is_wrong_item:
                        logger.info(f"[tray_state_machine] 錯誤菜單已移除 tray={tray_id}")
                    else:
                        # 確認中就離開：沒成立，什麼都不記，只是不再擋完成
                        logger.info(f"[wrong_confirm] tray={tray_id} 疑似錯誤的貼紙在成立前離開，不記錯誤 {ts.wrong_reads}")

            if not resolved_any:
                continue

            has_unresolved_wrong_item = any(blocks_completion(t) and not t.has_notified_missing for t in tray.stickers)
            all_completed = self.requirements.is_order_satisfied(tray, tray.confirmed_counts)

            logger.info(
                f"[tray_state_machine] 錯誤菜單移除後重新判定 tray={tray_id}, "
                f"state={tray.state.name}, 完成={all_completed}, "
                f"仍有未解除警示={has_unresolved_wrong_item}, "
                f"confirmed={dict(tray.confirmed_counts)}, "
                f"checked={dict(Counter(tray.checked_items))}"
            )

            if (tray.state == TrayState.WAITING_STICKERS
                    and tray.expected_items
                    and all_completed
                    and not has_unresolved_wrong_item):
                tray.state = TrayState.COMPLETED
                # 對外回報也統一用 confirmed_counts,跟 StickerReader.settle
                # 送出的 TRAY_COMPLETED 同一個來源,前端不會收到兩種語意的數量
                items_list = [
                    {item: count}
                    for item, count in tray.confirmed_counts.items()
                ]
                self.bus.publish_det_status({
                    "tray_id": tray_id,
                    "status": "TRAY_COMPLETED",
                    "items": items_list
                })

    def handle_missing_items(self, trays: Dict[str, Tray]) -> None:
        """
        checked_items：
            表示目前畫面實際存在的已確認餐點。

        confirmed_counts：
            表示歷史最高確認進度，不因餐點離開而下降。
        # 不修改 confirmed_counts
        # 不發 ITEM_CHECKED -1
        # 不取消 locked
        # 不讓 COMPLETED 回退
        """
        for tray_id, tray in trays.items():

            missing_stickers = [
                ts for ts in tray.stickers
                if (
                    ts.is_missing
                    and not ts.has_notified_missing
                    # 錯誤品項（含確認中的）離開交給 resolve_wrong_item_removal，它會重判完成
                    and not blocks_completion(ts)
                )
            ]

            if not missing_stickers:
                continue

            for ts in missing_stickers:

                ts.has_notified_missing = True

                if not ts.is_checked or not ts.item_name:
                    continue

                item_name = ts.item_name

                # ---------------------------------------------
                # checked_items 是「目前存在數」
                # 所以實體消失一定要扣
                # ---------------------------------------------
                if item_name in tray.checked_items:
                    tray.checked_items.remove(item_name)

                visible_count = Counter(
                    tray.checked_items
                ).get(item_name, 0)

                confirmed_count = tray.confirmed_counts.get(
                    item_name, 0
                )

                logger.info(
                    f"[tray_state_machine] 已確認餐點離開 "
                    f"tray={tray_id}, "
                    f"item={item_name}, "
                    f"visible={visible_count}, "
                    f"confirmed={confirmed_count}"
                )

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
                # Unknown 不擋完成，訂單完成後就不再提醒
                alerting = ts.is_wrong_item or (ts.is_unknown_item and tray.state != TrayState.COMPLETED)
                if not alerting or ts.is_missing:
                    continue
                if now - ts.last_wrong_notify_ts < heartbeat_interval:
                    continue

                item_name = ts.item_name or self.object_class_items.get(ts.cls_name) or "unknown"
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

                # 組合假設：這張票已標記 has_notified_missing，重算時會自動被排除
                self.requirements.recompute_expected_combos(tray)

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

                # 扣除後可能剛好湊滿核對數量，補一次 COMPLETED 判斷。
                # 跟 resolve_wrong_item_removal 同一個理由用 confirmed_counts
                # 而不是 checked_items：任一已達標品項剛好漏偵測，就會讓訂單
                # 永久卡住。三條 TRAY_COMPLETED 路徑（一般完成 / 錯誤菜單移除 /
                # 候選票離開）現在都用同一個來源，前端不會收到兩種語意的數量。
                has_unresolved_wrong_item = any(blocks_completion(t) for t in tray.stickers)
                if (tray.expected_items
                        and tray.state != TrayState.COMPLETED
                        and self.requirements.is_order_satisfied(tray, tray.confirmed_counts)
                        and not has_unresolved_wrong_item):
                    tray.state = TrayState.COMPLETED
                    items_list = [
                        {item: count}
                        for item, count in tray.confirmed_counts.items()
                    ]
                    self.bus.publish_det_status({
                        "tray_id": tray_id,
                        "status": "TRAY_COMPLETED",
                        "items": items_list
                    })
