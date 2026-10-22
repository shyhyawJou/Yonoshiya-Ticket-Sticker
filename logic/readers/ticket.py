"""
readers/ticket.py
=================
大傳票的 OCR 結果怎麼套用到訂單：主票開單（TICKET_READY、開始錄影）、
第二張傳票（同一筆訂單的另一張紙才合併，被移動的同一張只接回位置）、數量異常整張作廢。

傳票文字本身怎麼解析是 TicketParser 的事（各品牌的 order_parser.py，見 readers/base.py），這裡拿它的
(品項, 單號) 結果做業務判斷；品項怎麼換算成待核對的貼紙交給 OrderRequirements。
內容是從 TrayStateMachine 原封不動搬出來的。
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from logic.models import TrayState
from logic.order_requirements import OrderRequirements
from logic.readers.base import TicketParser, find_task_item
from utils.clip_video import get_now_iso_str


def order_number_key(order_number) -> Optional[tuple]:
    """
    把單號轉成可比較的形式：'50' -> (50,)、'50-01' -> (50, 1)。

    以數字比較，'50-1' 跟 '50-01'、'050' 跟 '50' 視為同一個單號（OCR 對前導零
    不穩）。讀不到（None）或狀態機的「未確認」佔位 '000' 回傳 None —— 未知的
    單號不能拿來判斷任何事。
    """
    if not order_number or order_number == "000":
        return None
    try:
        return tuple(int(part) for part in str(order_number).split("-"))
    except ValueError:
        return None


class TicketReader:
    def __init__(self, bus, order_parser: TicketParser, requirements: OrderRequirements,
                 max_item_qty: int = 0, on_recording_start=None):
        self.bus = bus
        self.order_parser = order_parser
        self.requirements = requirements
        # 傳票任一品項數量超過這個值就整張作廢重讀（見 find_anomaly）。0 = 關閉
        self.max_item_qty = int(max_item_qty or 0)
        # 主票確認（TICKET_READY）當下開始事件錄影
        self.on_recording_start = on_recording_start or (lambda reason="": None)

    def ticket_contributors(self, tray, exclude) -> list:
        """這筆訂單已經確認過的每一張紙：主票 + 已合併的候選票（不含 exclude）"""
        contributors = []
        if tray.ticket is not None and tray.ticket is not exclude:
            contributors.append(tray.ticket)
        for et in tray.extra_tickets:
            if et is exclude or not et.is_ticket_merged:
                continue
            contributors.append(et)
        return contributors

    def find_anomaly(self, parsed_items) -> Optional[str]:
        """
        傳票解析結果明顯不合理時回傳原因（寫 log 用），正常回傳 None。

        目前只檢查單一品項數量：一張傳票的單品不會超過 max_item_qty 份，讀到
        超過幾乎可以確定是數量 OCR 讀錯。帶著錯的數量開單，這筆訂單就永遠
        核對不完，不如整張作廢重讀。
        數量為 None（未確認）不算異常，那條路徑在 OrderRequirements.expand_parsed_items 另有處理。
        """
        if self.max_item_qty <= 0:
            return None
        over = []
        for name, qty in parsed_items or []:
            try:
                qty = int(qty)
            except (TypeError, ValueError):
                continue
            if qty > self.max_item_qty:
                over.append(f"{name}×{qty}")
        if over:
            return f"品項數量超過 {self.max_item_qty}: {', '.join(over)}"
        return None

    def apply_ticket_result(self, tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip):
        if tray.ticket:
            tray.ticket.is_ocr_busy = False
            tray.ticket.last_result_mono = time.monotonic()   # 下一輪計時的起點，見 TrayStateMachine._task_trace

        parsed_items, order_number = self.order_parser.parse(frame_crop, dt_boxes, rec_res, is_flip)

        anomaly = self.find_anomaly(parsed_items)
        if anomaly:
            # 當成解析失敗：走下面 else 分支，這張票回到 WAITING_TICKET 重新累積
            # 穩定幀數；single/preset_roi 模式下 LogicEngine.apply_ocr_result 還會
            # 把 tray.ticket 清掉，下次穩定時重新鎖定、重新 OCR。此時還沒發過
            # TICKET_READY、也還沒開始錄影，所以沒有其他狀態需要退回。
            logger.warning(
                f"[ticket_guard] tray={tray_id} 傳票內容異常（{anomaly}），整張作廢、重新偵測。"
                f"單號={order_number} 解析={parsed_items}"
            )
            parsed_items = []

        if parsed_items:
            expanded_expected_items, expected_items_list, combos, report_alias = \
                self.requirements.expand_parsed_items(parsed_items)

            tray.expected_items = expanded_expected_items
            tray.expected_items_display = expected_items_list
            # 主票重新解析等於整筆訂單重來，改名表直接換掉而不是疊加
            tray.report_alias = report_alias
            tray.ticket.contributed_combos = combos
            self.requirements.recompute_expected_combos(tray)
            tray.state = TrayState.WAITING_STICKERS
            tray.ticket_crop = frame_crop
            order_number = "000" if order_number is None else order_number
            tray.order_number = order_number
            tray.ticket.order_number = order_number   # 這張紙自己的單號，見 apply_extra_ticket_result

            # # 這裡要用複製(list(...))，不能直接指到同一個 list 物件，
            # 否則之後 tray.expected_items_display.extend(...) 會連帶
            # 污染 tray.ticket.contributed_display，造成扣除時範圍算錯
            tray.ticket.contributed_display = list(expected_items_list)
            tray.ticket.is_ticket_merged = True
            tray.start_time_str = get_now_iso_str(time.time(), utc=False)
            tray.order_started_mono = time.monotonic()   # 訂單超時從這裡起算（LogicEngine._finalize_missing_trays）

            self.requirements.log_mapping(tray_id, order_number, expected_items_list)

            payload = {
                "tray_id": tray_id, "status": "TICKET_READY",
                "expected_items": expected_items_list, "order_number": order_number,
            }
            # 新增欄位：這筆訂單可能的貼紙組合。expected_items 是所有變體的
            # 聯集，所以變體數 > 1 時它會比實際會到的品項多；App 可以拿這個
            # 欄位正確渲染「這幾樣是任一組合」。不處理也不會壞，只是顯示偏保守。
            if len(getattr(tray, "expected_combos", []) or []) > 1:
                payload["item_combos"] = [dict(c) for c in tray.expected_combos]
            # 新增欄位：數量未確認的品名（下游可據此要求人工確認）。
            # 舊訂閱端忽略未知欄位即可，格式其餘部分不變。
            unconfirmed = getattr(self.requirements, "last_unconfirmed_quantity", []) or []
            if unconfirmed:
                payload["unconfirmed_quantity"] = list(unconfirmed)
                logger.warning(
                    f"[tray_state_machine] tray={tray_id} 有 {len(unconfirmed)} 個品名"
                    f"數量未確認: {unconfirmed}"
                )
            # 新增欄位：這張傳票套用了哪些替換（換湯…）。expected_items 已經
            # 是替換後的結果，所以 App 不處理也不會壞；要顯示「已換湯」的提示
            # 時拿這個欄位就有 from/to/錨定的套餐名。
            if self.requirements.last_substitutions:
                payload["substitutions"] = list(self.requirements.last_substitutions)
            # 新增欄位：文法/台帳認出是修飾列，但在它上方找不到可換的套餐。
            # 代表傳票版面跟我們的假設不一樣，值得現場看一眼（這一列已退回
            # 當普通品項處理，不會影響核對）。
            if self.requirements.last_unattached_modifiers:
                payload["unattached_modifiers"] = list(self.requirements.last_unattached_modifiers)
            self.bus.publish_det_status(payload)
            tray.record_session_id = self.on_recording_start(reason=f"ticket_confirmed:{tray_id}")
        else:
            if tray.ticket:
                tray.ticket.stable_frames = 0
            tray.state = TrayState.WAITING_TICKET

    def apply_extra_ticket_result(self, tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, task_bbox,
                                  target=None):
        target = find_task_item(tray.extra_tickets, task_bbox, target)
        if target is None:
            return
        target.is_ocr_busy = False
        target.last_result_mono = time.monotonic()

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

        # --- 1. 主號（連字號前那段）不同 = 別筆訂單 ---
        # 50、50-01、50-02 是同一筆訂單（-01、-02 是加點單）。讀不到單號或主票
        # 單號未確認（'000'）時無從判斷，一樣淘汰。
        new_key = order_number_key(order_number)
        order_key = order_number_key(tray.order_number)
        if new_key is None or order_key is None or new_key[0] != order_key[0]:
            logger.warning(f"[tray_state_machine] 第二張 ticket 單號 {order_number} 與目前訂單 {tray.order_number} 不同，忽略合併")
            tray.extra_tickets.remove(target)  # 直接淘汰，避免每幀重送 OCR
            return

        # --- 2. 是另一張紙，還是已知那張被拿起來移動了？ ---
        # 一張紙移動不會換單號，所以單號跟每一張已知的票都不同（50 之後來了
        # 50-01）就一定是另一張紙。單號跟某張已知的票完全相同時才分不出來：
        # 長單會印出兩張同號傳票。這時看這張候選在追蹤期間有沒有跟「每一張同號
        # 的票」在同一幀一起被偵測到（seen_with，由 SingleOrderTracker 記錄）——
        # 同時出現過就是兩張紙；有任何一張同號的票從沒跟它同時出現，就是那張被
        # 移到這裡來了。
        # 不再比對兩次 OCR 的內容像不像：同一張票第一次漏讀一行、移動後重讀
        # 讀全了，內容就會「不像」，以前因此被當成第二張而重複疊加品項
        # （2026-10-13 實驗機單號 87：から揚げ3個 被算成 2 份）。
        contributors = self.ticket_contributors(tray, exclude=target)
        same_number = [c for c in contributors if order_number_key(c.order_number) == new_key]
        relocated_from = [c for c in same_number if id(c) not in target.seen_with]
        if relocated_from:
            # 剛被拿起來的那張消失最短；很久以前就離開的同號票 missing_count 會大很多
            missing = [c for c in relocated_from if c.missing_count > 0]
            anchor = min(missing or relocated_from, key=lambda c: c.missing_count)
            logger.info(
                f"[tray_state_machine] tray={tray_id} 候選 ticket 單號 {order_number} "
                f"從沒跟同號的既有傳票同時出現，判定為同一張票被移動："
                f"位置接回{'主票' if anchor is tray.ticket else '已合併的候選票'}"
                f"(missing_count={anchor.missing_count})，不加品項"
            )
            anchor.bbox = target.bbox
            anchor.xywhr = target.xywhr
            anchor.missing_count = 0
            anchor.is_missing = False
            anchor.has_notified_missing = False
            tray.extra_tickets.remove(target)
            return

        if not parsed_items:
            # 確定是另一張紙但一個品項都沒讀到：當成讀取失敗淘汰、重新偵測。
            # 合併一份空內容會把它標成已處理，之後就再也不會重讀。
            logger.warning(f"[tray_state_machine] 第二張 ticket 單號 {order_number} 沒有解析出品項，淘汰後重新偵測")
            tray.extra_tickets.remove(target)
            return

        anomaly = self.find_anomaly(parsed_items)
        if anomaly:
            # 只淘汰這張候選票，進行中的訂單不動。票還在畫面上的話，下一幀
            # 會被當成新候選重新累積穩定幀數、重新 OCR。
            logger.warning(
                f"[ticket_guard] tray={tray_id} 第二張 ticket 內容異常（{anomaly}），淘汰後重新偵測。"
                f"單號={order_number} 解析={parsed_items}"
            )
            tray.extra_tickets.remove(target)
            return

        # --- 3. 另一張紙：品項疊加 ---
        reason = "單號不同的另一張（原單/加點單）" if not same_number else "同號的另一張傳票（曾同時出現）"
        logger.info(f"[tray_state_machine] tray={tray_id} 候選 ticket 單號 {order_number} 判定為{reason}，合併品項")

        expanded_expected_items, expected_items_list, combos, report_alias = \
            self.requirements.expand_parsed_items(parsed_items)
        self.requirements.log_mapping(tray_id, order_number, expected_items_list, tag="（第二張合併）")

        tray.expected_items.extend(expanded_expected_items) # ex: ["菜單名"]
        tray.expected_items_display.extend(expected_items_list) # ex: [{"菜單名": 數量}]
        target.contributed_display = expected_items_list
        target.contributed_combos = combos
        target.order_number = order_number
        tray.report_alias.update(report_alias)   # 第二張票只追加，不覆蓋主票的

        target.is_ticket_merged = True  # 保留在 extra_tickets 裡，供之後「主 ticket 消失頂替」使用
        # 兩張票的假設要做笛卡兒積：各自可能有多個變體
        self.requirements.recompute_expected_combos(tray)

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
