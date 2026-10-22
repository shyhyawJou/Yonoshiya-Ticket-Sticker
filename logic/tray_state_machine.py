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

import itertools
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import time
from loguru import logger

from logic.text_utils import normalize_text
from logic.geometry import PolygonXYXY, iou_poly_poly
from logic.models import Tray, TrayState
from logic.order_parser import OrderParser
from logic.sticker_matcher import StickerMatcher
from logic.single_order_tracker import SINGLE_ORDER_ID
from logic.known_class_items import is_known_class_item, resolve_item_name
from logic.ticket_modifiers import resolve_modifiers

from utils.clip_video import get_now_iso_str, get_now_str


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


def _is_subsequence(short: str, long: str) -> bool:
    """short 的每個字依序出現在 long 裡（可以不連續）。'黒カレー' 之於 '牛黒カレー'、'豚丼特盛' 之於 '豚丼超特盛'"""
    it = iter(long)
    return len(short) < len(long) and all(ch in it for ch in short)


class TrayStateMachine:
    def __init__(
        self,
        bus,
        order_parser: OrderParser,
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
        # 傳票名稱 -> 多個貼紙組合變體。值一定是 List[List[str]]。
        self.menu_variants = menu_variants or {}
        # menu_aliases.yaml 的 distinct 台帳，轉成「無序配對」方便查詢。
        # 用途見 _expand_parsed_items：人工判定不是同一道菜的配對，
        # 不准被「同名多變體」自動推論成異寫。
        self.menu_distinct_pairs = {
            frozenset((normalize_text(a, ignore_space=True),
                       normalize_text(b, ignore_space=True)))
            for group in (menu_distinct or [])
            for a, b in itertools.combinations(group, 2)
        }
        if self.menu_distinct_pairs:
            logger.info(
                f"[tray_state_machine] 載入 distinct 台帳 {len(self.menu_distinct_pairs)} 組配對，"
                f"這些不會被自動視為異寫"
            )
        # 傳票修飾列知識（換湯/換飯…）。None 或空台帳時所有相關流程都是
        # no-op，行為與改動前完全一致 —— 沒有 menu_modifiers.yaml 的品牌
        # （例如 suzumo）不需要任何額外設定。
        self.modifier_knowledge = modifier_knowledge
        if modifier_knowledge:
            logger.info(
                f"[tray_state_machine] 傳票修飾列已啟用，台帳 "
                f"{len(modifier_knowledge.rules)} 筆"
            )
        # 最近一次解析出的替換與「錨不到的修飾列」，供 TICKET_READY 回報
        self.last_substitutions: List[dict] = []
        self.last_unattached_modifiers: List[dict] = []

        self.N = n_settle_frame
        self.wrong_item_margin = wrong_item_margin

        # 傳票任一品項數量超過這個值就整張作廢重讀（見 _find_ticket_anomaly）。0 = 關閉
        self.max_item_qty = int(max_item_qty or 0)

        # 錯誤品項確認次數（見 _settle_sticker_match）。0 = 舊行為：第一次判錯就成立
        self.wrong_item_confirm = int(wrong_item_confirm or 0)
        if self.wrong_item_confirm > 0:
            logger.info(
                f"[wrong_confirm] 錯誤品項確認啟用：同一張貼紙連續 {self.wrong_item_confirm} 次"
                f"沒比對成功才通知前端；讀不出菜名的以 Unknown 通知，不擋完成、不寫 CSV"
            )
        # 強證據一次成立：比對分數 >= strong_score 且 rec 信心 >= strong_conf 的錯誤品名不必等滿次數。
        # 歷史 log 裡看錯字（模糊、遮擋）都落在模糊比對（分數 < 100），一字不差讀成別的菜名幾乎都是真的放錯。
        # 0 = 關閉（每次都要等滿 wrong_item_confirm 次）
        self.wrong_item_strong_score = float(wrong_item_strong_score or 0)
        self.wrong_item_strong_conf = float(wrong_item_strong_conf or 0)
        if self.wrong_item_confirm > 0 and self.wrong_item_strong_score > 0:
            logger.info(
                f"[wrong_confirm] 強證據一次成立：比對分數 >= {self.wrong_item_strong_score:g}"
                f" 且 rec 信心 >= {self.wrong_item_strong_conf:g}"
            )

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
                    if ts.cls_name in self.object_class_items:
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

    def _ticket_contributors(self, tray, exclude) -> list:
        """這筆訂單已經確認過的每一張紙：主票 + 已合併的候選票（不含 exclude）"""
        contributors = []
        if tray.ticket is not None and tray.ticket is not exclude:
            contributors.append(tray.ticket)
        for et in tray.extra_tickets:
            if et is exclude or not et.is_ticket_merged:
                continue
            contributors.append(et)
        return contributors

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
            self._apply_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip)

        elif item_type == "ticket_extra":
            self._apply_extra_ticket_result(tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip, task_bbox)

        elif item_type == "sticker" and tray.state == TrayState.CHECKING_STICKERS:
            self._apply_sticker_result(tray, tray_id, dt_boxes, rec_res, is_flip, task_bbox)

    # ------------------------------------------------------------
    # 內部細節
    # ------------------------------------------------------------
    def _update_confirmed_count(self, tray: Tray, item_name: str) -> bool:
        """
        根據目前畫面中實際存在的 checked item 數量，
        更新前端歷史確認數。

        confirmed_counts 採 high-water mark：
        只能增加，不能因餐點暫時離開而下降。

        回傳：
            True  = confirmed count 有增加
            False = 沒有增加
        """
        if not item_name:
            return False

        expected_count = Counter(tray.expected_items).get(item_name, 0)
        if expected_count <= 0:
            return False

        # checked_items 代表目前實際仍存在的已確認物件
        visible_count = Counter(tray.checked_items).get(item_name, 0)

        old_confirmed = tray.confirmed_counts.get(item_name, 0)

        # high-water mark
        new_confirmed = min(
            expected_count,
            max(old_confirmed, visible_count)
        )

        if new_confirmed <= old_confirmed:
            return False

        tray.confirmed_counts[item_name] = new_confirmed

        logger.info(
            f"[tray_state_machine] 確認進度增加 "
            f"tray={tray.id}, "
            f"item={item_name}, "
            f"visible={visible_count}, "
            f"confirmed={old_confirmed}->{new_confirmed}, "
            f"expected={expected_count}"
        )

        # 真正達標才 lock
        if new_confirmed >= expected_count:
            tray.locked_items[item_name] = expected_count

            logger.info(
                f"[tray_state_machine] 品項達標並鎖定 "
                f"tray={tray.id}, "
                f"item={item_name}, "
                f"count={expected_count}"
            )

        return True

    # ------------------------------------------------------------
    # 多重假設：同一個傳票名稱可能對到多個貼紙組合
    # ------------------------------------------------------------
    MAX_COMBOS = 64   # 組合假設上限，超過就退回只取各品項的第一個變體

    def _cartesian_merge(self, groups: List[List[Counter]]) -> List[Counter]:
        """把「每個來源各自的變體選項」展開成整筆訂單的組合假設。

        groups = [[品項A的變體1, 品項A的變體2], [品項B的變體1], ...]
        回傳   = [整筆訂單需求表1, 整筆訂單需求表2, ...]

        刻意做笛卡兒積而不是「每個品項各自獨立判斷」：不同變體可能共用同一個
        子品項（例如兩列都含 ご飯），獨立判斷會讓同一份 ご飯 被重複計算成
        滿足兩邊。組合數有上限保護，正常資料遠遠用不到。
        """
        combos: List[Counter] = [Counter()]
        for options in groups:
            if not options:
                continue
            if len(combos) * len(options) > self.MAX_COMBOS:
                logger.warning(
                    f"[tray_state_machine] 組合假設數超過上限 {self.MAX_COMBOS}，"
                    f"該品項只採用第一個變體（變體數={len(options)}）"
                )
                options = options[:1]
            merged_list: List[Counter] = []
            for base in combos:
                for option in options:
                    merged = base.copy()
                    merged.update(option)   # Counter.update = 需求量相加
                    merged_list.append(merged)
            combos = merged_list

        # 去重：Counter 不可 hash，用 items 的 frozenset 當指紋
        unique: List[Counter] = []
        seen = set()
        for combo in combos:
            fingerprint = frozenset(combo.items())
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique.append(combo)
        return [c for c in unique if c]

    @staticmethod
    def _pick_primary_variant(item_name: str, variants: List[List[str]]) -> List[str]:
        """挑一個變體當「對外顯示用」的版本。

        規則（依序）：
          1. 品項數最少的 —— 顯示絕不要求「可能永遠不會來」的東西。
             チーズ 的兩個變體是 [チーズ] 與 [チーズ, 黒カレー]，取前者。
          2. 同樣品項數時，優先取「名稱剛好等於傳票品名」的那個，
             讓畫面上的寫法跟作業員手上的傳票一致（牛丼小 / 牛丼小盛 取前者）。
          3. 還是平手就取 menus_mapping.yaml 裡較前面的那一列（穩定、可重現）。
        """
        def rank(indexed):
            idx, combo = indexed
            distinct = len(set(combo))
            same_as_ticket = 0 if (distinct == 1 and combo[0] == item_name) else 1
            return (distinct, same_as_ticket, idx)

        return min(enumerate(variants), key=rank)[1]

    # ------------------------------------------------------------
    # 傳票修飾列（換湯 / 換飯…）
    # ------------------------------------------------------------
    def _slot_count(self, ticket_name: str, axis: str) -> int:
        """這個傳票名稱「每一份」提供幾個該軸的可替換槽位。

        用主變體算，因為主變體就是我們對外顯示、也認定最可能為真的那一版
        （見 _pick_primary_variant）。交給 ticket_modifiers.resolve_modifiers
        當錨定的容量上限，所以「牛丼お新香みそ汁セット x2 只換一碗湯」會
        正確地只扣一份。

        單品湯不算槽位：`あさり汁` 的主變體就是 `['あさり汁']`，它本身就是
        那碗湯，不是「附湯的套餐」。不排除的話，傳票上單獨點的一碗湯會變成
        合法的錨定對象，換湯就可能扣掉它而不是套餐裡那碗（實測傳票
        441 號同時有 `しじみ汁変更` 與單品 `あさり汁`，只是順序剛好沒踩到）。
        """
        if not self.modifier_knowledge:
            return 0
        variants = self.menu_variants.get(ticket_name) or [[ticket_name]]
        primary = self._pick_primary_variant(ticket_name, variants)
        slots = sum(1 for sub in primary if self.modifier_knowledge.is_slot(axis, sub))
        if slots and len(primary) <= 1:
            return 0
        return slots

    def _remove_axis_slot(self, counter: Counter, axis: str, amount: int) -> Counter:
        """從一份需求表扣掉 amount 份該軸的槽位，回傳實際扣掉的內容。

        一份組合通常只有一個該軸的成員（套餐不會附兩碗湯）。真的有多個時
        依 Counter 既有的鍵順序扣（= menus_mapping.yaml 的 sub_items 順序），
        讓行為可重現。
        """
        removed: Counter = Counter()
        left = int(amount)
        for sub in list(counter.keys()):
            if left <= 0:
                break
            if not self.modifier_knowledge.is_slot(axis, sub):
                continue
            take = min(left, counter[sub])
            if take <= 0:
                continue
            counter[sub] -= take
            if counter[sub] <= 0:
                del counter[sub]
            removed[sub] += take
            left -= take
        return removed

    def _apply_substitution(self, mod, variant_counters, union, primary_counts) -> Counter:
        """把被換掉的槽位從這一列的三份輸出裡扣掉。

        **三份都扣**，理由各自不同：
          combos（完成判定）  不扣就是等一碗永遠不會來的湯，訂單永遠到不了
                             TRAY_COMPLETED。這是最嚴重的那個。
          display（顯示）     不扣就是客戶回報的「一份餐 + 兩種湯」。
          union（歸屬判定）   2026-09-21 客戶確認：換湯後**原本那碗湯的貼紙
                             不會印**。既然不會印，出現就是異常（例如前一單
                             殘留的貼紙），留在聯集裡等於靜默算成本訂單的一
                             部分 —— 那正是本專案最該避免的失效方式，所以
                             也扣，讓它走 WRONG_ITEM。

        回傳實際被換掉的品項（給 log 與 MQTT 回報用）。
        """
        removed: Counter = Counter()
        for counter in variant_counters:
            removed.update(self._remove_axis_slot(counter, mod.axis, mod.applied))
        self._remove_axis_slot(union, mod.axis, mod.applied)
        self._remove_axis_slot(primary_counts, mod.axis, mod.applied)
        return removed

    def _expand_parsed_items(self, parsed_items):
        """把 OrderParser 解析出的品項展開成待核對清單。

        回傳四份東西：
          expanded_expected_items  所有變體的**聯集**（攤平成一個個品項）。
                                   只負責「這張貼紙算不算屬於本訂單」的判定，
                                   以及 confirmed 進度的上限 —— 不拿來顯示。
          expected_items_list      前端顯示用的逐行清單 [{"菜單名": 數量}]，
                                   只取「主變體」。
          combos                   組合假設清單 List[Counter]，負責「訂單完成了
                                   沒有」的判定 —— 任一份被滿足就算完成。
          report_alias             {其他寫法: 主變體寫法}，對外回報時改名用。

        為什麼聯集、顯示、假設要分成三份：
          * 4切牛ﾀﾝ 有「含唐辛子 / 含たれ / 都不含」三個版本，傳票讀不到 POS
            代碼所以無法分辨。聯集拿來判斷歸屬（放たれ不會被誤報 WRONG_ITEM），
            假設拿來判斷完成（不會因為等一份永遠不會來的唐辛子而卡住）。
          * 但**顯示不能用聯集**。2026-09-18 現場回報：チーズ×3 的訂單被顯示成
            チーズ×3 + 黒カレー×3、牛丼小 被顯示成 牛丼小盛 + 牛丼小，
            「一份變成兩份」。所以顯示只取主變體。

        純異寫群組（每個變體都只有一個品項，例如 牛丼小 / 牛丼小盛）除了顯示
        取一種，對外回報也統一改成主變體的寫法 —— 否則貼紙讀到另一種寫法時，
        前端會收到一個它沒列出來的品名，進度對不起來。這類群組的組合假設也
        因此收斂成一份。

        2026-09-10：OrderParser 的數量可能是 None（未確認）。
        None 代表「讀不到，不知道」，不是 1 —— 因此：
          * 不展開為待核對項目（無法決定要核對幾份）
          * 仍以 {name: None} 保留在 expected_items_list，讓該列不會整列消失
          * 另外回傳 unconfirmed 清單，供呼叫端發布/告警

        2026-09-21：傳票修飾列（換湯）。`変更しじみ汁` 這種列不是獨立一道菜，
        而是「把上方最近那個套餐的湯換掉」。它自己照常展開（貼紙
        `※変更しじみ` 是真的會印、要核對的），但**額外**會從錨定到的那一列
        扣掉一個湯槽位。錨定與軸的判定在 logic/ticket_modifiers.py，扣除在
        _apply_substitution。沒有 menu_modifiers.yaml 的品牌完全不受影響。
        """
        expanded_expected_items = []
        expected_items_list = []
        unconfirmed = []
        groups: List[List[Counter]] = []
        report_alias: Dict[str, str] = {}

        # 先依傳票順序解析修飾列並完成錨定。parsed_items 的 index 就是傳票的
        # 上下順序（ticket_layout.group_rows 已按 cy 排序），所以這裡不需要
        # 任何 bbox 座標。
        lines, unattached = resolve_modifiers(
            parsed_items, self.modifier_knowledge, self._slot_count
        )
        self.last_substitutions = []
        self.last_unattached_modifiers = [
            {"item": m.ticket_name, "axis": m.axis, "quantity": m.quantity}
            for m in unattached
        ]

        for line in lines:
            item_name, item_num = line.name, line.quantity
            if item_num is None:
                unconfirmed.append(item_name)
                expected_items_list.append({item_name: None})
                logger.warning(
                    f"[tray_state_machine] 品名 '{item_name}' 數量未確認，"
                    "不展開為待核對項目（不預設 1）"
                )
                continue

            # 沒登記變體的品項，視為「只有一個變體、內容就是它自己」，
            # 這樣底下的流程不需要為單品項再分岔一次。
            variants = self.menu_variants.get(item_name) or [[item_name]]
            primary = self._pick_primary_variant(item_name, variants)

            # 純異寫群組：同一道菜的不同寫法，統一成主變體的名字。
            #
            # 「每個變體都只有一個品項」只是**推論**，不是事實 —— 例如傳票
            # ご飯増量 的兩個變體是 ['鰻ご飯増量'] 與 ['ご飯増量']，那是鰻魚飯
            # 加量 vs 白飯加量，根本不同的東西，統一改名會把錯的餐點說成對的。
            # 所以 menu_aliases.yaml 的 distinct 台帳在這裡有否決權：人工判定
            # 過「不是同一道菜」的配對，一律不做自動改名。
            is_spelling_group = len(variants) > 1 and all(len(set(c)) == 1 for c in variants)
            if is_spelling_group:
                names = [c[0] for c in variants]
                vetoed = [
                    pair for pair in itertools.combinations(names, 2)
                    if frozenset(pair) in self.menu_distinct_pairs
                ]
                if vetoed:
                    is_spelling_group = False
                    logger.info(
                        f"[tray_state_machine] 品名 '{item_name}' 的變體 {names} "
                        f"因 distinct 台帳判定不同，不做自動改名: {vetoed}"
                    )
            if is_spelling_group:
                target = primary[0]
                for combo in variants:
                    if combo[0] != target:
                        report_alias[combo[0]] = target
                effective_variants = [[target]]
            else:
                effective_variants = variants

            variant_counters = [
                Counter({sub: item_num * count for sub, count in Counter(combo).items()})
                for combo in effective_variants
            ]

            if len(variants) > 1:
                logger.info(
                    f"[tray_state_machine] 品名 '{item_name}' 有 {len(variants)} 個貼紙組合變體，"
                    f"任一變體湊齊即算完成: {variants}；"
                    f"顯示用主變體={primary}"
                    + (f"，回報改名={report_alias}" if is_spelling_group else "")
                )

            # 聯集：保留所有原始寫法（含非主變體），這樣貼紙印哪一種都不會
            # 被誤報 WRONG_ITEM。同一個品項在不同變體裡取需求量較大者。
            union: Counter = Counter()
            for combo in variants:
                for sub, count in Counter(combo).items():
                    union[sub] = max(union[sub], item_num * count)

            # 顯示：只用主變體，避免「一份變成兩份」
            primary_counts: Counter = Counter()
            for sub in primary:
                primary_counts[sub] += item_num

            # 掛在這一列上的修飾（換湯…）：在攤平之前先把被換掉的槽位扣掉，
            # 三份輸出一起扣（理由見 _apply_substitution）。
            for mod in line.modifiers:
                removed = self._apply_substitution(
                    mod, variant_counters, union, primary_counts
                )
                if not removed:
                    logger.warning(
                        f"[tray_state_machine] '{mod.ticket_name}' 錨定到 "
                        f"'{item_name}' 但它的組合裡找不到 {mod.axis} 槽位可扣，"
                        "本次替換未生效"
                    )
                    continue
                for sub, count in removed.items():
                    self.last_substitutions.append({
                        "axis": mod.axis,
                        "anchor": item_name,
                        "from": sub,
                        "to": mod.ticket_name,
                        "quantity": count,
                    })
                logger.info(
                    f"[tray_state_machine] '{item_name}' 依 '{mod.ticket_name}' "
                    f"扣除 {dict(removed)}（軸={mod.axis}）"
                )

            for sub, count in union.items():
                expanded_expected_items.extend([sub] * count)
            for sub, count in primary_counts.items():
                expected_items_list.append({sub: count})

            groups.append(variant_counters)

        self.last_unconfirmed_quantity = unconfirmed
        return (expanded_expected_items, expected_items_list,
                self._cartesian_merge(groups), report_alias)

    def _recompute_expected_combos(self, tray) -> None:
        """從目前還活著的貢獻者重算整筆訂單的組合假設。

        假設記在貢獻者（tray.ticket / 各 extra_ticket）身上，所以候選票離開時
        只要重算一次就能精準扣掉它的貢獻，不需要做「反向笛卡兒積」這種不可逆
        的運算。
        """
        sources: List[List[Counter]] = []
        if tray.ticket is not None and getattr(tray.ticket, "contributed_combos", None):
            sources.append(tray.ticket.contributed_combos)
        for et in tray.extra_tickets:
            if (et.is_ticket_merged
                    and not et.has_notified_missing
                    and getattr(et, "contributed_combos", None)):
                sources.append(et.contributed_combos)
        tray.expected_combos = self._cartesian_merge(sources)

    def resolve_effective_combo(self, tray, counts) -> Counter:
        """挑出「最能解釋現場狀況」的那一個組合假設，供 CSV / 報表使用。

        訂單完成時就是被湊齊的那一版；沒完成時取缺最少的那一版（最可能為真的
        假設）。CSV 若直接記聯集，多變體訂單會把「其實不會來的品項」也列成
        missing，事後查帳的人會被誤導。
        """
        combos = getattr(tray, "expected_combos", None)
        if not combos:
            return Counter(tray.expected_items)

        def shortfall(combo: Counter) -> int:
            return sum(max(0, need - counts.get(item, 0)) for item, need in combo.items())

        return min(combos, key=shortfall)

    def _is_order_satisfied(self, tray, counts) -> bool:
        """訂單是否已滿足 —— 任一個組合假設被湊齊即成立。

        Args:
            counts: {品項: 已確認份數}。呼叫端決定要用 confirmed_counts
                （歷史最高，不因餐點暫時離開而下降）還是目前畫面的 checked。
        """
        combos = getattr(tray, "expected_combos", None)
        if not combos:
            # 沒有假設資訊（舊資料、或品項全部數量未確認）時，退回
            # 「聯集全部湊滿」的舊判斷，行為與改動前一致。
            if not tray.expected_items:
                return False
            return all(
                counts.get(item, 0) >= need
                for item, need in Counter(tray.expected_items).items()
            )
        return any(
            all(counts.get(item, 0) >= need for item, need in combo.items())
            for combo in combos
        )

    def _find_ticket_anomaly(self, parsed_items) -> Optional[str]:
        """
        傳票解析結果明顯不合理時回傳原因（寫 log 用），正常回傳 None。

        目前只檢查單一品項數量：一張傳票的單品不會超過 max_item_qty 份，讀到
        超過幾乎可以確定是數量 OCR 讀錯。帶著錯的數量開單，這筆訂單就永遠
        核對不完，不如整張作廢重讀。
        數量為 None（未確認）不算異常，那條路徑在 _expand_parsed_items 另有處理。
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

    def _apply_ticket_result(self, tray, tray_id, frame_crop, dt_boxes, rec_res, is_flip):
        if tray.ticket:
            tray.ticket.is_ocr_busy = False

        parsed_items, order_number = self.order_parser.parse(frame_crop, dt_boxes, rec_res, is_flip)

        anomaly = self._find_ticket_anomaly(parsed_items)
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
                self._expand_parsed_items(parsed_items)

            tray.expected_items = expanded_expected_items
            tray.expected_items_display = expected_items_list
            # 主票重新解析等於整筆訂單重來，改名表直接換掉而不是疊加
            tray.report_alias = report_alias
            tray.ticket.contributed_combos = combos
            self._recompute_expected_combos(tray)
            tray.state = TrayState.WAITING_STICKERS
            tray.ticket_crop = frame_crop
            order_number = "000" if order_number is None else order_number
            tray.order_number = order_number
            tray.ticket.order_number = order_number   # 這張紙自己的單號，見 _apply_extra_ticket_result

            # # 這裡要用複製(list(...))，不能直接指到同一個 list 物件，
            # 否則之後 tray.expected_items_display.extend(...) 會連帶
            # 污染 tray.ticket.contributed_display，造成扣除時範圍算錯
            tray.ticket.contributed_display = list(expected_items_list)
            tray.ticket.is_ticket_merged = True
            tray.start_time_str = get_now_iso_str(time.time(), utc=False)
            tray.order_started_mono = time.monotonic()   # 訂單超時從這裡起算（LogicEngine._finalize_missing_trays）

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
            unconfirmed = getattr(self, "last_unconfirmed_quantity", []) or []
            if unconfirmed:
                payload["unconfirmed_quantity"] = list(unconfirmed)
                logger.warning(
                    f"[tray_state_machine] tray={tray_id} 有 {len(unconfirmed)} 個品名"
                    f"數量未確認: {unconfirmed}"
                )
            # 新增欄位：這張傳票套用了哪些替換（換湯…）。expected_items 已經
            # 是替換後的結果，所以 App 不處理也不會壞；要顯示「已換湯」的提示
            # 時拿這個欄位就有 from/to/錨定的套餐名。
            if self.last_substitutions:
                payload["substitutions"] = list(self.last_substitutions)
            # 新增欄位：文法/台帳認出是修飾列，但在它上方找不到可換的套餐。
            # 代表傳票版面跟我們的假設不一樣，值得現場看一眼（這一列已退回
            # 當普通品項處理，不會影響核對）。
            if self.last_unattached_modifiers:
                payload["unattached_modifiers"] = list(self.last_unattached_modifiers)
            self.bus.publish_det_status(payload)
            tray.record_session_id = self.on_recording_start(reason=f"ticket_confirmed:{tray_id}")
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
        contributors = self._ticket_contributors(tray, exclude=target)
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

        anomaly = self._find_ticket_anomaly(parsed_items)
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
            self._expand_parsed_items(parsed_items)

        tray.expected_items.extend(expanded_expected_items) # ex: ["菜單名"]
        tray.expected_items_display.extend(expected_items_list) # ex: [{"菜單名": 數量}]
        target.contributed_display = expected_items_list
        target.contributed_combos = combos
        target.order_number = order_number
        tray.report_alias.update(report_alias)   # 第二張票只追加，不覆蓋主票的

        target.is_ticket_merged = True  # 保留在 extra_tickets 裡，供之後「主 ticket 消失頂替」使用
        # 兩張票的假設要做笛卡兒積：各自可能有多個變體
        self._recompute_expected_combos(tray)

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

        raw_text = getattr(self.sticker_matcher, "last_raw_text", None) if len(rec_res) > 0 else None
        strong = match_status == "WRONG_ITEM" and self._is_strong_wrong(tray, matched_item)
        self._settle_sticker_match(tray, tray_id, target_ts, matched_item, match_status,
                                   confirm=self.wrong_item_confirm, raw_text=raw_text, strong=strong)

    def _is_strong_wrong(self, tray: Tray, matched_item) -> bool:
        """這次讀到的錯誤品名是不是強證據，可以不等滿 wrong_item_confirm 次就成立。

        分數、信心夠高之外，還要排除三種「讀得很清楚但其實不是放錯」的情況
        （2026-10-20 重放 S12~S14，都是 K 次確認本來擋得住、強證據會直接誤報成 NGtoOK）：
          1. 不是真的一字不差：ねぎ 前綴邏輯會把漏字的 'ネギ塩丼小' 以 100 分對到 'ねぎラー油'
          2. 品名本來就在訂單裡（只是數量已滿）：已核對的餐點被推開、tracker 換了新框，
             舊框還沒判定消失前新框就讀到同名 —— 真的多放一份同名則多等一次
          3. 品名跟還沒核對的品項互為子序列：'牛黒カレー' 左邊被遮住讀成 '黒カレー'、
             '豚丼超特盛' 漏一字成 '豚丼特盛'，遮擋造成的剛好是另一個真菜名
        """
        sm = self.sticker_matcher
        if self.wrong_item_strong_score <= 0 or not matched_item:
            return False
        if getattr(sm, "last_match_score", 0) < self.wrong_item_strong_score:
            return False
        if getattr(sm, "last_match_conf", 0) < self.wrong_item_strong_conf:
            return False
        if not getattr(sm, "last_match_exact", False):
            return False
        if matched_item in tray.expected_items:
            return False
        fold = getattr(sm, "_fold", None) or (lambda s: s)
        got = fold(matched_item)
        remaining = Counter(tray.expected_items) - Counter(tray.checked_items)
        for item in remaining:
            want = fold(item)
            if _is_subsequence(got, want) or _is_subsequence(want, got):
                return False
        return True

    @staticmethod
    def _blocks_completion(ts) -> bool:
        """這張貼紙會不會擋住訂單完成。

        已成立的錯誤品項會擋（舊行為）。還在確認中、而且至少一次讀到「不在訂單內的
        品名」的也要擋：不然其他品項剛好在這時湊齊，訂單進 COMPLETED 後就不再送
        貼紙 OCR，這張真的放錯的貼紙永遠到不了確認次數，整筆被記成 OK。
        只讀不出菜名（Unknown）的不擋：那多半是字印得差、或根本不是貼紙。
        """
        return ts.is_wrong_item or any(kind == "WRONG" for kind, _ in ts.wrong_reads)

    def _confirm_wrong_read(self, ts, kind: str, text: str, confirm: int, strong: bool = False):
        """累積一次沒比對成功的讀取，回傳 (成立種類, 名稱)；還沒到確認次數回 (None, None)。

        品名不要求每次相同：rec 信心在 0.5 上下時，同一張貼紙每次對到的品名不一定一樣。
        確認次數內只要有一次讀到真的品名就算錯誤品項，名稱取出現最多的（同票取最後一次）；
        全部都讀不出菜名才是 Unknown，名稱帶最後一次讀到的字。
        strong = 這次是強證據（一字不差讀成別的菜名），直接以這次的品名成立。
        """
        ts.wrong_reads.append((kind, text))
        del ts.wrong_reads[:-confirm]
        if strong and kind == "WRONG":
            return "WRONG", text
        if len(ts.wrong_reads) < confirm:
            return None, None
        names = [t for k, t in ts.wrong_reads if k == "WRONG"]
        if names:
            counts = Counter(names)
            best = max(counts.values())
            return "WRONG", [n for n in names if counts[n] == best][-1]
        return "UNKNOWN", f"Unknown:{ts.wrong_reads[-1][1]}"

    def _settle_sticker_match(self, tray: Tray, tray_id: str, ts, matched_item, match_status,
                              confirm: int = 0, raw_text: Optional[str] = None, strong: bool = False):
        """
        比對結果的共用收尾:不管身份是 OCR 得來或分類器直讀得來,
        「核對成功/失敗後該做什麼」都是同一套業務邏輯。

        confirm = 錯誤品項要連續幾次才成立（0 = 第一次就成立，舊行為；醬料包固定走 0，
        它看的是偵測類別，已經有 wrong_item_margin 多等幾幀）。成立前不通知前端、
        不記 CSV。raw_text 是這次讀到的原始文字，讀不出菜名時拿來組 Unknown 名稱。
        strong = 這次讀到的是強證據（見 wrong_item_strong_score），不必等滿次數。
        """
        wrong_alert = None  # 這次要對前端發 WRONG_ITEM_DETECTED 的名稱（None = 不發）
        # 純異寫群組（牛丼小 / 牛丼小盛）統一用主變體的寫法對外，否則前端會
        # 收到一個它沒在 TICKET_READY 列出來的品名，進度對不起來。
        # 比對本身已經完成（用的是含所有寫法的聯集），這裡只改對外的名字。
        if matched_item:
            renamed = getattr(tray, "report_alias", {}).get(matched_item)
            if renamed and renamed != matched_item:
                logger.info(
                    f"[tray_state_machine] 貼紙比到 '{matched_item}'，"
                    f"對外統一回報為主變體寫法 '{renamed}'"
                )
                matched_item = renamed
        if ts is not None:
            if match_status == "MATCHED":
                ts.is_checked = True
                ts.item_name = matched_item
                if (ts.wrong_reads or ts.is_unknown_item) and not ts.is_wrong_item:
                    # 還沒成立就比對成功 = 先前是一時看錯字（模糊、遮擋），什麼都不記
                    logger.info(
                        f"[wrong_confirm] tray={tray.id} 貼紙先前 {len(ts.wrong_reads)} 次沒比對成功"
                        f" {ts.wrong_reads}，這次比對成功 '{matched_item}'，視為看錯字、不記錯誤"
                    )
                ts.wrong_reads.clear()
                ts.is_unknown_item = False
                if ts.is_wrong_item:
                    # 這個實體之前被判為多餘/錯誤，現在重新比對成功了 ——
                    # 一定要解除警示，否則訂單永遠無法完成。
                    #
                    # 2026-09-22 實測的卡單（傳票 519，現場放了兩份牛丼並）：
                    #   1. 兩份牛丼並 -> A 判 MATCHED、B 判 WRONG_ITEM
                    #   2. 拿走多餘的一份，tracker 把「消失」歸給 A（兩張貼紙
                    #      幾乎同位置，IoU 配對無法保證配到哪一個），
                    #      handle_missing_items() 把牛丼並從 checked_items 扣掉
                    #   3. B 還在畫面上會被重新送 OCR，此時 remaining 又空出
                    #      牛丼並 -> B 這次判 MATCHED
                    #   4. 舊版只設 is_checked，is_wrong_item 留著沒清
                    #   5. B 實體還在，tracker 每幀把 is_missing 歸零（見
                    #      single_order_tracker 的貼紙更新），所以
                    #      resolve_wrong_item_removal() 那條「實體離開才解除」
                    #      的路徑永遠碰不到它 -> has_unresolved_wrong_item
                    #      永遠是 True -> 永久卡在 WAITING_STICKERS，
                    #      而且 heartbeat_wrong_items() 會一直對前端重發
                    #      WRONG_ITEM_DETECTED
                    #
                    # 只有「這個品項確實屬於本訂單、而且還有沒填滿的需求」時
                    # StickerMatcher 才會回 MATCHED（它看的是
                    # expected_items - checked_items），所以這裡解除警示不會
                    # 放過真正的錯餐點：不屬於訂單的貼紙下一次仍然判 WRONG_ITEM。
                    #
                    # tray.ever_had_wrong_item / wrong_item_history 刻意不清 ——
                    # 這筆訂單確實發生過錯誤，那是要寫進 CSV 的稽核紀錄。
                    logger.info(
                        f"[tray_state_machine] 原本被判定為錯誤/多餘的 "
                        f"'{matched_item}' 重新比對成功，解除警示 tray={tray.id}"
                    )
                    ts.is_wrong_item = False
            else:
                ts.stable_frames = 0
                if confirm <= 0:
                    # 舊行為：第一次判錯就成立
                    if match_status == "WRONG_ITEM":
                        if not ts.is_wrong_item:
                            # 只在「第一次」被判定為 wrong item 時記一筆，
                            # 避免同一個物件之後重複送 OCR 或 heartbeat 通知造成重複計數
                            tray.ever_had_wrong_item = True
                            tray.wrong_item_history.append(matched_item)
                        ts.is_wrong_item = True
                        ts.item_name = matched_item
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = matched_item
                elif ts.is_wrong_item:
                    # 已成立：照舊重發警示，名稱固定用成立時那個（之後每次讀到的品名可能不同）
                    if match_status == "WRONG_ITEM":
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = ts.item_name
                else:
                    # 確認中：讀到不在訂單內的品名，或讀出了字但比對不到菜名，都算一次。
                    # 一個字都沒讀到不算（沒有證據），也不歸零。
                    kind, text = None, None
                    if match_status == "WRONG_ITEM":
                        kind, text = "WRONG", matched_item
                    elif raw_text:
                        kind, text = "UNKNOWN", raw_text
                    verdict, name = (self._confirm_wrong_read(ts, kind, text, confirm, strong=strong)
                                     if kind else (None, None))
                    if verdict == "WRONG":
                        # 只在成立的那一次記一筆（一張貼紙只記一次，跟舊行為一樣）
                        tray.ever_had_wrong_item = True
                        tray.wrong_item_history.append(name)
                        ts.is_wrong_item = True
                        ts.is_unknown_item = False
                        ts.item_name = name
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = name
                        reason = ("強證據（一字不差、信心夠高）" if strong
                                  else f"連續 {confirm} 次沒比對成功")
                        logger.warning(
                            f"[wrong_confirm] tray={tray.id} {reason} {ts.wrong_reads}"
                            f" -> 錯誤品項成立 '{name}'"
                        )
                    elif verdict == "UNKNOWN":
                        if not ts.is_unknown_item:
                            logger.warning(
                                f"[wrong_confirm] tray={tray.id} 連續 {confirm} 次讀不出菜名 {ts.wrong_reads}"
                                f" -> 通知 '{name}'（不擋完成、不寫 CSV）"
                            )
                        ts.is_unknown_item = True
                        ts.item_name = name
                        ts.last_wrong_notify_ts = time.time()
                        wrong_alert = name
                    elif kind:
                        logger.info(
                            f"[wrong_confirm] tray={tray.id} 疑似錯誤 {len(ts.wrong_reads)}/{confirm}"
                            f" {ts.wrong_reads}，先不通知"
                        )
        elif match_status == "WRONG_ITEM":
            if confirm <= 0:
                wrong_alert = matched_item  # 舊行為：對不到追蹤中的貼紙也照發
            else:
                logger.warning(f"[wrong_confirm] tray={tray.id} OCR 結果對不到追蹤中的貼紙，無法累積確認次數: '{matched_item}'")

        if match_status == "MATCHED":
            # --------------------------------------------------
            # 1. checked_items = 現在畫面上存在的已確認餐點
            # --------------------------------------------------
            tray.checked_items.append(matched_item)

            # --------------------------------------------------
            # 2. confirmed_counts = 歷史最高確認進度
            #    只有真的突破歷史最高值，才通知 frontend
            # --------------------------------------------------
            confirmed_increased = self._update_confirmed_count(
                tray,
                matched_item
            )

            if confirmed_increased:
                items_list = [
                    {item: count}
                    for item, count in tray.confirmed_counts.items()
                ]

                payload = {
                    "tray_id": tray_id,
                    "status": "ITEM_CHECKED",
                    "items": items_list
                }
                # 新增欄位：這次實際從貼紙上讀到的原始文字。顯示名是折疊後
                # 選出的代表菜名（跟傳票同一種寫法），但店家貼紙可能印的是
                # 另一種寫法 —— 兩個都給，前端才能顯示得跟現場一致。
                raw_text = getattr(self.sticker_matcher, "last_matched_text", None)
                if raw_text and raw_text != matched_item:
                    payload["matched_text"] = raw_text
                self.bus.publish_det_status(payload)

            # --------------------------------------------------
            # 3. 判斷整份訂單是否完成
            #    多重假設：任一個貼紙組合變體被湊齊就算完成，不需要把
            #    「所有變體的聯集」都湊滿（那會等一份永遠不會來的品項）。
            # --------------------------------------------------
            all_completed = self._is_order_satisfied(tray, tray.confirmed_counts)

            has_unresolved_wrong_item = any(
                self._blocks_completion(t)
                for t in tray.stickers
            )

            if all_completed and not has_unresolved_wrong_item:
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

        elif wrong_alert is not None:
            self.bus.publish_det_status({
                "tray_id": tray_id,
                "status": "WRONG_ITEM_DETECTED",
                "items": [{wrong_alert: 1}]
            })

        if tray.state != TrayState.COMPLETED:
            tray.state = TrayState.WAITING_STICKERS
    
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
                self._settle_sticker_match(tray, tray_id, ts, matched_item, match_status) 

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

        用 self._is_order_satisfied(tray, tray.confirmed_counts) 而不是自己
        重寫一份聯集比較,是為了跟 _settle_sticker_match() 的一般完成路徑走
        **完全同一個判斷**:多重組合假設（同一個傳票名稱對到多個貼紙組合）
        只要任一版被湊齊就算完成。若在這裡改成比對 expected_items 聯集,
        多變體訂單會回頭去等一份永遠不會來的品項。
        """
        for tray_id, tray in trays.items():
            resolved_any = False
            for ts in tray.stickers:
                if self._blocks_completion(ts) and ts.is_missing and not ts.has_notified_missing:
                    ts.has_notified_missing = True  # 讓 tracker 下一幀清掉這筆幽靈紀錄
                    resolved_any = True
                    if ts.is_wrong_item:
                        logger.info(f"[tray_state_machine] 錯誤菜單已移除 tray={tray_id}")
                    else:
                        # 確認中就離開：沒成立，什麼都不記，只是不再擋完成
                        logger.info(f"[wrong_confirm] tray={tray_id} 疑似錯誤的貼紙在成立前離開，不記錯誤 {ts.wrong_reads}")

            if not resolved_any:
                continue

            has_unresolved_wrong_item = any(self._blocks_completion(t) and not t.has_notified_missing for t in tray.stickers)
            all_completed = self._is_order_satisfied(tray, tray.confirmed_counts)

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
                # 對外回報也統一用 confirmed_counts,跟 _settle_sticker_match
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
                    and not self._blocks_completion(ts)
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
                self._recompute_expected_combos(tray)

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
                has_unresolved_wrong_item = any(self._blocks_completion(t) for t in tray.stickers)
                if (tray.expected_items
                        and tray.state != TrayState.COMPLETED
                        and self._is_order_satisfied(tray, tray.confirmed_counts)
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