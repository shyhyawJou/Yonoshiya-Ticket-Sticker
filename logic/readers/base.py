"""
readers/base.py
===============
各類別解析（reader）共用的小工具。

reader 是「某一類物件的 OCR 結果回來之後，怎麼解讀、怎麼套用到訂單上」：
    readers/ticket.py   大傳票（主票 + 第二張傳票）
    readers/sticker.py  餐點貼紙
（小傳票之後照同一個形狀加一份自己的 reader，不跟大傳票共用狀態。）

什麼時候送 OCR、同時只送哪一個（排程）、每幀的生命週期檢查，仍然由
TrayStateMachine 決定；reader 只在結果回來時被呼叫。

文字本身怎麼解析依品牌版面而不同，放在品牌資料夾，由 LogicEngine 依
ticket_parser 挑一份交給 reader：
    logic/yoshinoya/order_parser.py  吉野家大傳票（ticket_parser: legacy）
    logic/szm/order_parser.py        Suzumo 表格傳票（ticket_parser: table）
reader 只認 TicketParser 這個介面，不認是哪一份。
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple

import numpy as np

from logic.geometry import PolygonXYXY, iou_poly_poly


class TicketParser(Protocol):
    """傳票解析器的介面：OCR 結果 -> ([(品名, 數量)], 單號)。

    數量讀不到時是 None（未確認），單號讀不到時是 None。大傳票、小傳票的解析器
    版面各自不同，但都輸出這個形狀，reader 才能用同一套規則套到訂單上。
    """

    def parse(
        self,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool,
    ) -> Tuple[List[Tuple[str, Optional[int]]], Optional[str]]: ...


def find_task_item(items, bbox: PolygonXYXY, target=None, skip=lambda it: False):
    """找出 OCR task 對應的那個追蹤物件。

    task 帶了 target（generate_tasks 挑中的物件本身）就只認它：還在清單裡就是它，
    不在了（例如 reset 換掉了整個 Tray）回 None，不拿別張頂替。
    舊的找法是「清單裡第一個跟 task 框 IoU > 0.1 的」，兩張貼紙疊在一起時會
    配到前面那張：2026-10 現場 log，牛丼大確認後，疊在它上面的另一張貼紙讀到
    牛丼大的字，結果卻記在已確認的牛丼大身上，連兩次就誤報錯誤品項；而真正
    被讀的那張從沒被標成 OCR 中，一回到 WAITING_STICKERS 就立刻重送。
    沒帶 target 的呼叫端（舊程式、測試）才退回 IoU 找法。
    """
    if target is not None:
        return next((it for it in items if it is target and not skip(it)), None)
    return next((it for it in items if iou_poly_poly(it.bbox, bbox) > 0.1 and not skip(it)), None)
