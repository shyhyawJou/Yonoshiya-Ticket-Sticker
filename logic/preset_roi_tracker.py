"""
preset_roi_tracker.py
======================
PresetRoiTracker:精神上是 SingleOrderTracker 的特化版本——
一樣「不追蹤 tray 盤位置、固定只維護一筆訂單、貼紙不受位置限制」,
唯一差異在於:「訂單是否要開始處理」這件事,不是看有沒有偵測到
ticket,而是看 ticket 的中心點是否落在設定檔預先定義好的
ticket_roi 區域內。

透過繼承 SingleOrderTracker、只覆寫 _filter_ticket_dets 這一個
擴充點,重用 SingleOrderTracker 裡「候選累積、貼紙穩定度、
ticket_leave_frame 判斷離開、reset 系列」的全部既有邏輯,避免
複製一份幾乎相同、卻可能各自漂移的程式碼。

「離開 ROI 就視為結束」這件事,直接沿用 SingleOrderTracker 既有的
ticket_leave_frame 機制:只要連續 N 幀「畫面上沒有任何一張 ticket
的中心點落在 ticket_roi 內」,就視為 ticket 離開,交由
LogicEngine._finalize_missing_trays 收尾(截圖 + CSV log + 移除)。

完成與否單純看收尾當下 expected_items / checked_items 是否一致,
不需要額外的「失敗」事件——這跟目前「人為 reset 才是唯一的主動重置
手段,其餘一律視為訂單自然結束」的設計一致。
"""

from __future__ import annotations

from typing import List

from logic.geometry import get_polygon_centroid, is_center_in_polygon, PolygonXYXY
from logic.single_order_tracker import SingleOrderTracker, SINGLE_ORDER_ID


class PresetRoiTracker(SingleOrderTracker):
    def __init__(
        self,
        bus,
        sticker_missing_frame: int,
        tray_missing_frame: int,
        frame_width: int,
        frame_height: int,
        ticket_leave_frame: int,
        ticket_roi_polygon: PolygonXYXY,
        tray_id: str = SINGLE_ORDER_ID,
    ):
        super().__init__(
            bus=bus,
            sticker_missing_frame=sticker_missing_frame,
            tray_missing_frame=tray_missing_frame,
            frame_width=frame_width,
            frame_height=frame_height,
            ticket_leave_frame=ticket_leave_frame,
            tray_id=tray_id,
        )
        self.ticket_roi_polygon = ticket_roi_polygon

    def _filter_ticket_dets(self, ticket_dets: List) -> List:
        """
        只保留中心點落在 ticket_roi 內的 ticket 偵測。
        同一時刻 ROI 內若有多張同號碼的 ticket(現場允許的情境),
        交由基底類別既有的「取第一張、其餘警告」邏輯處理,這裡
        只負責篩掉「不在 ROI 內」的偵測。
        """
        return [
            d for d in ticket_dets
            if is_center_in_polygon(get_polygon_centroid(d.xyxy), self.ticket_roi_polygon)
        ]