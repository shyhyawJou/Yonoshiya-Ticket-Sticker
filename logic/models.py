"""
models.py
=========
Tray 相關的資料結構。純資料容器，不含任何行為邏輯，
被 TrayTracker 與 TrayStateMachine 共用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

import numpy as np

from logic.geometry import ObbXYWHR, PolygonXYXY


class TrayState(Enum):
    WAITING_TICKET = auto()      # 1. 等待訂單放入並穩定
    CHECKING_TICKET = auto()     # 2. 訂單已穩定，正在跑 OCR
    WAITING_STICKERS = auto()    # 3. 訂單已解析，等待餐點(貼紙)放入並穩定
    CHECKING_STICKERS = auto()   # 4. 餐點貼紙已穩定，正在跑 OCR
    COMPLETED = auto()           # 5. 所有品項核對完成


@dataclass
class TrackedItem:
    bbox: PolygonXYXY
    xywhr: ObbXYWHR
    stable_frames: int = 0
    item_name: str = "unknown"
    is_ocr_busy: bool = False  # 標記這個物件是否正在跑 OCR
    is_checked: bool = False   # 標記這個貼紙是否已經核對成功，避免重複辨識
    is_missing: bool = False
    is_ticket_merged: bool = False   # 專給 ticket 候選用，標記「已經 OCR 過並處理完畢」，避免重複送 OCR
    is_wrong_item: bool = False
    has_notified_missing: bool = False # <--- 新增：避免重複發送 MQTT
    missing_count: int = 0     # 連續沒被偵測到的幀數，過期後 TrayTracker 會將其從 tray.stickers 移除
    cls_name: str = None
    last_wrong_notify_ts: float = 0.0    # 心跳用:上次通知前端的實際時間戳
    ticket_contributed_display: List[dict] = field(default_factory=list)  # 這張候選ticket合併時貢獻的逐行清單

    # --- 內容置換偵測用 ---
    geo_stable_frames: int = 0        # 連續「幾何完全靜止」的幀數,比 stable_frames 門檻更嚴格
    content_ref_image: Optional[np.ndarray] = None  # 凍結時的正規化參考影像(灰階、已減去自身平均亮度)
    content_check_counter: int = 0    # 距離上次做內容比對過了幾幀(每 K 幀比一次,省運算)
    content_mismatch_streak: int = 0  # 連續判定「內容不符」的次數(需連續達標才觸發重置)

@dataclass
class Tray:
    id: str
    rect: PolygonXYXY
    xywhr: ObbXYWHR
    ticket_crop: np.ndarray
    state: TrayState = TrayState.WAITING_TICKET
    order_number: int = None
    start_time_str: str = ''

    missing_count: int = 0
    drift_count: int = 0

    expected_items: List[str] = field(default_factory=list) # ['菜單名']
    expected_items_display: List[dict] = field(default_factory=list)  # 逐行顯示清單 [{"菜單名": 數量}]
    checked_items: List[str] = field(default_factory=list)

    ticket: Optional[TrackedItem] = None
    extra_tickets: List[TrackedItem] = field(default_factory=list)  # 疑似「同單號第二張」的候選 ticket
    stickers: List[TrackedItem] = field(default_factory=list)

    pending_stickers: List[dict] = field(default_factory=list)