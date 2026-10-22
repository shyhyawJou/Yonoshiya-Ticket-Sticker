"""
models.py
=========
Tray 相關的資料結構。純資料容器，不含任何行為邏輯，
被 TrayTracker 與 TrayStateMachine 共用。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict

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
    # 這張票貢獻的組合假設（List[Counter]）。候選票離開時要把它的貢獻扣回去，
    # 所以假設必須記在「貢獻者」身上，而不是只存一份合併後的結果。
    contributed_combos: List[Counter] = field(default_factory=list)
    # 這張紙自己的完整單號（50、50-01…），合併/確認時寫入。同一筆訂單可能
    # 有好幾張紙（長單分頁、加點單），各自記，才分得出「是不是同一張」。
    order_number: Optional[str] = None
    # 追蹤期間曾經跟哪些傳票（id()）在同一幀一起被偵測到。同號的票只有
    # 「同時出現過」才算另一張紙，否則就是已知那張被拿起來移動了。
    seen_with: set = field(default_factory=set)
    # 錯誤品項確認（stability.wrong_item_confirm）：這張貼紙連續幾次 OCR 沒有比對成功，
    # 每筆 ("WRONG", 品名) 或 ("UNKNOWN", 讀到的字)，只留最近 K 筆。比對成功就清空。
    wrong_reads: List[tuple] = field(default_factory=list)
    # 連續 K 次都讀不出菜名：只通知前端（Unknown:讀到的字）、畫面標紅，不擋訂單完成、不寫 CSV
    is_unknown_item: bool = False

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
    # 這筆訂單可能的「貼紙組合」假設，每個元素是一份 {菜單名: 需要幾份} 需求表。
    # 同一個傳票名稱在 menus_mapping.yaml 可能有多列（不同 POS 代碼對到不同
    # 組合），而傳票上讀不到 POS 代碼，所以哪一版為真是不可知的 —— 全部當成
    # 活著的假設，任一份被滿足就算完成（見 TrayStateMachine._is_order_satisfied）。
    # expected_items 則是這些假設的聯集，負責「這個品項算不算屬於本訂單」。
    expected_combos: List[Counter] = field(default_factory=list)
    # {其他寫法: 主變體寫法}。同一個傳票名稱有多種「純異寫」的貼紙組合時
    # （牛丼小 / 牛丼小盛），對外一律回報主變體那一種，前端才不會收到它沒在
    # TICKET_READY 列出來的品名。比對仍吃 expected_items（含所有寫法）。
    report_alias: Dict[str, str] = field(default_factory=dict)
    checked_items: List[str] = field(default_factory=list) # 目前 tracker 還看得到，而且已經成功確認身份的實體餐點
    locked_items: Dict[str, int] = field(default_factory=dict)  # confirmed 達到 expected 後的品項
    confirmed_counts: Dict[str, int] = field(default_factory=dict) # 已經向前端確認過的歷史最高數量，只增不減

    ever_had_wrong_item: bool = False
    wrong_item_history: List[str] = field(default_factory=list)  # 每次「首次判定」記一筆品項名稱

    ticket: Optional[TrackedItem] = None
    extra_tickets: List[TrackedItem] = field(default_factory=list)  # 疑似「同單號第二張」的候選 ticket
    stickers: List[TrackedItem] = field(default_factory=list)

    pending_stickers: List[dict] = field(default_factory=list)
    
    # ticket mat 相關參數
    ticket_mat_stable_frames: int = 0   # 連續幾幀看到完整 ticket_mat(容忍模式:一漏就歸零)
    order_session_active: bool = False  # False=idle, True=訂單處理中(mat 被遮住後才轉 True)

    # 對應事件的影片 ID
    record_session_id: Optional[str] = None

    # 主傳票確認（TICKET_READY）當下的 time.monotonic()，訂單超時用。
    # None = 這筆訂單還沒開始。用 monotonic 是因為跳電後 chrony 校時會把
    # 牆上時鐘一口氣往前撥（見 app.py 主迴圈的說明）。
    order_started_mono: Optional[float] = None