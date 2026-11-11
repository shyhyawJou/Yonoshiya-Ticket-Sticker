"""
models.py
=========
Tray 相關的資料結構。純資料容器，不含任何行為邏輯，
被 TrayTracker 與 TrayStateMachine 共用。
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Dict

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
    parsed_items: Optional[List[tuple]] = None
    # 追蹤期間曾經跟哪些傳票（id()）在同一幀一起被偵測到。同號的票只有
    # 「同時出現過」才算另一張紙，否則就是已知那張被拿起來移動了。
    seen_with: set = field(default_factory=set)
    # 錯誤品項確認（stability.wrong_item_confirm）：這張貼紙連續幾次 OCR 沒有比對成功，
    # 每筆 ("WRONG", 品名) 或 ("UNKNOWN", 讀到的字)，只留最近 K 筆。比對成功就清空。
    wrong_reads: List[tuple] = field(default_factory=list)
    # 連續 K 次都讀不出菜名：只通知前端（Unknown:讀到的字）、畫面標紅，不擋訂單完成、不寫 CSV
    is_unknown_item: bool = False
    # 計時 log 用（[OCR#n ■] 那行的「本輪」）：開始追蹤的 monotonic 時間、
    # 上次送 OCR 之後因為移動而穩定幀數歸零幾次、已經送過幾次 OCR、
    # 上一輪 OCR 結果套用回來的時間（第 2 輪起的起點）
    first_seen_mono: float = field(default_factory=time.monotonic)
    still_resets: int = 0
    ocr_reads: int = 0
    last_result_mono: Optional[float] = None
    # 沙拉日期標籤（readers/date_label.py）：這張是日期標籤時，記讀到的「サラダ:26.8.8-16:27」
    date_label: Optional[str] = None
    # 同一份サラダ的一般貼紙與日期標籤成組（readers/sticker.py::StickerReader._join_pair）。
    # pair_partner=True 的那張是後到的，不進 checked_items、不計數。
    # 互相指來指去，所以 repr / == 都不能看它（否則無限遞迴）
    pair_with: Optional["TrackedItem"] = field(default=None, repr=False, compare=False)
    pair_partner: bool = False


def note_seen_together(items: List[TrackedItem]) -> None:
    """同一幀一起被偵測到的物件互相記進 seen_with。一張紙同時只有一個框，所以「曾同時出現」
    = 確實是不同的紙；同名的從沒同時出現 = 同一張被移動（readers/sticker.py::_moved_here）。"""
    if len(items) < 2:
        return
    ids = {id(t) for t in items}
    for t in items:
        t.seen_with.update(ids - {id(t)})


@dataclass
class Stub:
    """客戶聯（small_order_ticket）的狀態封裝：一筆訂單只會有一張。"""
    item: TrackedItem
    attempts: int = 0
    result: Any = None                        # StubParse（解析成功後鎖定）
    verification: Any = None                  # StubVerification（與主票交叉比對結果）
    ocr_res: list = field(default_factory=list)
    order_json: Optional[dict] = None         # 標準客戶聯 JSON 結構
    fallback_rects: list = field(default_factory=list)  # CV 補救數量框座標 [(x0, y0, x1, y1), ...]


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
    # 活著的假設，任一份被滿足就算完成（見 OrderRequirements.is_order_satisfied）。
    # expected_items 則是這些假設的聯集，負責「這個品項算不算屬於本訂單」。
    expected_combos: List[Counter] = field(default_factory=list)
    # {其他寫法: 主變體寫法}。同一個傳票名稱有多種「純異寫」的貼紙組合時
    # （牛丼小 / 牛丼小盛），對外一律回報主變體那一種，前端才不會收到它沒在
    # TICKET_READY 列出來的品名。比對仍吃 expected_items（含所有寫法）。
    report_alias: Dict[str, str] = field(default_factory=dict)
    # {菜單名: 份數}：在 expected_items 裡（貼了不報錯），但不在顯示清單、不算完成條件的品項。
    # 換湯 show_removed: optional 時被換掉的 みそ汁。ITEM_CHECKED 只報到顯示清單的份數
    # （OrderRequirements.report_checked_items），舊 App 才不會收到清單上沒有的品名。
    optional_items: Dict[str, int] = field(default_factory=dict)
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

    # log 的「[訂單開始] / [訂單結束]」標記用（LogicEngine._mark_order_starts）。
    # None = 這個 Tray 還沒被當成一筆訂單開始過。perf_start 是開始當下主迴圈
    # 計時的累計快照，結束時相減就是這筆訂單期間的效能。
    order_seq: Optional[int] = None
    order_begin_mono: Optional[float] = None
    perf_start: Optional[dict] = None
    # 訂單照片組的「開始」那張拍過了沒（LogicEngine.apply_ocr_result，manager/order_photo.py）
    photo_started: bool = False
    # 完成（COMPLETED）時已經排定最終照了沒；退回未完成時清掉，再完成會再排一次
    photo_completed: bool = False

    # 客戶聯小票（small_order_ticket）：一筆訂單最多一張，與主 ticket 交叉驗證用，不影響訂單狀態與完成判定
    stub: Optional[Stub] = None

    @property
    def stubs(self) -> List[TrackedItem]:
        return [self.stub.item] if self.stub is not None else []

    @property
    def stub_results(self) -> dict:
        if self.stub is not None and self.stub.result is not None:
            return {id(self.stub.item): self.stub.result}
        return {}

    @property
    def stub_verifications(self) -> dict:
        if self.stub is not None and self.stub.verification is not None:
            return {id(self.stub.item): self.stub.verification}
        return {}