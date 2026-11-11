"""
order_photo.py
==============
每筆訂單的照片組，讓客戶不用看影片就能看到每一單（設定見 config.py::OrderPhotoCfg）：

    1. 開始：傳票確認（TICKET_READY）當下的原始高清全幀
             full_screen/<時間>_No<單號>-start.jpg
    2. 最終：最終狀態那一刻的原始高清全幀
             full_screen/<時間>_No<單號>-final-<OK|NG|NGtoOK>.jpg
    3. 最終 UI：跟 2 同一刻，請 App 截一張 UI 畫面（ocr/v1/app/cmd/screenshot）
             App 自己存在 <日期>/screen_capture/、自己上傳 Dashboard

「最終狀態那一刻」：
    OK / NGtoOK：訂單完成（TRAY_COMPLETED）後 completed_delay_ms 拍。完成之後不會再退回，
                 收尾時餐點多半已經收走，所以不等收尾。延後是因為 App 收到狀態後要一點時間
                 重畫，馬上截會截到切換中的畫面（2026-11-09 實測：0 ms 截到「OK」字樣但底色
                 還是紅的）。高清圖也在同一刻存，兩張才是同一個畫面。
    NG：         要到收尾（護貝板露出 / 傳票離開 / 超時）才知道，收尾當下拍。收尾會馬上發重置
                 事件讓 App 清畫面，所以送出截圖指令後用 MqttBus.hold_publishes 把那些事件延後
                 hold_events_ms 才發。
    完成後又退回未完成（例如第二張傳票加了品項），之後再完成會再拍一次；最後是 NG 就在收尾拍。

高清圖跟其他 full_screen 圖一樣，凌晨由 aibox_uploader 上傳。

時間差：App 收到指令約 40 ms 就截圖（2026-11-09 實驗機實測）；UI 裡的串流畫面本身還比
相機晚一點。log 的 [訂單照片] 照片組那行 = App 截圖檔名的時間 − 高清圖的取像時間。

App 的回報（ocr/v1/app/events/screenshot）沒有帶請求編號，只能照送出順序配對；
App 3 秒內只接受一次，太密會回 throttled，log 會記下來，高清圖照存。

全部在主迴圈執行緒呼叫（LogicEngine），只有 _on_screenshot_event 在 MQTT 執行緒。
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Callable, Optional

from loguru import logger

_SHOT_TS_RE = re.compile(r"screenshot_(\d{8})_(\d{6})_(\d{3})\.")


def _clock(t: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t * 1000) % 1000:03d}"


def _shot_time(path: str) -> Optional[float]:
    """App 截圖檔名 screenshot_YYYYMMDD_HHMMSS_mmm.png 裡的時間（本機時間）"""
    m = _SHOT_TS_RE.search(path or "")
    if not m:
        return None
    try:
        base = time.mktime(time.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"))
    except ValueError:
        return None
    return base + int(m.group(3)) / 1000.0


class OrderPhoto:
    """save_hd(tag) -> 存下的檔名（None = 寫檔佇列滿了沒存）；frame_time() -> 目前這一幀的取像時間"""

    PENDING_SEC = 15.0   # 送出後多久沒收到 App 回報就不等了

    def __init__(self, cfg, bus, save_hd: Callable[[str], Optional[str]], frame_time: Callable[[], float]):
        self.cfg = cfg
        self.bus = bus
        self.save_hd = save_hd
        self.frame_time = frame_time
        self._replies: deque = deque()     # 等 App 回報的截圖請求（照送出順序）
        self._lock = threading.Lock()
        self._scheduled: dict = {}         # tray_id -> 完成後排定要拍的那一組
        if cfg.enabled:
            if cfg.ui_screenshot:
                bus.subscribe("app/events/screenshot", self._on_screenshot_event)
            logger.info(
                f"[訂單照片] 已啟用：傳票確認存一張高清；OK / NGtoOK 完成後 {cfg.completed_delay_ms} ms、"
                f"NG 收尾時各存一張高清"
                + (f"並請 App 截 UI（收尾時重置事件延後 {cfg.hold_events_ms} ms）" if cfg.ui_screenshot else "")
            )

    @staticmethod
    def _tag(order_number, *parts) -> str:
        return "-".join([f"No{order_number or 'NA'}", *parts])

    # ---------- LogicEngine 呼叫 ----------
    def on_start(self, tray_id: str, order_number) -> None:
        if not self.cfg.enabled:
            return
        name = self.save_hd(self._tag(order_number, "start"))
        logger.info(f"[訂單照片] tray={tray_id} 單號={order_number} 開始：高清 {name or '（寫檔佇列滿，沒存到）'}")

    def on_completed(self, tray_id: str, order_number, status: str) -> None:
        """訂單剛完成（OK / NGtoOK）：completed_delay_ms 後由 tick() 拍"""
        if not self.cfg.enabled:
            return
        due = time.monotonic() + self.cfg.completed_delay_ms / 1000.0
        self._scheduled[tray_id] = {"order": order_number, "status": status, "due": due}

    def cancel(self, tray_id: str, reason: str) -> None:
        """還沒拍的最終照取消。App reset / 切換模式由 LogicEngine._end_open_orders 當下呼叫"""
        s = self._scheduled.pop(tray_id, None)
        if s is not None:
            logger.info(f"[訂單照片] tray={tray_id} 單號={s['order']} {reason}，取消排定的最終照")

    def on_reverted(self, tray_id: str) -> None:
        """每幀對沒完成的 tray 呼叫：完成後又退回未完成，還沒拍的那組取消"""
        self.cancel(tray_id, "完成後又退回未完成")

    def prune(self, active_tray_ids) -> None:
        """tray 已經不存在的排程取消"""
        for tray_id in [t for t in self._scheduled if t not in active_tray_ids]:
            self.cancel(tray_id, "tray 已不存在")

    def tick(self) -> None:
        """每幀呼叫：排定的時間到了就拍"""
        if not self._scheduled:
            return
        now = time.monotonic()
        for tray_id in [t for t, s in self._scheduled.items() if s["due"] <= now]:
            s = self._scheduled.pop(tray_id)
            self._shoot(tray_id, s["order"], s["status"], hold=False)

    def on_close(self, tray_id: str, order_number, status: str, completed_shot: bool) -> None:
        """訂單收尾，要在重置事件發出之前呼叫。

        completed_shot = 完成時已經排定 / 拍過最終照（OK / NGtoOK）。排定了還沒拍的（完成後馬上收尾）
        現在拍；NG，或沒在完成時拍到的，收尾當下拍。
        """
        if not self.cfg.enabled:
            return
        s = self._scheduled.pop(tray_id, None)
        if s is not None:
            self._shoot(tray_id, s["order"], s["status"], hold=True)
        elif status == "NG" or not completed_shot:
            self._shoot(tray_id, order_number, status, hold=True)

    # ---------- 內部 ----------
    def _shoot(self, tray_id: str, order_number, status: str, hold: bool) -> None:
        t_frame = self.frame_time()
        name = self.save_hd(self._tag(order_number, "final", status))
        head = f"[訂單照片] tray={tray_id} 單號={order_number} {status} 最終"
        if not self.cfg.ui_screenshot:
            logger.info(f"{head}：高清 {name}")
            return
        now = time.time()
        with self._lock:
            while self._replies and now - self._replies[0]["sent"] > self.PENDING_SEC:
                old = self._replies.popleft()
                logger.warning(f"[訂單照片] 單號={old['order']} 的 UI 截圖 {self.PENDING_SEC:.0f}s 內沒收到 App 回報")
            self._replies.append({"order": order_number, "status": status, "hd": name,
                                  "frame": t_frame, "sent": now})
        self.bus.publish_app_cmd("screenshot", {})
        if hold:
            self.bus.hold_publishes(self.cfg.hold_events_ms / 1000.0)
        logger.info(f"{head}：高清 {name}（取像 {_clock(t_frame)}），已請 App 截 UI（{_clock(now)}）")

    def _on_screenshot_event(self, payload: dict) -> None:
        """MQTT 執行緒：App 的截圖回報。只看 saved / failed / throttled（uploaded 之類的不影響配對）"""
        status = str(payload.get("status") or "")
        if status not in ("saved", "failed", "throttled"):
            if status == "upload_rejected":
                logger.warning(f"[訂單照片] App 截圖上傳被拒：{payload}")
            return
        with self._lock:
            req = self._replies.popleft() if self._replies else None
        if req is None:
            return   # 不是我們送的（例如有人手動送了截圖指令）
        head = f"[訂單照片] 單號={req['order']} {req['status']}"
        if status != "saved":
            logger.warning(f"{head} App 沒有截圖（{status}）；高清 {req['hd']} 照常存了")
            return
        path = str(payload.get("path") or "")
        t_shot = _shot_time(path)
        gap = f"，UI 比高清晚 {1000 * (t_shot - req['frame']):.0f} ms" if t_shot else ""
        logger.info(f"{head} 照片組：高清 {req['hd']}、UI {path}{gap}")
