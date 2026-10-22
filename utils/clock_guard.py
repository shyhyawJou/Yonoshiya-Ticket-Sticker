"""
clock_guard.py
==============
開機時確認系統時間已經校正過，才讓主程式開始記錄訂單。

問題背景（2026-10-02 現場跳電）
------------------------------
AI box 的 RTC 斷電後保不住時間，每次開機都從映像檔的預設時間
（2025-05-30 02:48）起跳，要等網路通了 chrony 才會把時鐘撥回正確時間。
主程式比 chrony 先起來的話：
  - 校時前的訂單 start_time 記成 2025 年
  - 影片 / 截圖 / CSV 落進 20250530/ 這種日期資料夾，而 crontab 的 rclone
    只搬「昨天往前 10 天」、雲端上傳只傳「昨天」，那個資料夾永遠沒人處理

判準刻意不依賴 chrony：只看「現在時間是否早於 MIN_VALID_TIME」。
RTC 的預設時間固定落在這個日期之前，而任何真正上線的時間都在它之後，
所以這個常數**不需要隨版本更新**。

等不到就照常啟動（只記 ERROR）：核對站不能因為現場沒網路就不能用。
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Tuple

from loguru import logger

MIN_VALID_TIME = datetime(2026, 1, 1)


def clock_is_plausible(now: datetime | None = None) -> bool:
    return (now or datetime.now()) >= MIN_VALID_TIME


def wait_for_valid_clock(timeout_sec: float = 120.0, poll_sec: float = 2.0) -> Tuple[bool, float]:
    """時間不合理就等，最多 timeout_sec 秒。回傳 (時間是否可信, 等了幾秒)。

    等待本身用 monotonic 計時 —— 等的就是牆上時鐘被撥動，不能拿它來量。
    """
    if clock_is_plausible():
        return True, 0.0

    logger.warning(
        f"[clock] 系統時間 {datetime.now():%Y-%m-%d %H:%M:%S} 早於 {MIN_VALID_TIME:%Y-%m-%d}，"
        f"判定開機後尚未校時（RTC 斷電後保不住時間），等待 chrony 校時，最多 {timeout_sec:.0f} 秒…"
    )
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_sec:
        time.sleep(poll_sec)
        if clock_is_plausible():
            return True, time.monotonic() - t0
    return False, time.monotonic() - t0


def report(ok: bool, waited: float) -> None:
    """把結果寫進 log。拆開是因為等待發生在當日 log 檔 sink 接上之前
    （時間不對時，log 路徑裡的日期也是錯的），結果要等 sink 接上後再補記。"""
    if ok and waited > 0:
        logger.warning(f"[clock] 開機時尚未校時，等了 {waited:.0f} 秒 chrony 校時完成，現在 {datetime.now():%Y-%m-%d %H:%M:%S}")
    elif not ok:
        logger.error(
            f"[clock] 等了 {waited:.0f} 秒仍未校時（多半是網路不通），照常啟動。"
            f"在校時之前，訂單時間、影片與 CSV 的日期資料夾都會是錯的"
            f"（目前 {datetime.now():%Y-%m-%d %H:%M:%S}）"
        )
