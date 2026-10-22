"""
disk_guard.py
=============
定時檢查錄影分割區（/mnt/reserved）的用量，快滿時先記 WARNING / ERROR。

問題背景（2026-10 現場）
------------------------
店家的 log 消失了一段時間，到凌晨 2 點多才恢復 —— 時間點跟 crontab 的
rclone 把舊日期資料夾 move 走、空間釋放出來吻合，研判是 /mnt/reserved 滿了。
log 檔、影片、截圖、CSV 全在這個分割區，滿了就一起寫不進去，**事後完全查不到
原因**。

所以這裡的重點是「還沒滿之前」先留下紀錄：
  - 90% 起 WARNING、95% 起 ERROR，寫進當天 log 時空間通常還夠
  - app 的 log 同時輸出到 stderr；用 systemd 跑時會進 journald（在根分割區
    /var/log/journal，不在 /mnt），分割區全滿時 `journalctl -u yoshinoya_ai`
    仍看得到
  - 訊息附上最舊的日期資料夾：好幾天前的資料夾還在，代表 rclone 沒在搬

只記錄、不刪資料：什麼可以丟要另外討論，還沒上傳的資料刪了就找不回來。
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

WARN_PCT = 90.0
ERROR_PCT = 95.0
CHECK_INTERVAL_SEC = 300      # 每 5 分鐘看一次
REPEAT_SEC = 1800             # 超標期間每 30 分鐘再提醒一次，不洗版

_DATE_DIR = re.compile(r"^\d{8}$")
_OK, _WARN, _ERROR = 0, 1, 2


def usage_pct(path: str | Path) -> Tuple[float, int]:
    """回傳 (已用百分比, 剩餘位元組)，算法與 `df` 的 Use% 相同：used / (used + avail)。"""
    du = shutil.disk_usage(path)
    denom = du.used + du.free
    return (100.0 * du.used / denom if denom else 0.0), du.free


def oldest_date_dir(root: str | Path) -> Optional[str]:
    """record/stream 底下最舊的 YYYYMMDD 資料夾名稱；沒有就回 None。"""
    try:
        names = [e.name for e in os.scandir(root) if e.is_dir() and _DATE_DIR.match(e.name)]
    except OSError:
        return None
    return min(names) if names else None


class DiskGuard:
    def __init__(self, mount_path: str | Path, record_root: str | Path,
                 interval_sec: float = CHECK_INTERVAL_SEC):
        self.mount_path = Path(mount_path)
        self.record_root = Path(record_root)
        self.interval_sec = interval_sec
        self._level = _OK
        self._last_report = 0.0     # time.monotonic()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self.mount_path.is_dir():
            logger.warning(f"[disk] {self.mount_path} 不存在，不監控磁碟用量")
            return
        self.check_once()           # 開機先看一次：開機時就快滿的話，第一時間就要知道
        self._thread = threading.Thread(target=self._loop, name="disk-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_sec):
            try:
                self.check_once()
            except Exception as e:   # 監控本身不能讓 app 出事
                logger.warning(f"[disk] 檢查磁碟用量失敗（已忽略）: {e}")

    def check_once(self) -> None:
        pct, free = usage_pct(self.mount_path)
        level = _ERROR if pct >= ERROR_PCT else _WARN if pct >= WARN_PCT else _OK
        now = time.monotonic()

        if level == _OK:
            if self._level != _OK:
                logger.info(f"[disk] {self.mount_path} 用量已降到 {pct:.1f}%（剩 {free / 1024**3:.1f} GB），恢復正常")
            self._level = _OK
            return

        if level == self._level and now - self._last_report < REPEAT_SEC:
            return
        self._level, self._last_report = level, now

        msg = f"[disk] {self.mount_path} 已用 {pct:.1f}%（剩 {free / 1024**3:.1f} GB）"
        oldest = oldest_date_dir(self.record_root)
        if oldest:
            try:
                days = (date.today() - datetime.strptime(oldest, "%Y%m%d").date()).days
                msg += f"。最舊的日期資料夾 {oldest}（已 {days} 天）"
                if days > 2:
                    msg += "，rclone 可能沒在搬檔"
            except ValueError:
                pass
        msg += "；磁碟滿了之後 log / 影片 / CSV 都會寫不進去"

        if level == _ERROR:
            logger.error(msg)
        else:
            logger.warning(msg)
