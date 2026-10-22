#!/usr/bin/env python3
"""aibox_aws —— AI Box 上傳用的可重用類別。

把 aibox_upload.py 的規格邏輯（AIBOX_UPLOAD.md v1.2）包成一個有連線生命週期、
可重連、可取消的類別，供 AI Box 主程式長駐使用。

    from aibox_aws import aibox_aws, Options

    # CSV：/data/<YYYYMMDD>/csv_logs/<YYYYMMDD>_7d2f.csv；媒體：/data/<YYYYMMDD>/event/
    box = aibox_aws(Options(prefix_path="/data", folder_names=["event"], store_code="7d2f"))
    box.start()                      # 連線與憑證驗證，並探測 S3（規格 §8 第 1 項）
    box.upload_day(day="20260909")   # 先 CSV、成功後媒體直傳 S3（v1.2）
    box.add_job("每日上傳", "upload_day", at="03:10", every="day", kwargs={"day": "yesterday"})
    box.stop()                       # 斷線

設計要點：
- **媒體上傳方式**：預設 v1.2 直傳 S3（取網址 → 逐支 PUT → 回報完成，缺少的最多重來 3 輪）。
  `Options(media_api="v1.1")` 仍可走舊的 POST /api/ingest/media，但已 **Deprecated**（正式上線後停用）。
- **參數** 都收在 `Options`；`prefix_path` 與 `folder_names` 未指定時使用本檔的預設常數。
- **連線生命週期**：`start()` / `stop()`，支援 `with` 區塊。
- **自動重連**：上傳途中連線中斷時，先依規格 §6 重試，仍失敗則進入重連循環；
  重連次數用盡會呼叫 `on_connection_lost` 回呼並拋出 `ConnectionLostError`。
- **任務取消**：每個動作都有 task id，`cancel(task_id)` 可取消排隊中或進行中的任務
  （進行中的會直接中斷 HTTP 連線）。
- **排程**：`add_job()` / `pause_job()` / `resume_job()` / `remove_job()`。每個排程有 4 碼
  job id、不可重複的名稱、時間（本機時間）、週期（指定日期／每天／每週／每月）與要呼叫的
  pipeline。排程在 `start()` 之後由背景執行緒計時，觸發時走一般的任務流程，所以 task id、
  取消與 log 都和手動呼叫一致。
- **全程 log**：所有動作都寫 `aibox_aws` logger，訊息會帶上 task id。

規格的資料規則（檔名、欄位、批量、時間格式）一律沿用 aibox_upload.py，不在此重複實作。
"""
from __future__ import annotations

import calendar
import hashlib
import logging
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime, timedelta
from datetime import time as time_of_day  # 別名：模組層的 time 是 stdlib 的 time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.parse import urlparse

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aibox_upload as ab  # noqa: E402

__all__ = [
    "aibox_aws", "AiboxAws", "Options", "State", "TaskState", "TaskInfo",
    "Job", "Recurrence",
    "AiboxError", "NotStartedError", "ConnectionLostError", "TaskCancelled", "ScheduleError",
    "DEFAULT_PREFIX_PATH", "DEFAULT_FOLDER_NAMES", "DEFAULT_CSV_FOLDER", "DEFAULT_LOG_FOLDER", "setup_logging",
]

# ── 預設字串常數（未帶 options 時使用）──────────────────────────────────────
# 程式啟動（載入本模組）時的工作目錄。存成絕對路徑：之後就算 chdir，prefix 也不會跟著跑掉
DEFAULT_PREFIX_PATH = os.getcwd()
# 媒體在 <prefix>/<YYYYMMDD>/ 底下：event/ 是影片檔（.mp4），full_screen/ 是高清影像檔（.jpg，也可能 .jpeg / .png）
DEFAULT_FOLDER_NAMES = ("event", "full_screen")
DEFAULT_CSV_FOLDER = "csv_logs"    # CSV 在 <prefix>/<YYYYMMDD>/csv_logs/ 底下
DEFAULT_LOG_FOLDER = "stream_log"  # v1.2.4：AI Box log 在 <prefix>/<YYYYMMDD>/stream_log/<YYYYMMDD>.log

LOGGER_NAME = "aibox_aws"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """給主程式用的方便函式：把 aibox_aws 的 log 導到 stdout（與選用的檔案）。"""
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout,
    )
    if log_file:
        h = logging.FileHandler(log_file, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        logging.getLogger(LOGGER_NAME).addHandler(h)


# ── 狀態與例外 ──────────────────────────────────────────────────────────────
class State(str, Enum):
    IDLE = "IDLE"                  # 尚未 start()
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    DISCONNECTED = "DISCONNECTED"  # 連線中斷，下次動作會自動重連
    STOPPED = "STOPPED"            # 已 stop()


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AiboxError(Exception):
    """本模組的基底例外。"""


class NotStartedError(AiboxError):
    """尚未 start() 或已 stop()。"""


class ConnectionLostError(AiboxError):
    """重連次數用盡，連線無法恢復。"""


class TaskCancelled(AiboxError):
    """任務被 cancel() 取消。"""


class ScheduleError(AiboxError):
    """排程設定有問題：名稱重複、週期參數不全、pipeline 不存在等。"""


class _Transient(Exception):
    """內部用：一輪請求（含規格 §6 的重試）全部失敗，判定為連線問題。"""


# ── 參數 ────────────────────────────────────────────────────────────────────
@dataclass
class Options:
    """建構 aibox_aws 的所有參數。未指定的路徑參數會退回本檔的預設常數。"""

    # 路徑
    prefix_path: str | Path = DEFAULT_PREFIX_PATH      # 預設為啟動時的工作目錄；例：/data
    folder_names: str | Sequence[str] = DEFAULT_FOLDER_NAMES  # <prefix>/<YYYYMMDD>/<folder>；例 ["event", "snapshot"]
    csv_folder: str = DEFAULT_CSV_FOLDER               # CSV 位置：<prefix>/<YYYYMMDD>/<csv_folder>/
    log_folder: str = DEFAULT_LOG_FOLDER               # v1.2.4 log 位置：<prefix>/<YYYYMMDD>/<log_folder>/<YYYYMMDD>.log
    store_code: str | None = None                      # 預設店代號，呼叫時可再覆寫

    # v1.2.4：上傳 CSV 時一併附上當日 AI Box log（選填，只供 RD 除錯）
    upload_log: bool = True                # 找得到 log 檔就自動附上；False = 只傳 CSV
    log_max_bytes: int | None = None       # log 檔上限（bytes）。None → YOSHINOYA_LOG_MAX_MB → 規格的 20 MB

    # 連線
    base_url: str | None = None      # None → 環境變數 YOSHINOYA_URL → 規格預設值
    ca_file: str | None = None       # None → 環境變數 YOSHINOYA_CA → 設定檔 env_file 的 YOSHINOYA_CA → 腳本旁的憑證
    api_key: str | None = None       # None → 環境變數 YOSHINOYA_API_KEY → 設定檔 env_file
    env_file: str | Path | None = None  # API 金鑰與憑證路徑的設定檔；None → /etc/yoshinoya_ai/cloud_upload.env
    connect_timeout: float = ab.CONNECT_TIMEOUT
    read_timeout: float = ab.READ_TIMEOUT

    # 媒體上傳方式：v1.2（直傳 S3）或 v1.1（Deprecated，經 API 伺服器轉存）。
    # None → 環境變數 YOSHINOYA_MEDIA_API → v1.2
    media_api: str | None = None

    # v1.2 直傳 S3。None → 對應的 YOSHINOYA_* 環境變數 → 規格預設值
    s3_url: str | None = None             # start() 探測 S3 用；None → YOSHINOYA_S3_URL → 正式的 bucket
    presign_batch: int | None = None      # 每次取網址／回報完成的檔案數；規格上限 1000
    media_rounds: int | None = None       # 缺少的檔案最多重來幾輪；規格建議 3
    put_timeout: float | None = None      # 單支 PUT 的讀取逾時（秒）；規格建議 ≥ 1800
    presign_refresh: float | None = None  # 網址剩不到幾秒有效就重新取；預設 300
    put_abort_after: int | None = None    # 連續幾支因連線問題 PUT 失敗就提前結束本輪；預設 5，0 = 不提前

    # 批量：只對 v1.1（Deprecated）有意義，v1.2 一支檔案一個 PUT
    batch_max_files: int = ab.BATCH_MAX_FILES
    batch_max_bytes: int = ab.BATCH_MAX_BYTES
    # 單檔上限（bytes）。None → 環境變數 YOSHINOYA_MAX_FILE_MB → 規格值（v1.2 = 5 GB、v1.1 = 200 MB）。
    # 超限的檔案會在送出前擋下；v1.1 若讓伺服器自己拒收，它是收到一半就切連線，用戶端只看到 SSL 寫入錯誤。
    max_file_bytes: int | None = None

    # 重試與重連
    request_retries: int = ab.MAX_ATTEMPTS        # 單一請求的嘗試次數（§6）
    retry_backoff: float = ab.RETRY_BACKOFF       # 10、20、30… 秒遞增
    reconnect_max_attempts: int = 3               # 重連循環次數，用盡即拋 ConnectionLostError
    reconnect_backoff: float = 10.0               # 重連間隔，同樣遞增

    # 行為
    verify_rows: bool = True         # 核對伺服器回報的 rows 與本機列數（§3.3）

    # log
    log_level: int = logging.INFO
    log_file: str | Path | None = None
    log_to_stdout: bool = True

    # 回呼（都可不給；任何回呼自己拋的例外都會被記錄後忽略）
    on_state_change: Callable[[State, State], None] | None = None
    on_reconnect: Callable[[int, int, BaseException | None], None] | None = None
    on_connection_lost: Callable[[BaseException], None] | None = None
    on_progress: Callable[[str, dict], None] | None = None
    on_warning: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        self.prefix_path = str(self.prefix_path or DEFAULT_PREFIX_PATH)
        self.folder_names = _normalize_folders(self.folder_names)
        if self.csv_folder is None:
            self.csv_folder = DEFAULT_CSV_FOLDER
        if not self.log_folder:
            self.log_folder = DEFAULT_LOG_FOLDER
        if self.log_max_bytes is None:
            self.log_max_bytes = ab.env_log_max_bytes()
        elif self.log_max_bytes <= 0:
            raise ab.ConfigError(f"log_max_bytes 必須大於 0，實際為 {self.log_max_bytes}")
        self.media_api = ab.resolve_media_api(self.media_api)
        s3 = ab.S3Settings.resolve(
            presign_batch=self.presign_batch, rounds=self.media_rounds, put_timeout=self.put_timeout,
            refresh_margin=self.presign_refresh, abort_after=self.put_abort_after, s3_url=self.s3_url,
        )
        self.s3_url, self.presign_batch, self.media_rounds = s3.s3_url, s3.presign_batch, s3.rounds
        self.put_timeout, self.presign_refresh, self.put_abort_after = s3.put_timeout, s3.refresh_margin, s3.abort_after
        if self.max_file_bytes is None:
            # 規格預設值跟著上傳方式走：S3 單次 PUT 上限 5 GB；v1.1 的 API 伺服器是 200 MB
            self.max_file_bytes = ab.env_max_file_bytes(default=ab.default_max_file_bytes(self.media_api))
        elif self.max_file_bytes <= 0:
            raise ab.ConfigError(f"max_file_bytes 必須大於 0，實際為 {self.max_file_bytes}")

    def s3_settings(self) -> ab.S3Settings:
        """把 v1.2 的參數打包給 aibox_upload.run_s3_rounds()。"""
        return ab.S3Settings(
            presign_batch=self.presign_batch, rounds=self.media_rounds, put_timeout=self.put_timeout,
            refresh_margin=self.presign_refresh, abort_after=self.put_abort_after, s3_url=self.s3_url,
        )


def _normalize_folders(names: str | Sequence[str] | None) -> tuple[str, ...]:
    """接受 "media"、"media,snapshot"、["media", "snapshot"] 三種寫法。"""
    if not names:
        return tuple(DEFAULT_FOLDER_NAMES)
    if isinstance(names, (str, Path)):
        names = [s for s in re.split(r"[,;]", str(names))]
    out = tuple(str(n).strip() for n in names if str(n).strip())
    return out or tuple(DEFAULT_FOLDER_NAMES)


# ── 任務 ────────────────────────────────────────────────────────────────────
@dataclass
class TaskInfo:
    id: str
    name: str
    state: TaskState = TaskState.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: Any = None
    error: BaseException | None = None
    reason: str = ""
    job_id: str | None = None  # 由排程觸發時記下是哪個 job；task id 本身不帶前綴
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    future: Future | None = field(default=None, repr=False)

    def snapshot(self) -> dict:
        return {
            "id": self.id, "name": self.name, "state": self.state.value, "job_id": self.job_id,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "elapsed": (
                ((self.finished_at or datetime.now()) - self.started_at).total_seconds()
                if self.started_at else None
            ),
            "error": str(self.error) if self.error else None,
            "reason": self.reason,
        }


# ── 排程 ────────────────────────────────────────────────────────────────────
class Recurrence(str, Enum):
    ONCE = "date"       # 指定日期，只跑一次
    DAILY = "day"       # 每天
    WEEKLY = "week"     # 每週某一天
    MONTHLY = "month"   # 每月某一日


_RECURRENCE_ALIASES = {
    "date": Recurrence.ONCE, "once": Recurrence.ONCE, "指定日期": Recurrence.ONCE,
    "day": Recurrence.DAILY, "daily": Recurrence.DAILY, "每天": Recurrence.DAILY,
    "week": Recurrence.WEEKLY, "weekly": Recurrence.WEEKLY, "每週": Recurrence.WEEKLY,
    "month": Recurrence.MONTHLY, "monthly": Recurrence.MONTHLY, "每月": Recurrence.MONTHLY,
}
_WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")  # 0 = 星期一，同 date.weekday()
_WEEKDAY_ZH = ("一", "二", "三", "四", "五", "六", "日")
_SCHED_IDLE_WAIT = 30.0  # 沒有近期排程時的最長睡眠；新增／暫停／移除會即時叫醒


def _now_local() -> datetime:
    """排程一律以本機時間為準，不換算時區（正式環境的 AI Box 在日本，本機時間就是日本時間）。"""
    return datetime.now()


def _time_text(when: datetime) -> str:
    return when.strftime("%Y-%m-%d %H:%M:%S")


def _next_run_text(job: "Job") -> str:
    if job.paused:
        return "（已暫停）"
    if job.next_run is None:
        if job.every is Recurrence.ONCE:
            return "（無，單次排程已執行）" if job.done else "（無，等 start() 之後立刻補跑）"
        return "（無）"
    return _time_text(job.next_run)


def _parse_recurrence(every: Any) -> Recurrence:
    if isinstance(every, Recurrence):
        return every
    key = str(every).strip().lower()
    if key not in _RECURRENCE_ALIASES:
        raise ScheduleError(f"不認識的週期：{every!r}；可用 date（指定日期）／day／week／month")
    return _RECURRENCE_ALIASES[key]


def _parse_at(at: Any) -> time_of_day:
    """接受 "03:10"、"03:10:05"、datetime.time、datetime。一律當成本機時間。"""
    if isinstance(at, datetime):
        # 帶時區的 datetime 先換成本機時間；不帶時區的就是本機時間
        return (at.astimezone() if at.tzinfo else at).time().replace(tzinfo=None)
    if isinstance(at, time_of_day):
        return at.replace(tzinfo=None)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", str(at).strip())
    if not m:
        raise ScheduleError(f"時間必須是 HH:MM 或 HH:MM:SS（本機時間），實際為 {at!r}")
    h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if h > 23 or mi > 59 or s > 59:
        raise ScheduleError(f"時間超出範圍：{at!r}")
    return time_of_day(h, mi, s)


def _parse_on_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ScheduleError(f"日期必須是 YYYY-MM-DD 或 YYYYMMDD，實際為 {value!r}")


def _parse_weekday(value: Any) -> int:
    """回傳 0（星期一）到 6（星期日）。接受 0-6、"mon"、"monday"、"一"。"""
    if isinstance(value, bool):
        raise ScheduleError(f"星期必須是 0-6 或 mon…sun，實際為 {value!r}")
    if isinstance(value, int):
        if 0 <= value <= 6:
            return value
        raise ScheduleError(f"星期必須是 0（一）到 6（日），實際為 {value}")
    key = str(value).strip().lower()
    if key in _WEEKDAY_ZH:
        return _WEEKDAY_ZH.index(key)
    if key.isdigit():
        return _parse_weekday(int(key))
    full = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
    for i, (short, long) in enumerate(zip(_WEEKDAY_NAMES, full)):
        if key in (short, long):
            return i
    raise ScheduleError(f"不認識的星期：{value!r}；可用 0-6（0 = 一）或 mon…sun")


def _parse_day_of_month(value: Any) -> int:
    try:
        dom = int(value)
    except (TypeError, ValueError):
        raise ScheduleError(f"每月的日必須是 1-31 或 -1（月底），實際為 {value!r}") from None
    if dom == -1 or 1 <= dom <= 31:
        return dom
    raise ScheduleError(f"每月的日必須是 1-31 或 -1（月底），實際為 {dom}")


def _day_in_month(year: int, month: int, day_of_month: int) -> int:
    """把 29/30/31 或 -1 收斂到該月實際存在的日：2 月的 31 日視為 28/29 日。"""
    last = calendar.monthrange(year, month)[1]
    return last if day_of_month == -1 else min(day_of_month, last)


def make_job_id(name: str, taken: Sequence[str] = ()) -> str:
    """由名稱推出 4 碼 hash 當 job id：同一個名稱每次都得到同一個 id，方便對 log。

    4 碼只有 65536 種，撞到既有 id 時加鹽重算，所以 id 仍保證唯一。
    """
    for salt in range(4096):
        raw = name if salt == 0 else f"{name}#{salt}"
        jid = hashlib.blake2s(raw.encode("utf-8"), digest_size=2).hexdigest()
        if jid not in tuple(taken):
            return jid
    raise ScheduleError("無法產生不重複的 job id，請減少排程數量")


@dataclass
class Job:
    """一個排程。時間一律是本機時間。"""

    id: str                                  # 4 碼 hash
    name: str                                # 不可重複
    pipeline: str                            # 要呼叫的 API flow
    at: time_of_day                          # 幾點幾分
    every: Recurrence                        # 週期
    on_date: date | None = None              # every=date 用
    weekday: int | None = None               # every=week 用，0 = 星期一
    day_of_month: int | None = None          # every=month 用，-1 = 月底
    kwargs: dict = field(default_factory=dict)  # 傳給 pipeline 的參數
    catch_up: bool = False                   # 週期性排程：錯過的時段要不要補跑一次（單次排程一律補跑）
    paused: bool = False
    done: bool = False                       # 單次排程：已被排程自動觸發過，不會再觸發
    next_run: datetime | None = None
    last_run: datetime | None = None
    last_task_id: str | None = None
    last_state: str = ""
    runs: int = 0
    successes: int = 0
    failures: int = 0
    skips: int = 0
    created_at: datetime = field(default_factory=datetime.now)

    def once_at(self) -> datetime | None:
        """單次排程原定的時間（本機時間）。"""
        if self.every is not Recurrence.ONCE or self.on_date is None:
            return None
        return datetime.combine(self.on_date, self.at)

    def describe(self) -> str:
        hhmm = self.at.strftime("%H:%M:%S")
        if self.every is Recurrence.ONCE:
            return f"{self.on_date:%Y-%m-%d} {hhmm}（單次）"
        if self.every is Recurrence.DAILY:
            return f"每天 {hhmm}"
        if self.every is Recurrence.WEEKLY:
            return f"每週{_WEEKDAY_ZH[self.weekday or 0]} {hhmm}"
        dom = "月底" if self.day_of_month == -1 else f"{self.day_of_month} 日"
        return f"每月 {dom} {hhmm}"

    def snapshot(self) -> dict:
        def fmt(d: datetime | None) -> str | None:
            return d.strftime("%Y-%m-%d %H:%M:%S") if d else None

        return {
            "id": self.id, "name": self.name, "pipeline": self.pipeline,
            "every": self.every.value, "at": self.at.strftime("%H:%M:%S"),
            "schedule": self.describe(), "kwargs": dict(self.kwargs),
            "paused": self.paused, "catch_up": self.catch_up, "done": self.done,
            "next_run": fmt(self.next_run), "last_run": fmt(self.last_run),
            "last_task_id": self.last_task_id, "last_state": self.last_state,
            "runs": self.runs, "successes": self.successes,
            "failures": self.failures, "skips": self.skips,
        }


class _TaskAdapter(logging.LoggerAdapter):
    """讓每一行 log 自動帶上目前執行中的 task id。"""

    def __init__(self, logger: logging.Logger, owner: "aibox_aws"):
        super().__init__(logger, {})
        self._owner = owner

    def process(self, msg, kwargs):
        t = self._owner.current_task()
        return (f"[{t.id}] {msg}" if t else msg), kwargs


# ── 主類別 ──────────────────────────────────────────────────────────────────
class aibox_aws:
    """AI Box → Dashboard 上傳器。一個 instance 對應一台 AI Box（一家店）。"""

    SUBMITTABLE = ("upload_day", "upload_csv", "upload_log", "upload_media", "upload_screenshot")

    # ── 建構與 log ─────────────────────────────────────────────────────────
    def __init__(self, options: Options | None = None, **overrides: Any):
        opt = options or Options()
        if overrides:
            known = {f.name for f in fields(Options)}
            bad = sorted(set(overrides) - known)
            if bad:
                raise AiboxError(f"不認識的參數：{', '.join(bad)}（可用參數見 Options）")
            opt = replace(opt, **overrides)
        self.opt = opt

        base_logger = logging.getLogger(LOGGER_NAME)
        base_logger.setLevel(opt.log_level)
        # hasHandlers() 會往上找到 root，避免主程式已經 basicConfig 過時重複輸出
        if opt.log_to_stdout and not base_logger.hasHandlers():
            h = logging.StreamHandler(sys.stdout)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
            base_logger.addHandler(h)
        if opt.log_file:
            paths = {getattr(h, "baseFilename", None) for h in base_logger.handlers}
            if str(Path(opt.log_file).resolve()) not in paths:
                fh = logging.FileHandler(opt.log_file, encoding="utf-8")
                fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
                base_logger.addHandler(fh)
        self.log = _TaskAdapter(base_logger, self)

        self.base_url = (opt.base_url or os.environ.get("YOSHINOYA_URL") or ab.BASE_URL_DEFAULT).rstrip("/")
        self._api_key: str | None = None   # start() 與每次排程觸發時由 load_api_key() 載入
        self._api_key_source = ""
        self._verify: Any = True

        self._state = State.IDLE
        self._session: requests.Session | None = None
        # v1.2 的 S3 PUT 用另一個 session：不帶 X-Api-Key（金鑰不能送到 AWS），用系統內建 CA（§2.2）
        self._s3_session: requests.Session | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._tasks: dict[str, TaskInfo] = {}
        # 伺服器以 413 拒收過的最小單檔大小：≥ 這個大小的檔案之後一律在送出前排除（整個實例存活期間有效）
        self._server_reject_bytes: int | None = None
        self._jobs: dict[str, Job] = {}    # job id → Job
        self._pipelines: dict[str, Callable[..., Any]] = {}  # 自訂 pipeline
        self._sched_thread: threading.Thread | None = None
        self._sched_wake = threading.Event()   # 新增／暫停／移除時叫醒排程執行緒重算
        self._sched_stop = threading.Event()
        self._lock = threading.RLock()     # 保護 state / session / tasks / jobs
        self._op_lock = threading.RLock()  # 上傳動作序列化（規格要求 CSV → 影片依序）
        self._local = threading.local()

        self.log.info(
            "初始化 aibox_aws：base=%s prefix=%s folders=%s store=%s 媒體上傳=%s 單檔上限=%d MB",
            self.base_url, self.opt.prefix_path, ",".join(self.opt.folder_names), self.opt.store_code or "(未指定)",
            self._media_api_text(), self.opt.max_file_bytes // 1024 // 1024,
        )
        if self.opt.media_api == ab.MEDIA_API_V12:
            self.log.info(
                "v1.2 直傳 S3：每次取網址 ≤ %d 檔、最多 %d 輪、單支 PUT 逾時 %.0f 秒、網址剩 %.0f 秒重取、"
                "連續 %s 支連線失敗提前結束本輪",
                self.opt.presign_batch, self.opt.media_rounds, self.opt.put_timeout, self.opt.presign_refresh,
                self.opt.put_abort_after or "∞（不提前）",
            )
        else:
            ab.warn_deprecated('Options(media_api="v1.1")', logger=self.log)
        if self.opt.media_api == ab.MEDIA_API_V11 and self.opt.max_file_bytes > self.opt.batch_max_bytes:
            # 檔案切不開：比 batch_max_bytes 大的單檔會自己成一批，那次請求就會超過每批上限
            self.log.warning(
                "max_file_bytes（%s）大於 batch_max_bytes（%s）：超過 %s 的單一檔案會自己成一批送出，"
                "該次請求會大於 batch_max_bytes。若要每個請求都不超過 %s，請把 max_file_bytes 調到 ≤ batch_max_bytes",
                ab.human(self.opt.max_file_bytes), ab.human(self.opt.batch_max_bytes),
                ab.human(self.opt.batch_max_bytes), ab.human(self.opt.batch_max_bytes),
            )

    def __repr__(self) -> str:
        return (f"<aibox_aws state={self._state.value} base={self.base_url} prefix={self.opt.prefix_path} "
                f"media_api={self.opt.media_api}>")

    def _media_api_text(self) -> str:
        return "v1.2（直傳 S3）" if self.opt.media_api == ab.MEDIA_API_V12 else "v1.1（Deprecated）"

    # ── 路徑解析（需求 2）──────────────────────────────────────────────────
    @property
    def prefix(self) -> Path:
        return Path(self.opt.prefix_path)

    def media_dirs(self, folders: str | Sequence[str] | None = None, day: Any = None) -> list[Path]:
        """<prefix>/<YYYYMMDD>/<folder>，例：/data/20260921/event。可多組。

        folder 給絕對路徑時原樣使用，不接 prefix 也不接日期。
        """
        names = _normalize_folders(folders) if folders else self.opt.folder_names

        def rooted(n: str) -> bool:  # Windows 上 "/mnt/usb" 不算 is_absolute()，但它有根目錄，同樣不該接 prefix
            return Path(n).is_absolute() or bool(Path(n).root)

        if all(rooted(n) for n in names):
            return [Path(n) for n in names]  # 全是絕對路徑就不必換算營業日（也不會多一行「使用今天」的 log）
        d = self._day(day)
        return [Path(n) if rooted(n) else self.prefix / d / n for n in names]

    def csv_path(self, store: str | None = None, day: str | None = None) -> Path:
        """<prefix>/<YYYYMMDD>/<csv_folder>/<YYYYMMDD>_<store>.csv，例：/data/20260921/csv_logs/20260921_7d2f.csv"""
        d = self._day(day)
        return self.prefix / d / self.opt.csv_folder / f"{d}_{self._store(store)}.csv"

    def log_path(self, day: str | None = None) -> Path:
        """v1.2.4：<prefix>/<YYYYMMDD>/<log_folder>/<YYYYMMDD>.log，例：/data/20260921/stream_log/20260921.log"""
        d = self._day(day)
        return self.prefix / d / self.opt.log_folder / f"{d}.log"

    def _store(self, store: str | None) -> str:
        s = store or self.opt.store_code
        if not s:
            raise ab.ConfigError("未指定店代號：請帶 store 參數或在 Options.store_code 設定")
        if not ab.STORE_CODE_RE.match(s):
            raise ab.ConfigError(f"店代號格式不符：{s}")
        return s

    def _day(self, day: Any = None) -> str:
        """接受 None（今天）、'today'、'yesterday'、datetime/date、或 YYYYMMDD 字串。今天／昨天以本機時間為準。"""
        if day is None or (isinstance(day, str) and day.lower() in ("today", "")):
            d = datetime.now().strftime("%Y%m%d")
            self.log.info("未指定營業日，使用今天：%s", d)
            return d
        if isinstance(day, str) and day.lower() == "yesterday":
            d = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
            self.log.info("營業日指定為 yesterday：%s", d)
            return d
        if isinstance(day, datetime):
            return day.strftime("%Y%m%d")
        if hasattr(day, "strftime"):
            return day.strftime("%Y%m%d")
        d = str(day)
        if not ab.BIZ_DATE_RE.match(d):
            raise ab.ConfigError(f"營業日必須是 YYYYMMDD：{d}")
        return d

    def _resolve(self, p: str | Path) -> Path:
        """相對路徑一律接在 prefix_path 底下。"""
        path = Path(p)
        return path if path.is_absolute() else self.prefix / path

    # ── 狀態與回呼 ─────────────────────────────────────────────────────────
    @property
    def state(self) -> State:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state is State.CONNECTED

    def _set_state(self, new: State) -> None:
        with self._lock:
            old, self._state = self._state, new
        if old is not new:
            self.log.info("狀態變更：%s → %s", old.value, new.value)
            self._fire("on_state_change", old, new)

    def _fire(self, name: str, *args: Any) -> None:
        cb = getattr(self.opt, name, None)
        if cb is None:
            return
        try:
            cb(*args)
        except Exception as e:  # 回呼是主程式的碼，不能讓它中斷上傳
            self.log.warning("回呼 %s 拋出例外（已忽略）：%s", name, e)

    def _progress(self, event: str, info: dict) -> None:
        self.log.debug("progress：%s %s", event, info)
        self._fire("on_progress", event, info)

    # ── 連線生命週期（需求 3、4）───────────────────────────────────────────
    def start(self) -> dict:
        """連線並驗證憑證（等同規格 §8 第 1 項）。失敗會重試，用盡則拋 ConnectionLostError。

        v1.2 另外探測 S3（§8 第 1 項後半）。S3 連不到只記 error、不讓 start() 失敗：
        CSV 與設備截圖不經過 S3，照樣可以上傳；影片則會在回報完成時列為缺少。
        """
        with self._lock:
            if self._state is State.CONNECTED:
                self.log.info("start()：已是連線狀態，略過")
                return {"ok": True}
            self.load_api_key(reason="start()")  # 找不到會拋 ConfigError
            self._verify = self._resolve_verify()
            self._set_state(State.CONNECTING)
            self._session = requests.Session()
            self._s3_session = requests.Session()
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aibox_aws")

        attempts = max(1, self.opt.reconnect_max_attempts)
        last: BaseException | None = None
        for attempt in range(1, attempts + 1):
            try:
                body = self._healthz()
                if body.get("ok") is not True:
                    raise AiboxError(f"healthz 回應異常：{body}")
                self._set_state(State.CONNECTED)
                self.log.info("start()：連線與憑證驗證通過（%s）", self.base_url)
                if self.opt.media_api == ab.MEDIA_API_V12:
                    self.check_s3()
                self._progress("started", {"base_url": self.base_url, "media_api": self.opt.media_api})
                self._start_scheduler()
                return body
            except Exception as e:
                last = e
                self.log.warning("start()：第 %d/%d 次連線失敗：%s", attempt, attempts, e)
                if attempt < attempts:
                    time.sleep(self.opt.reconnect_backoff * attempt)

        self._set_state(State.DISCONNECTED)
        err = ConnectionLostError(f"start() 連線失敗，已嘗試 {attempts} 次：{last}")
        self.log.error("%s", err)
        self._fire("on_connection_lost", err)
        raise err

    def stop(self, cancel_running: bool = True) -> None:
        """斷線。取消未完成的任務、關閉連線與工作執行緒。之後可再 start()。"""
        self.log.info("stop()：準備斷線（cancel_running=%s）", cancel_running)
        self._stop_scheduler()  # 排程保留在清單裡，下次 start() 會重新排時間
        if cancel_running:
            self.cancel_all(reason="stop()")
        with self._lock:
            executor, self._executor = self._executor, None
            session, self._session = self._session, None
            s3_session, self._s3_session = self._s3_session, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for s in (session, s3_session):
            if s is not None:
                s.close()
        self._set_state(State.STOPPED)
        self._progress("stopped", {})
        self.log.info("stop()：已斷線")

    def reconnect(self) -> bool:
        """手動觸發一次重連。回傳是否成功。"""
        return self._reconnect(1, None, None)

    def load_api_key(self, reason: str = "") -> str:
        """（重新）載入 API 金鑰，回傳來源說明；找不到就拋 ConfigError。

        順序：Options.api_key → 環境變數 YOSHINOYA_API_KEY → 設定檔 Options.env_file
        （預設 /etc/yoshinoya_ai/cloud_upload.env）。start() 會先呼叫一次；排程每次觸發時，
        也會先呼叫它再跑排定的工作，所以設定檔裡換了金鑰不必重啟程式。log 只記來源，不記金鑰本身。
        """
        tag = f"（{reason}）" if reason else ""
        try:
            key, source = ab.load_api_key(self.opt.env_file, self.opt.api_key)
        except ab.ConfigError as e:
            self.log.error("load_api_key()%s：%s", tag, e)
            raise
        if source == "參數":
            source = "Options.api_key"
        changed = self._api_key is not None and key != self._api_key
        self._api_key, self._api_key_source = key, source
        self.log.info("load_api_key()%s：已載入 API 金鑰（來源：%s%s）", tag, source,
                      "；與上次載入的不同，之後的請求改用新金鑰" if changed else "")
        self._progress("api_key_loaded", {"source": source, "changed": changed, "reason": reason})
        return source

    def check_s3(self) -> dict:
        """探測 S3 連不連得到（v1.2 §8 第 1 項後半）：有任何 HTTP 回應（含 403）就算連得到。

        用系統內建 CA 驗證，不帶 API 金鑰。不會拋錯，結果以 dict 回傳並觸發 on_progress("s3_checked")。
        """
        self.log.info("check_s3()：探測 %s", self.opt.s3_url)
        with self._lock:
            session = self._s3_session
        res = ab.probe_s3(self.opt.s3_url, timeout=self.opt.connect_timeout, session=session)
        if res["ok"]:
            self.log.info("check_s3()：S3 連得到（HTTP %s，403 也算正常）", res["status"])
        else:
            self.log.error(
                "check_s3()：S3 連不到 —— %s。影片會上傳失敗（CSV 與設備截圖不受影響）；"
                "v1.2 起門市網路需能連出 TCP 443 到 %s（§2.1）",
                res["error"], urlparse(res["url"]).hostname or res["url"],
            )
        self._progress("s3_checked", res)
        return res

    def _resolve_verify(self) -> Any:
        """https 一律要求憑證檔（規格 §2.2 禁止關閉驗證）；http 僅供本機測試。"""
        if urlparse(self.base_url).scheme == "https":
            # Options.ca_file → 環境變數 YOSHINOYA_CA → 設定檔 env_file 的 YOSHINOYA_CA → 程式旁邊的憑證
            ca, source = ab.ca_source(self.opt.ca_file, self.opt.env_file)
            self.log.info("憑證：%s（來源：%s）", ca, source)
            return ca
        self.log.warning("Base URL 不是 https（%s），跳過憑證驗證 —— 僅限本機測試使用", self.base_url)
        return True

    def _healthz(self) -> dict:
        session = self._session or requests.Session()
        self.log.debug("GET /api/healthz")
        r = session.get(
            self.base_url + "/api/healthz", verify=self._verify,
            timeout=(self.opt.connect_timeout, min(30.0, self.opt.read_timeout)),
        )
        r.raise_for_status()
        return r.json()

    def _reconnect(self, cycle: int, error: BaseException | None, task: TaskInfo | None) -> bool:
        attempts = max(1, self.opt.reconnect_max_attempts)
        self._set_state(State.RECONNECTING)
        self.log.warning("重連：第 %d/%d 次（前次錯誤：%s）", cycle, attempts, error)
        self._fire("on_reconnect", cycle, attempts, error)
        self._sleep(self.opt.reconnect_backoff * cycle, task)
        with self._lock:
            for old in (self._session, self._s3_session):
                if old is not None:
                    old.close()
            self._session = requests.Session()
            self._s3_session = requests.Session()  # 斷線後池裡的 S3 連線多半也死了，一併換掉
        try:
            body = self._interruptible(task, self._healthz, "GET /api/healthz")
            if body.get("ok") is not True:
                raise AiboxError(f"healthz 回應異常：{body}")
        except TaskCancelled:
            raise
        except Exception as e:
            self.log.warning("重連第 %d 次失敗：%s", cycle, e)
            self._set_state(State.DISCONNECTED)
            return False
        self._set_state(State.CONNECTED)
        self.log.info("重連成功（第 %d 次）", cycle)
        self._progress("reconnected", {"cycle": cycle})
        return True

    # ── 任務管理（需求 3 的 cancel）─────────────────────────────────────────
    def current_task(self) -> TaskInfo | None:
        return getattr(self._local, "task", None)

    def tasks(self, include_finished: bool = True) -> list[dict]:
        with self._lock:
            return [
                t.snapshot() for t in self._tasks.values()
                if include_finished or t.state in (TaskState.PENDING, TaskState.RUNNING)
            ]

    def task(self, task_id: str) -> dict | None:
        with self._lock:
            t = self._tasks.get(task_id)
        return t.snapshot() if t else None

    def cancel(self, task_id: str, reason: str = "") -> bool:
        """取消一個任務。排隊中的直接移除；進行中的會立刻放棄正在等的 HTTP 請求（含 S3 PUT），
        PUT 的檔案讀取也會停止，不會在背景繼續上傳。"""
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                self.log.warning("cancel()：找不到任務 %s", task_id)
                return False
            if t.state in (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED):
                self.log.info("cancel()：任務 %s 已結束（%s），無需取消", task_id, t.state.value)
                return False
            t.cancel.set()
            t.reason = reason
            self.log.info("cancel()：要求取消任務 %s（目前 %s）%s", task_id, t.state.value, reason and f"—— {reason}")
            if t.future is not None:
                t.future.cancel()  # 還在排隊就直接拿掉
            if t.state is TaskState.RUNNING:
                # 中斷進行中的請求：關掉連線池，送出中的 request 會立刻拋錯。API 與 S3 兩個都要關
                interrupted = False
                for attr in ("_session", "_s3_session"):
                    old = getattr(self, attr)
                    if old is not None:
                        setattr(self, attr, requests.Session())
                        threading.Thread(target=old.close, daemon=True).start()
                        interrupted = True
                if interrupted:
                    self.log.info("cancel()：已中斷任務 %s 進行中的連線", task_id)
        return True

    def cancel_all(self, reason: str = "") -> int:
        with self._lock:
            ids = [t.id for t in self._tasks.values() if t.state in (TaskState.PENDING, TaskState.RUNNING)]
        if ids:
            self.log.info("cancel_all()：取消 %d 個未完成任務", len(ids))
        return sum(1 for i in ids if self.cancel(i, reason))

    def submit(self, method: str, *args: Any, task_id: str | None = None, **kwargs: Any) -> str:
        """把動作丟到背景執行緒排隊，立刻回傳 task id（可用於 cancel / wait）。"""
        if method not in self.SUBMITTABLE:
            raise AiboxError(f"submit() 不支援 {method}；可用：{', '.join(self.SUBMITTABLE)}")
        fn = getattr(self, f"_do_{method}")
        t = self._submit_task(method, lambda task: fn(task, *args, **kwargs), task_id=task_id)
        self.log.info("submit()：任務 %s（%s）已排入佇列", t.id, method)
        return t.id

    def _submit_task(self, name: str, fn: Callable[[TaskInfo], Any], task_id: str | None = None,
                     job_id: str | None = None) -> TaskInfo:
        """共用的排入佇列：submit() 與排程都走這裡，任務生命週期與 log 才會一致。"""
        self._require_started()
        t = self._register(name, task_id, job_id)
        with self._lock:
            if self._executor is None:
                raise NotStartedError("尚未 start()，無法排入佇列")
            t.future = self._executor.submit(lambda: self._execute(t, fn))
        return t

    def wait(self, task_id: str, timeout: float | None = None) -> Any:
        """等一個 submit() 的任務結束；失敗會把原始例外拋出來。"""
        with self._lock:
            t = self._tasks.get(task_id)
        if t is None:
            raise AiboxError(f"找不到任務 {task_id}")
        if t.future is None:
            return t.result
        try:
            return t.future.result(timeout)
        except CancelledError:
            raise TaskCancelled(f"任務 {task_id} 在開始前已被取消") from None

    def _register(self, name: str, task_id: str | None = None, job_id: str | None = None) -> TaskInfo:
        with self._lock:
            tid = task_id or self._new_task_id()
            old = self._tasks.get(tid)
            if old is not None and old.state in (TaskState.PENDING, TaskState.RUNNING):
                raise AiboxError(f"任務 id 重複且尚未結束：{tid}")
            self._tasks[tid] = TaskInfo(id=tid, name=name, job_id=job_id)
            return self._tasks[tid]

    def _new_task_id(self) -> str:
        """8 碼十六進位的 task id，不帶任何前綴（動作名稱另記在 TaskInfo.name）。"""
        # 也不能和已結束的任務撞號：撞到會蓋掉那筆的狀態與結果。呼叫端已持有 self._lock
        while True:
            tid = uuid.uuid4().hex[:8]
            if tid not in self._tasks:
                return tid

    def _execute(self, t: TaskInfo, fn: Callable[[TaskInfo], Any]) -> Any:
        with self._op_lock, self._running(t):
            t.result = fn(t)
            return t.result

    @contextmanager
    def _running(self, t: TaskInfo):
        prev = getattr(self._local, "task", None)
        self._local.task = t
        t.started_at = datetime.now()
        t0 = time.monotonic()
        try:
            if t.cancel.is_set():
                raise TaskCancelled(f"任務 {t.id} 在開始前已被取消")
            t.state = TaskState.RUNNING
            self.log.info("任務開始：%s", t.name)
            yield t
        except TaskCancelled as e:
            t.state, t.error = TaskState.CANCELLED, e
            self.log.warning("任務取消：%s（%.1f 秒）—— %s", t.name, time.monotonic() - t0, e)
            raise
        except BaseException as e:
            t.state, t.error = TaskState.FAILED, e
            self.log.error("任務失敗：%s（%.1f 秒）—— %s", t.name, time.monotonic() - t0, e)
            raise
        else:
            t.state = TaskState.DONE
            self.log.info("任務完成：%s（%.1f 秒）", t.name, time.monotonic() - t0)
        finally:
            t.finished_at = datetime.now()
            self._local.task = prev

    def _check_cancel(self, task: TaskInfo | None) -> None:
        if task is not None and task.cancel.is_set():
            raise TaskCancelled(f"任務 {task.id} 已被取消{('：' + task.reason) if task.reason else ''}")

    def _sleep(self, seconds: float, task: TaskInfo | None) -> None:
        """可被 cancel 中斷的等待。"""
        if task is None:
            time.sleep(seconds)
            return
        if task.cancel.wait(seconds):
            raise TaskCancelled(f"任務 {task.id} 在等待重試時被取消")

    def _interruptible(self, task: TaskInfo | None, fn: Callable[[], Any], what: str) -> Any:
        """在輔助執行緒跑一個 HTTP 呼叫，任務每 0.05 秒檢查一次 cancel，被取消就立刻拋 TaskCancelled。

        requests 沒辦法從外面中斷一個已送出、正在等回應的請求（關掉 session 只會關閒置的連線），
        所以改成任務本身不等它：被放棄的請求留在背景自行結束，結果丟掉。S3 PUT 另外會在讀檔時
        看到 abort 而停止送出（aibox_upload.s3_put 的 abort），不會在背景繼續佔頻寬。
        注意：伺服器若已收到完整請求，它仍可能處理完成（CSV、媒體重送是覆蓋；截圖可能留下一張）。
        """
        if task is None:
            return fn()
        box: dict[str, Any] = {}
        done = threading.Event()

        def run() -> None:
            try:
                box["value"] = fn()
            except BaseException as e:  # 交回任務執行緒處理
                box["error"] = e
            finally:
                done.set()

        threading.Thread(target=run, name=f"aibox_aws-http-{task.id}", daemon=True).start()
        while not done.wait(0.05):
            if task.cancel.is_set():
                self.log.info("%s 進行中被取消，不再等它回應", what)
                break
        self._check_cancel(task)  # 取消優先：取消造成的連線錯誤或 UploadAborted 都算取消
        if "error" in box:
            raise box["error"]
        return box["value"]

    def _require_started(self) -> None:
        if self._state in (State.IDLE, State.STOPPED):
            raise NotStartedError(f"目前狀態 {self._state.value}，請先呼叫 start()")

    # ── 傳輸層（重試 §6 ＋ 重連 ＋ 取消）───────────────────────────────────
    def _request(self, path: str, data: dict | None = None, parts: Sequence[ab.Part] = (),
                 task: TaskInfo | None = None, json_body: dict | None = None) -> dict:
        """送到 API 伺服器的請求（multipart 或 json_body）。S3 的 PUT 不走這裡，見 _s3_put()。"""
        self._require_started()
        task = task if task is not None else self.current_task()
        attempts = max(1, self.opt.reconnect_max_attempts)
        last: BaseException | None = None

        for cycle in range(1, attempts + 1):
            self._check_cancel(task)
            if cycle > 1 and not self._reconnect(cycle - 1, last, task):
                continue  # 這一輪重連沒成功，換下一輪（間隔會遞增）
            try:
                return self._attempt(path, data, parts, task, json_body)
            except _Transient as e:
                last = e
                self._set_state(State.DISCONNECTED)
                self.log.warning("%s：連線中斷（%s）", path, e)

        err = ConnectionLostError(f"{path} 連線失敗，重連 {attempts} 次仍無法恢復：{last}")
        self.log.error("%s", err)
        self._fire("on_connection_lost", err)
        raise err

    def _attempt(self, path: str, data: dict | None, parts: Sequence[ab.Part], task: TaskInfo | None,
                 json_body: dict | None = None) -> dict:
        """一輪請求：依規格 §6 重試網路錯誤與 5xx；4xx 立即拋 ApiError。"""
        url = self.base_url + path
        retries = max(1, self.opt.request_retries)
        last = ""
        for attempt in range(1, retries + 1):
            self._check_cancel(task)
            try:
                with self._lock:
                    session = self._session
                if session is None:
                    raise NotStartedError("連線已關閉")
                self.log.debug("POST %s（第 %d/%d 次，%d 個檔案）", path, attempt, retries, len(parts))

                def send(session: requests.Session = session) -> requests.Response:
                    with ExitStack() as stack:  # 檔案在送出的那條執行緒裡開、關
                        files = [
                            (p.field, (p.path.name, stack.enter_context(p.path.open("rb")), ab.content_type(p.path)))
                            for p in parts
                        ]
                        return session.post(
                            url, headers={"X-Api-Key": self._api_key}, data=data, files=files or None,
                            json=json_body, verify=self._verify,
                            timeout=(self.opt.connect_timeout, self.opt.read_timeout),
                        )

                r = self._interruptible(task, send, f"POST {path}")
                if r.status_code < 500:
                    return ab.parse_response(r)  # 與 aibox_upload.py 共用同一套解析
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                # cancel() 會關掉連線池，這裡的錯誤其實是取消造成的
                self._check_cancel(task)
                last = f"{type(e).__name__}: {e}"

            if attempt == retries:
                raise _Transient(f"{path} 重試 {retries} 次後仍失敗：{last}")
            delay = self.opt.retry_backoff * attempt
            self.log.warning("%s 失敗（%s），%.0f 秒後重試（第 %d/%d 次）", path, last, delay, attempt + 1, retries)
            self._sleep(delay, task)
        raise AssertionError("unreachable")

    # ── 對外動作 ───────────────────────────────────────────────────────────
    def health(self) -> dict:
        """不需金鑰的健康檢查。不會改變連線狀態。"""
        self.log.info("health()：檢查 %s", self.base_url)
        if self._state is State.IDLE:
            self._verify = self._resolve_verify()
        body = self._healthz()
        self.log.info("health()：%s", body)
        return body

    def upload_csv(self, csv_path: str | Path | None = None, *, store: str | None = None,
                   day: Any = None, log_path: str | Path | bool | None = None, task_id: str | None = None) -> dict:
        """上傳 CSV。v1.2.4 起會一併附上當日 log：log_path=None 自動找 <prefix>/<日>/stream_log/<日>.log，
        給路徑就用那個檔，給 False 就只傳 CSV。"""
        t = self._register("upload_csv", task_id)
        return self._execute(
            t, lambda task: self._do_upload_csv(task, csv_path, store=store, day=day, log_path=log_path)
        )

    def upload_log(self, log_path: str | Path | None = None, *, store: str | None = None, day: Any = None,
                   task_id: str | None = None) -> dict:
        """v1.2.8：只上傳當日 AI Box log（POST /api/ingest/log），不需要 CSV。

        log_path=None 自動找 <prefix>/<日>/stream_log/<日>.log；找不到、檔名或日期不符、太大都在送出前拋 LocalCheckError。
        伺服器回報的 bytes / sha256 與本機不符時拋 VerifyError。
        """
        t = self._register("upload_log", task_id)
        return self._execute(t, lambda task: self._do_upload_log(task, log_path, store=store, day=day))

    def upload_media(self, *, store: str | None = None, day: Any = None,
                     folders: str | Sequence[str] | None = None, task_id: str | None = None) -> dict:
        t = self._register("upload_media", task_id)
        return self._execute(t, lambda task: self._do_upload_media(task, store=store, day=day, folders=folders))

    def upload_day(self, *, store: str | None = None, day: Any = None, csv_path: str | Path | None = None,
                   folders: str | Sequence[str] | None = None, log_path: str | Path | bool | None = None,
                   task_id: str | None = None) -> dict:
        t = self._register("upload_day", task_id)
        return self._execute(
            t, lambda task: self._do_upload_day(task, store=store, day=day, csv_path=csv_path, folders=folders,
                                                log_path=log_path)
        )

    def upload_screenshot(self, image_path: str | Path, *, store: str | None = None,
                          captured_at: str | None = None, task_id: str | None = None) -> dict:
        t = self._register("upload_screenshot", task_id)
        return self._execute(
            t, lambda task: self._do_upload_screenshot(task, image_path, store=store, captured_at=captured_at)
        )

    def collect_media(self, day: Any = None, folders: str | Sequence[str] | None = None) -> list[Path]:
        """掃描所有 folder，回傳該營業日要上傳的檔案（已做規格 §4.4 的檢查）。

        單檔超過 opt.max_file_bytes 的會被略過並記 warning，不存在的資料夾會略過並記 error，都不會中斷；
        檔名不合規或空檔仍會拋 LocalCheckError。
        """
        return self._collect_media(day, folders)[0]

    def _collect_media(self, day: Any = None,
                       folders: str | Sequence[str] | None = None) -> tuple[list[Path], list[dict]]:
        """回傳 (要上傳的檔案, 因超過單檔上限而略過的紀錄)。"""
        d = self._day(day)
        found: dict[str, Path] = {}
        out: list[Path] = []
        skipped: list[dict] = []
        for directory in self.media_dirs(folders, day=d):
            if not directory.is_dir():
                # folder_names 可能列了這家門市沒有的資料夾：記錯誤、略過這一個，其餘資料夾照常上傳
                self.log.error("找不到媒體資料夾：%s（營業日 %s）—— 略過這個資料夾，其餘照常上傳", directory, d)
                self._progress("media_dir_missing", {"dir": str(directory), "biz_date": d})
                continue
            files = ab.scan_media(directory, d, check_size=False)  # 大小由下面自己處理：略過而不是中止
            self.log.info("掃描 %s：%d 個檔案（營業日 %s）", directory, len(files), d)
            for p in files:
                if p.name in found:
                    # 同名檔案在伺服器上是同一個路徑（店/日期/檔名），會互相覆蓋
                    raise ab.LocalCheckError(f"多個資料夾有同名檔案：{p.name}（{found[p.name]} 與 {p}）")
                found[p.name] = p
                size = p.stat().st_size
                if size > self.opt.max_file_bytes:
                    self._record_skip(
                        skipped, p, size, "local",
                        f"超過本機單檔上限 {self.opt.max_file_bytes // 1024 // 1024} MB",
                    )
                    continue
                out.append(p)
        out.sort(key=lambda p: p.name)
        return out, skipped

    def _record_skip(self, skipped: list[dict], path: Path, size: int, source: str, reason: str) -> None:
        """記下一個被略過的檔案：留 warning、觸發 on_progress，並收進結果的 skipped。"""
        item = {"name": path.name, "size": size, "source": source, "reason": reason}
        skipped.append(item)
        self.log.warning("略過媒體檔：%s（%s）—— %s；其餘檔案照常上傳", path.name, ab.human(size), reason)
        self._progress("media_skipped", item)

    def _exclude_server_rejected(self, files: Sequence[Path], skipped: list[dict]) -> list[Path]:
        """v1.1（Deprecated）專用：伺服器曾以 413 拒收過某個大小的單檔，≥ 這個大小的檔案一定也會被拒，送出前就先排除。"""
        with self._lock:
            limit = self._server_reject_bytes
        if limit is None:
            return list(files)
        keep: list[Path] = []
        for p in files:
            size = p.stat().st_size
            if size >= limit:
                self._record_skip(skipped, p, size, "server",
                                  f"伺服器先前已拒收 {ab.human(limit)} 的單檔，送出前先排除")
            else:
                keep.append(p)
        return keep

    # ── 實作（都接受 task，供 cancel 使用）────────────────────────────────
    def _resolve_log(self, day: str, log_path: str | Path | bool | None) -> Path | None:
        """決定這次 CSV 要附上哪個 log 檔（v1.2.4），回傳 None 代表只傳 CSV。

        規格：log 檔名不符時整個請求被拒、CSV 也不會寫入。所以：
        - 明確指定的檔案有問題（找不到、檔名或日期不符、太大）→ 拋 LocalCheckError，什麼都不送；
        - 自動找的檔案（<prefix>/<日>/<log_folder>/<日>.log）不存在或太大 → 留 log 後略過，CSV 照傳。
          log 只供 RD 除錯，不該因為它讓當天的 CSV 傳不上去。
        """
        if log_path is False:
            self.log.info("不附 log 檔（log_path=False）")
            return None
        if log_path not in (None, True):
            path = self._resolve(log_path)
            ab.inspect_log(path, day, self.opt.log_max_bytes)
            return path
        if not self.opt.upload_log:
            return None
        path = self.log_path(day)
        if not path.is_file():
            self.log.info("找不到 log 檔：%s —— 只上傳 CSV", path)
            return None
        try:
            ab.inspect_log(path, day, self.opt.log_max_bytes)
        except ab.LocalCheckError as e:
            self.log.warning("略過 log 檔：%s —— CSV 照常上傳", e)
            self._progress("log_skipped", {"path": str(path), "reason": str(e)})
            return None
        return path

    def _do_upload_csv(self, task: TaskInfo, csv_path: str | Path | None = None, *,
                       store: str | None = None, day: Any = None, inspected: tuple | None = None,
                       log_path: str | Path | bool | None = None) -> dict:
        path = self._resolve(csv_path) if csv_path else self.csv_path(store, day)
        d, s, local_rows = inspected or ab.inspect_csv(path)
        log_file = self._resolve_log(d, log_path)
        parts = [ab.Part("file", path)]
        if log_file is not None:
            parts.append(ab.Part("log", log_file))
            self.log.info("附上 log 檔：%s（%s）", log_file, ab.human(log_file.stat().st_size))
        self._check_cancel(task)
        data = self._request("/api/ingest/csv", parts=parts, task=task)
        data["log_file"] = log_file.name if log_file is not None else None

        warnings = data.get("warnings") or []
        self.log.info(
            "CSV 上傳成功：rows=%s upserted=%s modified=%s removed=%s warnings=%d",
            data.get("rows"), data.get("upserted"), data.get("modified"), data.get("removed"), len(warnings),
        )
        for w in warnings:
            self.log.warning("CSV 格式警告：%s", w)  # §3.5 該列仍寫入，但壞欄位存成空值
            self._fire("on_warning", w)
        if self.opt.verify_rows and data.get("rows") != local_rows:
            raise ab.VerifyError(f"列數不符：本機 {local_rows} 列，伺服器回報 {data.get('rows')} 列")
        self._progress("csv_done", {"store_code": s, "biz_date": d, "rows": data.get("rows"),
                                    "warnings": len(warnings), "log_file": data["log_file"]})
        return data

    def _do_upload_log(self, task: TaskInfo, log_path: str | Path | None = None, *,
                       store: str | None = None, day: Any = None) -> dict:
        """v1.2.8 §6：POST /api/ingest/log，欄位 store_code + file（log 本身，檔名 YYYYMMDD.log）。

        這支 API 只存 log，不碰 CSV 與 Dashboard 的訂單資料。
        （別拿 /api/ingest/csv 送空的 file 來夾帶 log：伺服器會把空檔當成這天 0 列，把訂單整批移除。）
        """
        d, s = self._day(day), self._store(store)
        path = self._resolve(log_path) if log_path else self.log_path(d)
        size = ab.inspect_log(path, d, self.opt.log_max_bytes)  # 找不到、檔名或日期不符、太大 → 送出前擋下
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.log.info("上傳 log 檔（/api/ingest/log）：%s（%s）店=%s", path, ab.human(size), s)
        self._check_cancel(task)
        data = self._request("/api/ingest/log", data={"store_code": s}, parts=[ab.Part("file", path)], task=task)

        # §6.2：回應的 bytes / sha256 是伺服器實際收到的內容，與本機比對
        if data.get("bytes") is not None and data["bytes"] != size:
            raise ab.VerifyError(f"log 大小不符：本機 {size} bytes，伺服器回報 {data['bytes']} bytes")
        if data.get("sha256") and str(data["sha256"]).lower() != digest:
            raise ab.VerifyError(f"log 內容不符：本機 sha256 {digest}，伺服器回報 {data['sha256']}")
        self.log.info("log 上傳成功：%s（%s）店=%s 營業日=%s sha256=%s…", data.get("name") or path.name,
                      ab.human(size), data.get("store_code") or s, data.get("biz_date") or d, digest[:12])
        data["log_file"] = path.name
        self._progress("log_done", {"store_code": s, "biz_date": d, "log_file": path.name, "bytes": size})
        return data

    def _do_upload_media(self, task: TaskInfo, *, store: str | None = None, day: Any = None,
                         folders: str | Sequence[str] | None = None, files: Sequence[Path] | None = None,
                         skipped: Sequence[dict] | None = None) -> dict:
        s, d = self._store(store), self._day(day)
        api = self.opt.media_api
        skipped_out: list[dict] = list(skipped or [])
        if files is not None:
            items = list(files)
        else:
            items, local_skips = self._collect_media(d, folders)
            skipped_out += local_skips
        if api == ab.MEDIA_API_V11:
            items = self._exclude_server_rejected(items, skipped_out)
        if not items:
            if skipped_out:
                self.log.warning("營業日 %s 沒有可上傳的媒體檔（%d 檔被略過）", d, len(skipped_out))
            else:
                self.log.warning("營業日 %s 在 %s 底下沒有任何媒體檔",
                                 d, [str(p) for p in self.media_dirs(folders, day=d)])
            return {"store_code": s, "biz_date": d, "api": api, "stored": [], "skipped": skipped_out,
                    "missing": [], "rounds": 0, "bytes": 0}

        total = sum(p.stat().st_size for p in items)
        self.log.info("待上傳媒體：%d 檔 / %s（%s）", len(items), ab.human(total), self._media_api_text())
        rounds = 0
        if api == ab.MEDIA_API_V11:
            stored = self._upload_media_v11(task, s, d, items, skipped_out)
        else:
            result = self._upload_media_s3(task, s, d, items)
            stored, rounds = result.stored, result.rounds

        done = set(stored)
        uploaded = sum(p.stat().st_size for p in items if p.name in done)
        self.log.info("媒體完成：共 %d 檔 / %s%s%s", len(stored), ab.human(uploaded),
                      f"，{rounds} 輪" if rounds else "",
                      f"，略過 {len(skipped_out)} 檔" if skipped_out else "")
        if skipped_out:
            # CSV 若引用了這些檔案，Dashboard 上對應的那幾筆會沒有影片
            self.log.warning("以下媒體檔未上傳：%s",
                             "；".join(f'{k["name"]}（{k["reason"]}）' for k in skipped_out))
        return {"store_code": s, "biz_date": d, "api": api, "stored": stored, "skipped": skipped_out,
                "missing": [], "rounds": rounds, "bytes": uploaded}

    # ── 媒體 v1.2：直傳 S3 ────────────────────────────────────────────────
    def _upload_media_s3(self, task: TaskInfo, store: str, day: str, items: Sequence[Path]) -> ab.S3UploadResult:
        """取網址 → 逐支 PUT → 回報完成，缺少的重來。迴圈本身是 aibox_upload.run_s3_rounds()，
        這裡只注入傳輸層：API 呼叫走 _request()（重試、重連、取消），PUT 走 _s3_put()。"""
        result = ab.run_s3_rounds(
            items,
            presign=lambda names: self._request(
                "/api/ingest/media/presign", json_body=ab.media_payload(store, day, names), task=task),
            put=lambda path, upload: self._s3_put(task, path, upload),
            complete=lambda names: self._request(
                "/api/ingest/media/complete", json_body=ab.media_payload(store, day, names), task=task),
            settings=self.opt.s3_settings(),
            backoff=self.opt.retry_backoff,
            sleep=lambda seconds: self._sleep(seconds, task),
            checkpoint=lambda: self._check_cancel(task),
            logger=self.log,
            progress=self._progress,
        )
        if result.missing:
            err = ab.MediaIncompleteError(
                f"{result.rounds} 輪後仍有 {len(result.missing)} 檔不在 S3：{', '.join(result.missing[:10])}"
                + (" …" if len(result.missing) > 10 else ""),
                result.stored, result.missing, result.rounds,
            )
            for name in result.missing:
                self.log.error("媒體未完成：%s —— %s", name, result.failures.get(name, ""))
            self.log.error("%s；下個排程時段整天重跑即可（CSV 與媒體重送都是覆蓋）", err)
            self._progress("media_incomplete", {"store_code": store, "biz_date": day, "rounds": result.rounds,
                                                "stored": len(result.stored), "missing": list(result.missing)})
            raise err
        return result

    def _s3_put(self, task: TaskInfo, path: Path, upload: dict) -> ab.PutOutcome:
        with self._lock:
            session = self._s3_session
        if session is None:
            raise NotStartedError("連線已關閉")
        self.log.debug("PUT %s（%s）→ S3", path.name, ab.human(path.stat().st_size))
        return self._interruptible(
            task,
            lambda: ab.s3_put(session, path, upload, timeout=(self.opt.connect_timeout, self.opt.put_timeout),
                              abort=task.cancel if task is not None else None),
            f"PUT {path.name}",
        )

    # ── 媒體 v1.1：經 API 伺服器 multipart 轉存（Deprecated）──────────────
    def _upload_media_v11(self, task: TaskInfo, store: str, day: str, items: Sequence[Path],
                          skipped: list[dict]) -> list[str]:
        """.. deprecated:: v1.2  POST /api/ingest/media 正式上線後停用，請改用預設的 media_api="v1.2"。"""
        ab.warn_deprecated("POST /api/ingest/media（media_api=\"v1.1\"）", logger=self.log)
        stored: list[str] = []
        batches = list(ab.make_batches(items, self.opt.batch_max_files, self.opt.batch_max_bytes))
        for i, batch in enumerate(batches, start=1):
            self._check_cancel(task)
            self.log.info("── 第 %d/%d 批 ──", i, len(batches))
            stored += self._upload_batch(task, store, day, batch, skipped)
            self._progress("media_batch_done", {"batch": i, "batches": len(batches), "stored": len(stored)})
        return stored

    def _upload_batch(self, task: TaskInfo, store: str, day: str, batch: Sequence[Path],
                      skipped: list[dict]) -> list[str]:
        """.. deprecated:: v1.2  v1.1 的單批上傳（含 413 拆批與伺服器上限學習）。"""
        batch = self._exclude_server_rejected(batch, skipped)  # 拆批後的另一半也要先套用剛學到的上限
        if not batch:
            return []
        total = sum(p.stat().st_size for p in batch)
        self.log.info("上傳媒體批次：%d 檔 / %s", len(batch), ab.human(total))
        try:
            data = self._request(
                "/api/ingest/media", data={"store_code": store, "biz_date": day},
                parts=[ab.Part("files[]", p) for p in batch], task=task,
            )
        except ab.TooLargeError as e:
            if len(batch) == 1:
                # 拆到單檔還是 413：伺服器的上限比這個檔案小。略過它，並記住這個大小，之後 ≥ 它的檔案送出前就排除
                with self._lock:
                    if self._server_reject_bytes is None or total < self._server_reject_bytes:
                        self._server_reject_bytes = total
                self.log.warning(
                    "伺服器拒收單一檔案（%s）：%s %s；之後 ≥ %s 的檔案會在送出前先排除"
                    "（本機單檔上限目前是 %d MB，調小 max_file_bytes 可在掃描時就略過）",
                    e, batch[0].name, ab.human(total), ab.human(self._server_reject_bytes),
                    self.opt.max_file_bytes // 1024 // 1024,
                )
                self._record_skip(skipped, batch[0], total, "server", f"伺服器回 413：{e}")
                return []
            mid = len(batch) // 2
            self.log.warning("批次過大（%s），拆成 %d + %d 兩批重送", e, mid, len(batch) - mid)
            return (self._upload_batch(task, store, day, batch[:mid], skipped)
                    + self._upload_batch(task, store, day, batch[mid:], skipped))

        stored = data.get("stored") or []
        if len(stored) != len(batch):  # v1.1 §4.3 請確認 stored 數量等於本批送出的檔案數
            missing = sorted({p.name for p in batch} - set(stored))
            raise ab.VerifyError(f"本批送出 {len(batch)} 檔，伺服器只回報 {len(stored)} 檔；缺：{', '.join(missing)}")
        self.log.info("批次成功：已存 %d 檔", len(stored))
        return list(stored)

    def _do_upload_day(self, task: TaskInfo, *, store: str | None = None, day: Any = None,
                       csv_path: str | Path | None = None,
                       folders: str | Sequence[str] | None = None,
                       log_path: str | Path | bool | None = None) -> dict:
        t0 = time.monotonic()
        path = self._resolve(csv_path) if csv_path else self.csv_path(store, day)
        inspected = ab.inspect_csv(path)
        d, s, _ = inspected
        # 先驗證媒體，免得 CSV 進去了影片卻整個請求被拒（§4.4）；超過單檔上限的在這裡就略過，不中止
        items, skipped = self._collect_media(d, folders)
        self.log.info("每日上傳開始：店=%s 營業日=%s CSV=%s 媒體=%d 檔%s", s, d, path.name, len(items),
                      f"（略過 {len(skipped)} 檔）" if skipped else "")

        csv_result = self._do_upload_csv(task, path, inspected=inspected, log_path=log_path)  # ① CSV 成功才送媒體
        media_result = self._do_upload_media(task, store=s, day=d, files=items, skipped=skipped)

        elapsed = time.monotonic() - t0
        result = {
            "store_code": s, "biz_date": d, "csv": csv_result, "api": media_result.get("api"),
            "stored": media_result.get("stored", []), "skipped": media_result.get("skipped", []),
            "rounds": media_result.get("rounds", 0), "bytes": media_result.get("bytes", 0), "elapsed": elapsed,
        }
        self.log.info(
            "每日上傳完成：店=%s 營業日=%s CSV %s 列、媒體 %d 檔 / %s%s，耗時 %.1f 秒",
            s, d, csv_result.get("rows"), len(result["stored"]), ab.human(result["bytes"]),
            f"、略過 {len(result['skipped'])} 檔" if result["skipped"] else "", elapsed,
        )
        self._progress("day_done", {k: result[k] for k in ("store_code", "biz_date", "elapsed")})
        return result

    def _do_upload_screenshot(self, task: TaskInfo, image_path: str | Path, *, store: str | None = None,
                              captured_at: str | None = None) -> dict:
        s = self._store(store)
        image = self._resolve(image_path)
        if not image.is_file():
            raise ab.LocalCheckError(f"找不到圖檔：{image}")
        if image.suffix.lower() not in ab.SHOT_EXTS:
            raise ab.LocalCheckError(f"截圖只接受 .jpg / .jpeg / .png，實際為 {image.suffix}")
        size = image.stat().st_size
        if size > ab.SHOT_MAX_BYTES:
            raise ab.LocalCheckError(f"截圖超過 10 MB 上限：{ab.human(size)}")

        if captured_at is None:
            # §5.1 建議帶實際拍攝時間；檔案修改時間是最貼近的自動來源
            captured_at = datetime.fromtimestamp(image.stat().st_mtime, ab.JST).strftime("%Y-%m-%dT%H:%M:%S")
            self.log.info("captured_at 未指定，改用檔案修改時間（日本時間）：%s", captured_at)
        if not ab.JST_TIME_RE.match(captured_at):
            raise ab.LocalCheckError(f"captured_at 必須是 YYYY-MM-DDTHH:mm:ss（日本時間）：{captured_at}")

        # §5.3 這支不是冪等的：每次 200 都會新增一張
        data = self._request(
            "/api/ingest/screenshot", data={"store_code": s, "captured_at": captured_at},
            parts=[ab.Part("file", image)], task=task,
        )
        self.log.info("截圖上傳成功：id=%s captured_at=%s(UTC)", data.get("id"), data.get("captured_at"))
        self._progress("screenshot_done", {"id": data.get("id"), "store_code": s})
        return data

    # ── 排程：新增／暫停／恢復／移除（每個動作都留 log）─────────────────────
    def register_pipeline(self, name: str, fn: Callable[..., Any]) -> None:
        """註冊自訂 pipeline（多步驟的 API flow）。排程觸發時以 fn(box, **job.kwargs) 呼叫。"""
        if not callable(fn):
            raise ScheduleError(f"pipeline「{name}」必須是可呼叫的物件")
        if name in self.SUBMITTABLE:
            raise ScheduleError(f"pipeline 名稱與內建的重複：{name}")
        with self._lock:
            self._pipelines[name] = fn
        self.log.info("register_pipeline()：已註冊自訂 pipeline %s", name)

    def pipelines(self) -> list[str]:
        """可排程的 pipeline：內建四個 API flow ＋ register_pipeline() 註冊的。"""
        with self._lock:
            return list(self.SUBMITTABLE) + sorted(self._pipelines)

    def jobs(self) -> list[dict]:
        """所有排程的快照，依下次執行時間排序。"""
        with self._lock:
            items = list(self._jobs.values())
        items.sort(key=lambda j: (j.next_run is None, j.next_run.timestamp() if j.next_run else 0.0, j.name))
        return [j.snapshot() for j in items]

    def job(self, key: str) -> dict | None:
        """用 job id 或 job name 查一個排程。"""
        j = self._find_job(key)
        return j.snapshot() if j else None

    def add_job(self, name: str, pipeline: str, at: Any, every: Any = "day", *,
                on_date: Any = None, weekday: Any = None, day_of_month: Any = None,
                kwargs: dict | None = None, catch_up: bool = False, paused: bool = False) -> dict:
        """新增排程，回傳含 4 碼 job id 的快照。時間一律是本機時間。

        週期（every）與對應的必要參數：
            "date"  指定日期跑一次 → on_date="2026-10-05"。指定的時間已經過了也會觸發：
                    連線中就立刻補跑一次，尚未 start() 則在 start() 之後補跑；跑過一次就不再觸發
            "day"   每天
            "week"  每週 → weekday=0…6（0 = 星期一）或 "mon"…"sun"
            "month" 每月 → day_of_month=1…31 或 -1（月底）；小月會收斂到該月最後一天

        pipeline 是要呼叫的 API flow（見 pipelines()），kwargs 會原樣傳給它。
        名稱不可與現有排程重複；catch_up=True 時，stop() 期間錯過的時段會在下次 start() 補跑一次。
        """
        jname = str(name).strip()
        if not jname:
            raise ScheduleError("job name 不可為空")
        available = self.pipelines()
        if pipeline not in available:
            raise ScheduleError(f"不認識的 pipeline：{pipeline}；可用：{', '.join(available)}")

        rec = _parse_recurrence(every)
        job_at = _parse_at(at)
        on, wd, dom = None, None, None
        if rec is Recurrence.ONCE:
            if on_date is None:
                raise ScheduleError('every="date" 必須指定 on_date，例如 on_date="2026-10-05"')
            on = _parse_on_date(on_date)
        elif rec is Recurrence.WEEKLY:
            if weekday is None:
                raise ScheduleError('every="week" 必須指定 weekday：0-6（0 = 星期一）或 "mon"…"sun"')
            wd = _parse_weekday(weekday)
        elif rec is Recurrence.MONTHLY:
            if day_of_month is None:
                raise ScheduleError('every="month" 必須指定 day_of_month：1-31，或 -1 代表月底')
            dom = _parse_day_of_month(day_of_month)

        with self._lock:
            dup = next((j for j in self._jobs.values() if j.name == jname), None)
            if dup is not None:
                raise ScheduleError(f"排程名稱重複：「{jname}」已經存在（job id {dup.id}）")
            job = Job(
                id=make_job_id(jname, tuple(self._jobs)), name=jname, pipeline=pipeline,
                at=job_at, every=rec, on_date=on, weekday=wd, day_of_month=dom,
                kwargs=dict(kwargs or {}), catch_up=bool(catch_up), paused=bool(paused),
            )
            job.next_run = None if job.paused else self._plan_next(job, _now_local())
            self._jobs[job.id] = job

        self.log.info(
            "add_job()：新增排程 %s「%s」pipeline=%s %s%s，下次執行 %s",
            job.id, job.name, job.pipeline, job.describe(),
            f" kwargs={job.kwargs}" if job.kwargs else "", _next_run_text(job),
        )
        once = job.once_at()
        if once is not None and not job.paused and once <= _now_local():
            self.log.warning("add_job()：%s「%s」指定的時間 %s 已經過了 → %s", job.id, job.name, _time_text(once),
                             "立刻補跑一次" if self._state is State.CONNECTED else "start() 之後立刻補跑一次")
        if self._state is not State.CONNECTED:
            self.log.info("add_job()：目前狀態 %s，排程要等 start() 之後才開始計時", self._state.value)
        self._progress("job_added", job.snapshot())
        self._sched_wake.set()
        return job.snapshot()

    def pause_job(self, key: str) -> bool:
        """暫停排程（job id 或 name）。回傳是否真的改變了狀態。"""
        job = self._find_job(key)
        if job is None:
            self.log.warning("pause_job()：找不到排程 %s", key)
            return False
        if job.paused:
            self.log.info("pause_job()：排程 %s「%s」本來就是暫停狀態", job.id, job.name)
            return False
        with self._lock:
            planned, job.next_run, job.paused = job.next_run, None, True
        self.log.info("pause_job()：排程 %s「%s」已暫停（原定 %s）", job.id, job.name,
                      planned.strftime("%Y-%m-%d %H:%M:%S") if planned else "無")
        self._progress("job_paused", job.snapshot())
        self._sched_wake.set()
        return True

    def resume_job(self, key: str) -> bool:
        """恢復排程。週期性排程暫停期間錯過的時段不補跑，直接從下一個時段開始；
        還沒跑過的單次排程若時間已過，恢復後立刻補跑。"""
        job = self._find_job(key)
        if job is None:
            self.log.warning("resume_job()：找不到排程 %s", key)
            return False
        if not job.paused:
            self.log.info("resume_job()：排程 %s「%s」不在暫停狀態", job.id, job.name)
            return False
        with self._lock:
            job.paused = False
            job.next_run = self._plan_next(job, _now_local())
        self.log.info("resume_job()：排程 %s「%s」已恢復，下次執行 %s", job.id, job.name, _next_run_text(job))
        self._progress("job_resumed", job.snapshot())
        self._sched_wake.set()
        return True

    def remove_job(self, key: str) -> bool:
        """移除排程。已經在跑的那次不會被中斷（要停就自己呼叫 cancel()）。"""
        job = self._find_job(key)
        if job is None:
            self.log.warning("remove_job()：找不到排程 %s", key)
            return False
        with self._lock:
            self._jobs.pop(job.id, None)
        self.log.info("remove_job()：排程 %s「%s」已移除（%s，共觸發 %d 次、成功 %d 次）",
                      job.id, job.name, job.describe(), job.runs, job.successes)
        last = self.task(job.last_task_id) if job.last_task_id else None
        if last and last["state"] in (TaskState.PENDING.value, TaskState.RUNNING.value):
            self.log.info("remove_job()：%s 最後觸發的任務 %s 仍在進行中，未一併取消", job.id, job.last_task_id)
        self._progress("job_removed", job.snapshot())
        self._sched_wake.set()
        return True

    def run_job_now(self, key: str) -> str:
        """立刻觸發一次，不影響原本排定的時間（補跑或驗證用）。回傳 task id。"""
        job = self._find_job(key)
        if job is None:
            raise ScheduleError(f"找不到排程：{key}")
        self.log.info("run_job_now()：手動觸發排程 %s「%s」", job.id, job.name)
        task_id = self._fire_job(job, _now_local(), manual=True)
        if task_id is None:
            raise NotStartedError(f"目前狀態 {self._state.value}，無法觸發排程 {job.id}")
        return task_id

    # ── 排程內部 ───────────────────────────────────────────────────────────
    def _find_job(self, key: str) -> Job | None:
        k = str(key).strip()
        with self._lock:
            return self._jobs.get(k) or next((j for j in self._jobs.values() if j.name == k), None)

    def _plan_next(self, job: Job, after: datetime) -> datetime | None:
        """算出 after 之後的下一次執行時間。

        單次排程：已觸發過回 None；指定的時間已過但還沒跑過，回 after（＝現在就到期，排程執行緒這一輪就觸發）。
        """
        if job.every is Recurrence.ONCE:
            if job.done:
                return None
            when = job.once_at()
            return when if when > after else after
        if job.every is Recurrence.DAILY:
            when = datetime.combine(after.date(), job.at)
            return when if when > after else when + timedelta(days=1)
        if job.every is Recurrence.WEEKLY:
            d = after.date() + timedelta(days=(job.weekday - after.weekday()) % 7)
            when = datetime.combine(d, job.at)
            return when if when > after else when + timedelta(days=7)
        year, month = after.year, after.month
        for _ in range(14):  # 往後找，每月的日可能被小月收斂而落在 after 之前
            when = datetime.combine(date(year, month, _day_in_month(year, month, job.day_of_month)),
                                    job.at)
            if when > after:
                return when
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return None

    def _pipeline_fn(self, job: Job) -> Callable[[TaskInfo], Any]:
        if job.pipeline in self.SUBMITTABLE:
            do = getattr(self, f"_do_{job.pipeline}")
            work = lambda task: do(task, **job.kwargs)  # noqa: E731
        else:
            with self._lock:
                fn = self._pipelines.get(job.pipeline)
            if fn is None:
                raise ScheduleError(f"pipeline 已不存在：{job.pipeline}")
            work = lambda task: fn(self, **job.kwargs)  # noqa: E731

        def run(task: TaskInfo) -> Any:
            # 排定的時間到了：先重新載入 API 金鑰，再跑排定的工作。載入失敗就不跑（任務記為失敗並留 log）
            self.load_api_key(reason=f"排程 {job.id}「{job.name}」")
            return work(task)

        return run

    def _start_scheduler(self) -> None:
        with self._lock:
            if self._sched_thread is not None and self._sched_thread.is_alive():
                return
            now = _now_local()
            armed = 0
            for job in self._jobs.values():
                if job.paused:
                    continue
                missed = job.next_run is not None and job.next_run <= now
                job.next_run = self._plan_next(job, now)
                if job.every is Recurrence.ONCE:
                    if job.next_run is not None and job.next_run <= now:
                        self.log.warning("排程 %s「%s」單次排程的時間 %s 已過、還沒跑過 → 立刻補跑一次",
                                         job.id, job.name, _time_text(job.once_at()))
                elif missed and job.catch_up:
                    job.next_run = now  # 讓排程執行緒這一輪就補跑
                    self.log.warning("排程 %s「%s」錯過了原定時段，catch_up=True → 立刻補跑一次", job.id, job.name)
                elif missed:
                    self.log.warning("排程 %s「%s」錯過了原定時段（未連線期間），不補跑，下次 %s",
                                     job.id, job.name, _next_run_text(job))
                armed += 1
            self._sched_stop.clear()
            self._sched_wake.clear()
            self._sched_thread = threading.Thread(
                target=self._scheduler_loop, name="aibox_aws-scheduler", daemon=True
            )
            self._sched_thread.start()
        self.log.info("排程執行緒啟動：共 %d 個排程，其中 %d 個在計時", len(self._jobs), armed)

    def _stop_scheduler(self) -> None:
        with self._lock:
            thread, self._sched_thread = self._sched_thread, None
        if thread is None:
            return
        self._sched_stop.set()
        self._sched_wake.set()
        thread.join(timeout=5)
        self.log.info("排程執行緒停止：%d 個排程暫停計時（下次 start() 會重新排時間）", len(self._jobs))

    def _scheduler_loop(self) -> None:
        while not self._sched_stop.is_set():
            try:
                wait = self._scheduler_tick()
            except Exception as e:
                # 這條執行緒不能死，不然之後所有排程都不會再被觸發
                self.log.error("排程執行緒發生未預期錯誤（已忽略，繼續計時）：%s", e)
                wait = 5.0
            self._sched_wake.wait(wait)
            self._sched_wake.clear()

    def _scheduler_tick(self) -> float:
        """觸發所有到期的排程，回傳下一次該醒來的秒數。"""
        now = _now_local()
        with self._lock:
            due = [j for j in self._jobs.values()
                   if not j.paused and j.next_run is not None and j.next_run <= now + timedelta(milliseconds=50)]
        for job in due:
            self._fire_job(job, now)
        with self._lock:
            upcoming = [j.next_run for j in self._jobs.values() if not j.paused and j.next_run is not None]
        if not upcoming:
            return _SCHED_IDLE_WAIT
        return max(0.05, min(_SCHED_IDLE_WAIT, (min(upcoming) - _now_local()).total_seconds()))

    def _fire_job(self, job: Job, now: datetime, *, manual: bool = False) -> str | None:
        planned = job.next_run
        once = job.every is Recurrence.ONCE and not manual  # 手動觸發不算數，原定的單次排程照樣會跑
        if not manual:
            # 先排好下一次再送出：這次失敗也不影響後面的時段。單次排程先標成已觸發，下一輪才不會重複送
            with self._lock:
                if once:
                    job.done = True
                job.next_run = self._plan_next(job, now)
        if self._state in (State.IDLE, State.STOPPED):
            with self._lock:
                job.skips += 1
                if once:
                    job.done = False  # 沒送出去：還原，下次 start() 會再補跑
            self.log.warning("排程略過：%s「%s」目前狀態 %s，沒有可用連線（下次 %s）",
                             job.id, job.name, self._state.value, _next_run_text(job))
            self._progress("job_skipped", job.snapshot())
            return None
        try:
            t = self._submit_task(job.pipeline, self._pipeline_fn(job), job_id=job.id)
        except Exception as e:
            with self._lock:
                job.skips += 1
                job.last_state = "SUBMIT_FAILED"
                if once:  # 單次排程只有這一次機會：稍後再試，不要就此消失
                    job.done = False
                    job.next_run = _now_local() + timedelta(seconds=_SCHED_IDLE_WAIT)
            self.log.error("排程無法排入佇列：%s「%s」—— %s（下次 %s）", job.id, job.name, e, _next_run_text(job))
            self._progress("job_skipped", job.snapshot())
            return None
        with self._lock:
            job.runs += 1
            job.last_run = now
            job.last_task_id = t.id
            job.last_state = t.state.value
        self.log.info("排程觸發（%s）：%s「%s」→ pipeline %s，task %s（原定 %s，下次 %s）",
                      "手動" if manual else "自動", job.id, job.name, job.pipeline, t.id,
                      _time_text(job.once_at() or planned) if (job.once_at() or planned) else "—",
                      _next_run_text(job))
        self._progress("job_fired", {**job.snapshot(), "task_id": t.id, "manual": manual})
        if t.future is not None:
            t.future.add_done_callback(lambda _f, j=job, ti=t: self._job_finished(j, ti))
        return t.id

    def _job_finished(self, job: Job, t: TaskInfo) -> None:
        """任務結束後回填統計並留下 log。在工作執行緒裡跑，絕不能往外拋。"""
        try:
            with self._lock:
                job.last_state = t.state.value
                if t.state is TaskState.DONE:
                    job.successes += 1
                elif t.state is TaskState.FAILED:
                    job.failures += 1
            if t.state is TaskState.DONE:
                self.log.info("排程完成：%s「%s」task %s", job.id, job.name, t.id)
            else:
                self.log.warning("排程未完成：%s「%s」task %s 狀態 %s —— %s",
                                 job.id, job.name, t.id, t.state.value, t.error)
            self._progress("job_done", {**job.snapshot(), "task_id": t.id})
        except Exception as e:
            self.log.warning("排程結果回報失敗（已忽略）：%s", e)

    # ── with 區塊 ──────────────────────────────────────────────────────────
    def __enter__(self) -> "aibox_aws":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


AiboxAws = aibox_aws  # PEP8 風格的別名
