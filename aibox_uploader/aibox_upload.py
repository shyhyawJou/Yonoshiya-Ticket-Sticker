#!/usr/bin/env python3
"""AI Box → 吉野家 Dashboard 上傳工具（Python 實作）

依 AIBOX_UPLOAD.md v1.2 規格實作，可直接掛排程使用。

v1.2 起影片與訂單截圖改為**直傳 S3**（§4）：先向 API 取得短效上傳網址
（/api/ingest/media/presign），逐支 PUT 到 S3，再回報完成（/api/ingest/media/complete）；
伺服器列出缺少的檔名，就只對那些重新取網址重來，最多 3 輪。

v1.1 的 POST /api/ingest/media（經 API 伺服器以 multipart 轉存）仍保留，但已 **Deprecated**：
規格 v1.2 §4.7 表示正式上線後停用。只有指定 --media-api v1.1（或 YOSHINOYA_MEDIA_API=v1.1）才會用到。

用法：
    python aibox_upload.py health
    python aibox_upload.py day   /data/20260909_7d2f.csv /data/media
    python aibox_upload.py csv   /data/20260909_7d2f.csv
    python aibox_upload.py csv   /data/20260909_7d2f.csv --log /data/stream_log/20260909.log   # v1.2.4 附 log
    python aibox_upload.py media 7d2f 20260909 /data/media
    python aibox_upload.py shot  7d2f /tmp/screenshot.png
    python aibox_upload.py day   /data/20260909_7d2f.csv /data/media --media-api v1.1   # Deprecated

環境變數（明確的旗標優先，其次環境變數，最後規格預設值）：
    YOSHINOYA_API_KEY          必要，由系統管理者提供。沒設定時改從設定檔讀取（見下）
    YOSHINOYA_URL             預設 https://57.181.173.247:5000
    YOSHINOYA_CA               沒設定時讀設定檔，再退回 <本檔所在目錄>/prod-57.181.173.247.crt（只用於 API；S3 用系統內建 CA）
    YOSHINOYA_MEDIA_API        媒體上傳方式：v1.2（預設，直傳 S3）或 v1.1（Deprecated）
    YOSHINOYA_MAX_FILE_MB      媒體單檔上限 MB；預設 v1.2 = 5120（S3 單次 PUT 上限）、v1.1 = 200；`--max-file-mb` 優先
    YOSHINOYA_S3_URL           health 探測用的 S3 位址，預設 https://yoshinoya-dashboard-media.s3.ap-northeast-1.amazonaws.com
    YOSHINOYA_PRESIGN_BATCH    每次取上傳網址／回報完成的檔案數，預設 1000（規格上限）
    YOSHINOYA_MEDIA_ROUNDS     缺少的檔案最多重來幾輪，預設 3
    YOSHINOYA_PUT_TIMEOUT      單支 S3 PUT 的讀取逾時（秒），預設 1800
    YOSHINOYA_PRESIGN_REFRESH  上傳網址剩不到幾秒有效就重新取，預設 300
    YOSHINOYA_PUT_ABORT_AFTER  連續幾支因連線問題 PUT 失敗就提前結束本輪，預設 5（0 = 不提前）
    YOSHINOYA_LOG_MAX_MB       v1.2.4：隨 CSV 附上的 AI Box log 檔上限 MB，預設 20；`--log-max-mb` 優先

API 金鑰的來源（依序找）：環境變數 YOSHINOYA_API_KEY → 設定檔 /etc/yoshinoya_ai/cloud_upload.env
（`--env-file` 可改路徑）。設定檔是 KEY=VALUE 格式，# 開頭為註解。
憑證的來源（依序找）：--ca → 環境變數 YOSHINOYA_CA → 設定檔裡的 YOSHINOYA_CA → 本檔旁邊的 prod-57.181.173.247.crt。

結束代碼：0 = 全部成功、1 = 失敗、2 = 參數或環境設定錯誤。
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
import time
import warnings
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import requests

log = logging.getLogger("aibox")

# ── 規格常數（AIBOX_UPLOAD.md §2–§5）────────────────────────────────────────
BASE_URL_DEFAULT = "https://57.181.173.247:5000"
CA_FILE_DEFAULT = "prod-57.181.173.247.crt"
JST = timezone(timedelta(hours=9))

CSV_NAME_RE = re.compile(r"^(\d{8})_([A-Za-z0-9_-]{1,32})\.csv$")
STORE_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
BIZ_DATE_RE = re.compile(r"^\d{8}$")
MEDIA_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")  # §4.4 不可含空白、中日文
JST_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?$")

CSV_REQUIRED_COLUMNS = ("order_number", "start_time", "end_time", "final_status")
CSV_MAX_BYTES = 20 * 1024 * 1024

# v1.2.4 §3.1：CSV 可選擇附上當日 AI Box log 檔（欄位 log），檔名必須是 YYYYMMDD.log、日期與 CSV 相同
LOG_MAX_MB = 20  # 規格上限，也是預設值；實際採用值見 env_log_max_bytes()
LOG_MAX_MB_ENV = "YOSHINOYA_LOG_MAX_MB"

MEDIA_EXTS = {".mp4", ".jpg", ".jpeg", ".png"}
MEDIA_MAX_FILE_BYTES = 200 * 1024 * 1024  # v1.1（Deprecated）的伺服器單檔上限，也是 v1.1 模式的預設值
S3_MAX_FILE_BYTES = 5 * 1024 * 1024 * 1024  # v1.2 §4.4：S3 單次 PUT 上限，也是 v1.2 模式的預設值
MAX_FILE_MB_ENV = "YOSHINOYA_MAX_FILE_MB"
BATCH_MAX_FILES = 20  # v1.1（Deprecated）：規格上限 100，建議值 20
BATCH_MAX_BYTES = 200 * 1024 * 1024  # v1.1（Deprecated）：規格單次請求上限 250 MB，建議值 200 MB

# v1.2 直傳 S3（§4.1–§4.4）
MEDIA_API_V12 = "v1.2"
MEDIA_API_V11 = "v1.1"  # Deprecated：POST /api/ingest/media
MEDIA_API_ENV = "YOSHINOYA_MEDIA_API"
S3_URL_DEFAULT = "https://yoshinoya-dashboard-media.s3.ap-northeast-1.amazonaws.com"
S3_URL_ENV = "YOSHINOYA_S3_URL"
PRESIGN_MAX_FILES = 1000  # §4.1 每次取網址的上限，也是預設的分段大小
PRESIGN_BATCH_ENV = "YOSHINOYA_PRESIGN_BATCH"
MEDIA_ROUNDS = 3  # §4.3 缺少的檔案建議最多重來 3 輪
MEDIA_ROUNDS_ENV = "YOSHINOYA_MEDIA_ROUNDS"
PUT_TIMEOUT = 1800  # §6 單支影片 PUT 建議 ≥ 30 分鐘
PUT_TIMEOUT_ENV = "YOSHINOYA_PUT_TIMEOUT"
PRESIGN_REFRESH = 300  # 網址剩不到 5 分鐘有效就重新取（§4.4 網址 1 小時失效）
PRESIGN_REFRESH_ENV = "YOSHINOYA_PRESIGN_REFRESH"
PUT_ABORT_AFTER = 5  # 連續 5 支因連線問題失敗就提前結束本輪，0 = 不提前
PUT_ABORT_AFTER_ENV = "YOSHINOYA_PUT_ABORT_AFTER"

SHOT_EXTS = {".jpg", ".jpeg", ".png"}
SHOT_MAX_BYTES = 10 * 1024 * 1024

CONTENT_TYPES = {
    ".csv": "text/csv",
    ".log": "text/plain",
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

CONNECT_TIMEOUT = 10
READ_TIMEOUT = 600  # §6 API 呼叫的讀取逾時建議 ≥ 300 秒
MAX_ATTEMPTS = 3  # §6 最多 3 次
RETRY_BACKOFF = 10  # 間隔 10、20 秒遞增


# ── 例外 ────────────────────────────────────────────────────────────────────
class ConfigError(Exception):
    """參數或環境設定錯誤，結束代碼 2。"""


class LocalCheckError(Exception):
    """本機驗證未通過，沒有送出任何請求。"""


class ApiError(Exception):
    """伺服器明確回報請求有問題（4xx 或 ok:false）—— 依規格不重試。"""

    def __init__(self, status: int, error: str):
        super().__init__(f"HTTP {status}: {error}")
        self.status = status
        self.error = error


class TooLargeError(ApiError):
    """HTTP 413。v1.1 的媒體批次可拆小後重送，單檔過大則無解。"""


class TransientError(Exception):
    """連線問題或 5xx，重試用盡後仍失敗。"""


class VerifyError(Exception):
    """上傳有回 200，但回應內容與本機送出的量不符。"""


class UploadAborted(Exception):
    """s3_put() 的 abort 事件被設定，檔案讀到一半就停止送出（aibox_aws 的 cancel 用）。"""


class MediaIncompleteError(VerifyError):
    """v1.2：所有輪數跑完，伺服器回報完成時仍有檔案不在 S3。下一個排程時段整天重跑即可。"""

    def __init__(self, message: str, stored: Sequence[str], missing: Sequence[str], rounds: int):
        super().__init__(message)
        self.stored = list(stored)
        self.missing = list(missing)
        self.rounds = rounds


# ── 可設定的值（明確參數 > 環境變數 > 規格預設值）────────────────────────────
_MEDIA_API_ALIASES = {
    "v1.2": MEDIA_API_V12, "1.2": MEDIA_API_V12, "s3": MEDIA_API_V12, "presign": MEDIA_API_V12,
    "v1.1": MEDIA_API_V11, "1.1": MEDIA_API_V11, "legacy": MEDIA_API_V11, "multipart": MEDIA_API_V11,
}


def resolve_media_api(explicit: str | None = None) -> str:
    """媒體上傳方式：明確指定 > YOSHINOYA_MEDIA_API > v1.2。回傳 "v1.2" 或 "v1.1"。"""
    raw = explicit if explicit not in (None, "") else (os.environ.get(MEDIA_API_ENV) or "").strip()
    if not raw:
        return MEDIA_API_V12
    key = str(raw).strip().lower()
    if key not in _MEDIA_API_ALIASES:
        where = "media_api" if explicit not in (None, "") else MEDIA_API_ENV
        raise ConfigError(f"{where} 必須是 v1.2（直傳 S3）或 v1.1（Deprecated），實際為 {raw!r}")
    return _MEDIA_API_ALIASES[key]


def default_max_file_bytes(media_api: str | None = None) -> int:
    """單檔上限的規格預設值：v1.2 是 S3 的 5 GB，v1.1（Deprecated）是 API 伺服器的 200 MB。"""
    return MEDIA_MAX_FILE_BYTES if resolve_media_api(media_api) == MEDIA_API_V11 else S3_MAX_FILE_BYTES


def env_max_file_bytes(explicit_mb: int | None = None, default: int | None = None) -> int:
    """決定媒體單檔上限（bytes）：明確指定的 MB > 環境變數 > 規格預設值。

    default 未給時依目前的媒體上傳方式（YOSHINOYA_MEDIA_API）取 5 GB 或 200 MB。
    這個值一定要在送出前擋：v1.1 是一批裡有一檔超限就整批被拒，而伺服器是在收到超限的
    位元組時就中止連線，用戶端只會看到 SSL 寫入錯誤而不是乾淨的 413，很難判斷原因。
    做成可設定是因為前面若還有代理層設了更小的上限，得能跟著調小。
    """
    if explicit_mb is None:
        raw = (os.environ.get(MAX_FILE_MB_ENV) or "").strip()
        if not raw:
            return default_max_file_bytes() if default is None else default
        try:
            explicit_mb = int(raw)
        except ValueError:
            raise ConfigError(f"{MAX_FILE_MB_ENV} 必須是整數 MB，實際為 {raw!r}") from None
    if explicit_mb <= 0:
        raise ConfigError(f"單檔上限必須大於 0 MB，實際為 {explicit_mb}")
    return explicit_mb * 1024 * 1024


def env_number(env: str, explicit: Any, default: Any, *, label: str, cast: Callable = int,
               minimum: float = 1, unit: str = "") -> Any:
    """通用的三層解析：explicit（None 代表往下一層）> 環境變數 env > default。"""
    source = label
    value = explicit
    if value is None:
        raw = (os.environ.get(env) or "").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError:
            kind = "整數" if cast is int else "數字"
            raise ConfigError(f"{env} 必須是{kind}（單位：{unit.strip() or '無'}），實際為 {raw!r}") from None
        source = env
    if value < minimum:
        raise ConfigError(f"{source} 必須 ≥ {minimum}{unit}，實際為 {value}")
    return value


@dataclass(frozen=True)
class S3Settings:
    """v1.2 直傳 S3 的參數。用 resolve() 建立，每個值都是 明確參數 > 環境變數 > 規格預設值。"""

    presign_batch: int = PRESIGN_MAX_FILES
    rounds: int = MEDIA_ROUNDS
    put_timeout: float = PUT_TIMEOUT
    refresh_margin: float = PRESIGN_REFRESH
    abort_after: int = PUT_ABORT_AFTER
    s3_url: str = S3_URL_DEFAULT

    @classmethod
    def resolve(cls, *, presign_batch: int | None = None, rounds: int | None = None,
                put_timeout: float | None = None, refresh_margin: float | None = None,
                abort_after: int | None = None, s3_url: str | None = None) -> "S3Settings":
        return cls(
            presign_batch=env_number(PRESIGN_BATCH_ENV, presign_batch, PRESIGN_MAX_FILES,
                                     label="presign_batch", unit=" 檔"),
            rounds=env_number(MEDIA_ROUNDS_ENV, rounds, MEDIA_ROUNDS, label="media_rounds", unit=" 輪"),
            put_timeout=env_number(PUT_TIMEOUT_ENV, put_timeout, PUT_TIMEOUT, label="put_timeout",
                                   cast=float, unit=" 秒"),
            refresh_margin=env_number(PRESIGN_REFRESH_ENV, refresh_margin, PRESIGN_REFRESH,
                                      label="presign_refresh", cast=float, minimum=0, unit=" 秒"),
            abort_after=env_number(PUT_ABORT_AFTER_ENV, abort_after, PUT_ABORT_AFTER,
                                   label="put_abort_after", minimum=0, unit=" 支"),
            s3_url=(s3_url or os.environ.get(S3_URL_ENV) or S3_URL_DEFAULT).rstrip("/"),
        )


# ── Deprecated 標記 ─────────────────────────────────────────────────────────
def warn_deprecated(what: str, logger: Any = None, stacklevel: int = 3) -> None:
    """v1.1 媒體上傳的入口都呼叫這裡：發 DeprecationWarning，並在 log 留一行 warning。"""
    msg = (f"{what} 是 API v1.1 的媒體上傳方式，已 Deprecated（規格 v1.2 §4.7：正式上線後停用）。"
           f"請改用 v1.2 直傳 S3：--media-api v1.2 / media_api=\"v1.2\"")
    warnings.warn(msg, DeprecationWarning, stacklevel=stacklevel)
    (logger or log).warning("[Deprecated] %s", msg)


# ── 工具函式 ────────────────────────────────────────────────────────────────
def human(n: int) -> str:
    if n >= 1024 * 1024 * 1024:
        return f"{n / 1024 / 1024 / 1024:.2f} GB"
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def chunks(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for i in range(0, len(items), max(1, size)):
        yield list(items[i:i + max(1, size)])


def parse_response(r: requests.Response) -> dict:
    """依 §2.4 解析回應：先看 HTTP 狀態碼，再讀 JSON。4xx 一律轉成 ApiError（不重試）。"""
    if "json" in r.headers.get("content-type", "").lower():
        body = r.json()
    else:
        # §2.4 例外：v1.1 送到 API 的請求總量超過 250 MB 時，前端代理直接回 HTML 413
        body = {"ok": False, "error": f"非 JSON 回應：{r.text[:200]!r}"}
    if r.ok and body.get("ok"):
        return body.get("data") or {}
    error = body.get("error") or f"HTTP {r.status_code}"
    if r.status_code == 413:
        raise TooLargeError(r.status_code, error)
    raise ApiError(r.status_code, error)


@dataclass(frozen=True)
class Part:
    """一個 multipart 檔案欄位。只記路徑，每次嘗試都重新開檔，避免重試時位置沒歸零。"""

    field: str
    path: Path


class Client:
    def __init__(self, base: str, ca: str, api_key: str):
        self.base = base.rstrip("/")
        self.ca = ca
        self.headers = {"X-Api-Key": api_key}
        self.session = requests.Session()
        # S3 用另一個 session：不帶 X-Api-Key（金鑰不能送到 AWS），並用系統內建 CA 而不是釘選的自簽憑證（§2.2）
        self.s3 = requests.Session()

    # ── 健康檢查（不需金鑰）────────────────────────────────────────────────
    def healthz(self) -> dict:
        r = self.session.get(
            self.base + "/api/healthz", verify=self.ca, timeout=(CONNECT_TIMEOUT, 30)
        )
        r.raise_for_status()
        return r.json()

    # ── 共通 POST：網路錯誤與 5xx 重試，4xx 立即失敗（§6）──────────────────
    def post(
        self,
        path: str,
        data: dict | None = None,
        parts: Sequence[Part] = (),
        attempts: int = MAX_ATTEMPTS,
    ) -> dict:
        """multipart/form-data 的 POST（CSV、截圖，以及 Deprecated 的 v1.1 媒體）。"""

        def send() -> requests.Response:
            with ExitStack() as stack:
                files = [
                    (p.field, (p.path.name, stack.enter_context(p.path.open("rb")), content_type(p.path)))
                    for p in parts
                ]
                return self.session.post(
                    self.base + path,
                    headers=self.headers,
                    data=data,
                    files=files or None,
                    verify=self.ca,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                )

        return self._retrying(path, send, attempts)

    def post_json(self, path: str, payload: dict, attempts: int = MAX_ATTEMPTS) -> dict:
        """JSON 的 POST（v1.2 的 presign / complete）。重試規則同 post()。"""
        return self._retrying(
            path,
            lambda: self.session.post(self.base + path, headers=self.headers, json=payload,
                                      verify=self.ca, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)),
            attempts,
        )

    def _retrying(self, path: str, send: Callable[[], requests.Response], attempts: int) -> dict:
        last = ""
        for attempt in range(1, attempts + 1):
            try:
                r = send()
                if r.status_code < 500:
                    return parse_response(r)
                last = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"

            if attempt == attempts:
                raise TransientError(f"{path} 重試 {attempts} 次後仍失敗：{last}")
            delay = RETRY_BACKOFF * attempt
            log.warning("%s 失敗（%s），%d 秒後重試（第 %d/%d 次）", path, last, delay, attempt + 1, attempts)
            time.sleep(delay)
        raise AssertionError("unreachable")


# ── CSV：本機檢查（§3.1–§3.2）──────────────────────────────────────────────
def inspect_csv(path: Path) -> tuple[str, str, int]:
    """回傳 (biz_date, store_code, 本機資料列數)。不合規就在送出前擋下來。"""
    if not path.is_file():
        raise LocalCheckError(f"找不到 CSV：{path}")
    m = CSV_NAME_RE.match(path.name)
    if not m:
        raise LocalCheckError(f"CSV 檔名必須是 YYYYMMDD_<store_code>.csv，實際為 {path.name}")
    day, store = m.group(1), m.group(2)
    try:
        datetime.strptime(day, "%Y%m%d")
    except ValueError:
        raise LocalCheckError(f"CSV 檔名的日期不是有效日期：{day}") from None

    size = path.stat().st_size
    if size > CSV_MAX_BYTES:
        raise LocalCheckError(f"CSV 超過 20 MB 上限：{human(size)}")
    if size == 0:
        raise LocalCheckError(f"CSV 是空檔：{path.name}")

    # utf-8-sig：規格說 BOM 可有可無
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = [c.strip() for c in next(reader)]
        except StopIteration:
            raise LocalCheckError("CSV 沒有標題列") from None
        missing = [c for c in CSV_REQUIRED_COLUMNS if c not in header]
        if missing:
            raise LocalCheckError(f"CSV 缺少必要欄位：{', '.join(missing)}")
        rows = sum(1 for row in reader if any(cell.strip() for cell in row))

    log.info("CSV 檢查通過：%s 店＝%s 日期＝%s 列數＝%d 大小＝%s", path.name, store, day, rows, human(size))
    return day, store, rows


# ── log 檔：本機檢查（v1.2.4 §3.1）──────────────────────────────────────────
def env_log_max_bytes(explicit_mb: int | None = None) -> int:
    """log 檔上限（bytes）：明確指定的 MB > 環境變數 YOSHINOYA_LOG_MAX_MB > 規格的 20 MB。"""
    return env_number(LOG_MAX_MB_ENV, explicit_mb, LOG_MAX_MB, label="log 檔上限", unit=" MB") * 1024 * 1024


def inspect_log(path: Path, day: str, max_bytes: int | None = None) -> int:
    """檢查要隨 CSV 附上的 log 檔，回傳大小。

    規格：log 檔名不符時「整個請求拒收（HTTP 400），CSV 也不會寫入」，所以一定要在送出前擋下，
    免得因為 log 的問題連 CSV 都沒傳上去。
    """
    limit = env_log_max_bytes() if max_bytes is None else max_bytes
    if not path.is_file():
        raise LocalCheckError(f"找不到 log 檔：{path}")
    if path.name != f"{day}.log":
        raise LocalCheckError(f"log 檔名必須是 {day}.log（YYYYMMDD.log，日期與 CSV 相同），實際為 {path.name}")
    size = path.stat().st_size
    if size > limit:
        raise LocalCheckError(f"log 檔超過 {limit // 1024 // 1024} MB 上限：{path.name} {human(size)}")
    return size


def upload_csv(
    client: Client, path: Path, verify: bool = True, inspected: tuple[str, str, int] | None = None,
    *, log_path: Path | None = None, log_max_bytes: int | None = None,
) -> dict:
    """上傳 CSV。log_path 給了就一起送（v1.2.4，欄位名 log）；不給則行為與之前相同。"""
    day, store, local_rows = inspected or inspect_csv(path)
    parts = [Part("file", path)]
    if log_path is not None:
        size = inspect_log(log_path, day, log_max_bytes)
        parts.append(Part("log", log_path))
        log.info("附上 log 檔：%s（%s）", log_path.name, human(size))
    data = client.post("/api/ingest/csv", parts=parts)

    warnings_ = data.get("warnings") or []
    log.info(
        "CSV 上傳成功：rows=%s upserted=%s modified=%s removed=%s warnings=%d",
        data.get("rows"),
        data.get("upserted"),
        data.get("modified"),
        data.get("removed"),
        len(warnings_),
    )
    for w in warnings_:
        log.warning("CSV 格式警告：%s", w)  # §3.5 該列仍寫入，但壞欄位存成空值

    # §3.3 請與本機列數核對
    if verify and data.get("rows") != local_rows:
        raise VerifyError(f"列數不符：本機 {local_rows} 列，伺服器回報 {data.get('rows')} 列")
    data["store_code"] = data.get("store_code", store)
    data["biz_date_compact"] = day
    data["log_file"] = log_path.name if log_path is not None else None
    return data


# ── 媒體：掃描（v1.2 與 v1.1 共用）─────────────────────────────────────────
def scan_media(media_dir: Path, day: str, max_file_bytes: int | None = None, *,
               check_size: bool = True) -> list[Path]:
    """挑出檔名以該營業日開頭、副檔名合規的檔案，並先驗證檔名與大小。

    §4.4：files 中只要有一個檔名不合規定，整個請求都會被拒絕。所以寧可先在本機擋下來。
    max_file_bytes 未給時取 env_max_file_bytes()（環境變數，或依媒體上傳方式取 5 GB／200 MB）。
    check_size=False 時不檢查大小（檔名與空檔照樣檢查），由呼叫端自行決定超限的檔案怎麼處理。
    """
    if max_file_bytes is None:
        max_file_bytes = env_max_file_bytes()
    if not media_dir.is_dir():
        raise LocalCheckError(f"找不到媒體資料夾：{media_dir}")

    files = sorted(
        p
        for p in media_dir.iterdir()
        if p.is_file() and p.name.startswith(day) and p.suffix.lower() in MEDIA_EXTS
    )
    problems = []
    for p in files:
        size = p.stat().st_size
        if not MEDIA_NAME_RE.match(p.name):
            problems.append(f"{p.name}：檔名只能用英數、'.'、'-'、'_'，且不超過 200 字元")
        elif check_size and size > max_file_bytes:
            problems.append(f"{p.name}：單檔超過 {max_file_bytes // 1024 // 1024} MB 上限（{human(size)}）")
        elif size == 0:
            problems.append(f"{p.name}：空檔")
    if problems:
        raise LocalCheckError("媒體檔不合規，整天中止（未送出任何檔案）：\n  - " + "\n  - ".join(problems))

    skipped = [p.name for p in media_dir.iterdir() if p.is_file() and p.name.startswith(day) and p.suffix.lower() not in MEDIA_EXTS]
    for name in skipped:
        log.info("略過非媒體檔：%s", name)
    return files


# ── 媒體 v1.2：直傳 S3（§4.1–§4.3）─────────────────────────────────────────
@dataclass
class PutOutcome:
    """一支檔案 PUT 到 S3 的結果。network=True 代表連線層失敗（逾時、斷線、TLS），算進「連續失敗」。"""

    ok: bool
    status: int | None = None
    detail: str = ""
    network: bool = False


@dataclass
class S3UploadResult:
    stored: list[str]                 # 伺服器 complete 確認已在 S3 的檔名
    missing: list[str]                # 所有輪數跑完仍缺少的檔名（空 = 完成）
    rounds: int                       # 實際跑了幾輪
    presign_calls: int = 0
    put_ok: int = 0
    put_failed: int = 0
    bytes_put: int = 0                # 成功 PUT 的位元組數
    failures: dict[str, str] = field(default_factory=dict)  # 檔名 → 最後一次失敗的原因


def media_payload(store: str, day: str, names: Sequence[str]) -> dict:
    """presign 與 complete 的請求本體（§4.1、§4.3 相同）。"""
    return {"store_code": store, "biz_date": day, "files": list(names)}


class _AbortableReader:
    """包住檔案：每讀一塊就檢查 abort，被設定就拋 UploadAborted，讓上傳在送到一半時停下來。

    提供 __len__ 給 requests 算 Content-Length（S3 的預簽 PUT 不接受 chunked 傳輸）。
    """

    def __init__(self, f: Any, size: int, abort: Any):
        self._f, self._size, self._abort = f, size, abort

    def read(self, n: int = -1) -> bytes:
        if self._abort.is_set():
            raise UploadAborted("上傳已中止")
        return self._f.read(n)

    def __len__(self) -> int:
        return self._size


def s3_put(session: requests.Session, path: Path, upload: dict,
           timeout: tuple[float, float] = (CONNECT_TIMEOUT, PUT_TIMEOUT), abort: Any = None) -> PutOutcome:
    """§4.2：請求本體就是檔案內容（不是 multipart），並原樣帶上 presign 回應的 headers。

    Content-Type 是簽名的一部分，必須與回應一致，否則 S3 回 403。不帶 X-Api-Key、不用釘選憑證。
    失敗不在這裡重試同一個網址 —— 交給 complete 的 missing 決定（§4.2、§6）。
    abort（threading.Event）被設定時，下一次讀檔就拋 UploadAborted，不再繼續送出。
    """
    headers = {str(k): str(v) for k, v in (upload.get("headers") or {}).items()}
    try:
        with path.open("rb") as f:
            body = f if abort is None else _AbortableReader(f, path.stat().st_size, abort)
            r = session.put(upload["url"], data=body, headers=headers, timeout=timeout)
        if r.status_code == 200:
            return PutOutcome(True, 200)
        return PutOutcome(False, r.status_code, f"S3 HTTP {r.status_code}：{r.text[:200]!r}")
    except requests.RequestException as e:
        return PutOutcome(False, None, f"{type(e).__name__}: {e}", network=True)


def probe_s3(url: str | None = None, timeout: float = CONNECT_TIMEOUT,
             session: requests.Session | None = None) -> dict:
    """§8 第 1 項後半：S3 有任何 HTTP 回應（403 也算）就是連得到。用系統內建 CA 驗證。"""
    target = (url or os.environ.get(S3_URL_ENV) or S3_URL_DEFAULT).rstrip("/")
    try:
        r = (session or requests).head(target, timeout=(timeout, 30), allow_redirects=False)
        return {"ok": True, "status": r.status_code, "url": target, "error": None}
    except requests.RequestException as e:
        return {"ok": False, "status": None, "url": target, "error": f"{type(e).__name__}: {e}"}


def run_s3_rounds(
    files: Sequence[Path],
    *,
    presign: Callable[[list[str]], dict],
    put: Callable[[Path, dict], PutOutcome],
    complete: Callable[[list[str]], dict],
    settings: S3Settings | None = None,
    backoff: float = RETRY_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
    checkpoint: Callable[[], None] | None = None,
    logger: Any = None,
    progress: Callable[[str, dict], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> S3UploadResult:
    """v1.2 §4 的對帳迴圈：取網址 → 逐支 PUT → 回報完成 → 只對缺少的重來，最多 settings.rounds 輪。

    傳輸層由呼叫端注入，CLI 與 aibox_aws 共用同一套邏輯：
        presign(names)  → /api/ingest/media/presign 的 data
        put(path, u)    → PutOutcome（u 是 presign 回應中的一筆 upload）
        complete(names) → /api/ingest/media/complete 的 data
    checkpoint() 在每支檔案前呼叫（aibox_aws 用來檢查 cancel）；sleep 可換成可被中斷的等待。

    另外處理兩件規格沒寫、但現場一定會遇到的事：
    - 網址 1 小時失效：頻寬差時一輪可能超過 1 小時，所以網址剩不到 refresh_margin 秒有效時，
      對還沒傳的檔案重新取網址，不必等到下一輪才發現 403。
    - 網路整個斷掉：連續 abort_after 支都是連線層失敗就提前結束本輪，
      不必對幾百支檔案逐一等連線逾時；剩下的交給回報完成列為缺少，下一輪再來。
    """
    st = settings or S3Settings.resolve()
    lg = logger or log
    cp = checkpoint or (lambda: None)
    emit = progress or (lambda event, info: None)

    by_name = {p.name: p for p in files}
    todo = sorted(by_name)
    total = len(todo)
    stored: set[str] = set()
    failures: dict[str, str] = {}
    counts = {"presign": 0, "ok": 0, "failed": 0, "bytes": 0}
    rounds_run = 0

    def do_presign(names: list[str]) -> tuple[dict, float, float]:
        counts["presign"] += 1
        data = presign(list(names))
        uploads = {u.get("name"): u for u in (data.get("uploads") or []) if isinstance(u, dict)}
        expires = float(data.get("expires_in") or 3600)
        lg.info("取得上傳網址：%d 檔（有效 %.0f 秒）", len(names), expires)
        emit("s3_presign", {"files": len(names), "expires_in": expires})
        return uploads, expires, clock()

    def do_complete(names: list[str]) -> set[str]:
        ok: set[str] = set()
        for part in chunks(names, st.presign_batch):  # complete 的請求格式同 presign，一樣受 1000 檔限制
            ok |= set(complete(part).get("stored") or [])
        return ok

    for rnd in range(1, st.rounds + 1):
        rounds_run = rnd
        cp()
        lg.info("── S3 第 %d/%d 輪：%d 檔待上傳 ──", rnd, st.rounds, len(todo))
        emit("s3_round_start", {"round": rnd, "rounds": st.rounds, "files": len(todo)})
        streak = 0
        index = 0
        pending = list(todo)
        while pending:
            chunk, pending = pending[:st.presign_batch], pending[st.presign_batch:]
            uploads, expires, issued = do_presign(chunk)
            i = 0
            while i < len(chunk):
                cp()
                name = chunk[i]
                age = clock() - issued
                margin = min(st.refresh_margin, expires / 2)  # 伺服器若給很短的有效期，至少用一半時間再換
                if age >= expires - margin:
                    lg.info("上傳網址已用 %.0f 秒（有效 %.0f 秒），重新取得剩下 %d 檔的網址", age, expires, len(chunk) - i)
                    uploads, expires, issued = do_presign(chunk[i:])
                index += 1
                path = by_name[name]
                size = path.stat().st_size
                u = uploads.get(name)
                t0 = clock()
                if u is None:
                    out = PutOutcome(False, detail="presign 回應沒有這個檔名")
                elif str(u.get("method") or "PUT").upper() != "PUT" or not u.get("url"):
                    raise VerifyError(f"presign 回應的上傳方式不是 PUT 或沒有網址：{name} {u.get('method')!r}")
                else:
                    out = put(path, u)
                if out.ok:
                    counts["ok"] += 1
                    counts["bytes"] += size
                    streak = 0
                    lg.info("S3 上傳 %d/%d：%s（%s，%.1f 秒）", index, len(todo), name, human(size), clock() - t0)
                else:
                    counts["failed"] += 1
                    failures[name] = out.detail
                    streak = streak + 1 if out.network else 0
                    lg.warning("S3 上傳失敗 %d/%d：%s —— %s（不重試同一網址，交給回報完成判斷）",
                               index, len(todo), name, out.detail)
                emit("s3_put", {"name": name, "ok": out.ok, "size": size, "round": rnd,
                                "index": index, "total": len(todo), "detail": out.detail})
                i += 1
                if not out.ok and st.abort_after and streak >= st.abort_after:
                    left = len(chunk) - i + len(pending)
                    lg.warning("連續 %d 支因連線問題失敗，本輪提前結束（剩下 %d 檔留到下一輪）", streak, left)
                    emit("s3_round_aborted", {"round": rnd, "streak": streak, "left": left})
                    pending = []
                    break

        # §4.3：回報本輪所有檔名（含 PUT 失敗的 —— 伺服器會實際到 S3 確認，傳到了就算數）
        confirmed = do_complete(todo)
        stored |= confirmed & set(todo)
        todo = sorted(n for n in todo if n not in confirmed)
        lg.info("回報完成（第 %d 輪）：已確認 %d/%d 檔，缺少 %d 檔", rnd, len(stored), total, len(todo))
        emit("s3_round_done", {"round": rnd, "stored": len(stored), "missing": len(todo), "total": total})
        if not todo:
            break
        if rnd < st.rounds:
            delay = backoff * rnd
            shown = "、".join(todo[:5]) + (f" …等 {len(todo)} 檔" if len(todo) > 5 else "")
            lg.warning("缺少：%s；%.0f 秒後第 %d 輪只對這些檔案重新取網址", shown, delay, rnd + 1)
            sleep(delay)

    for n in todo:
        failures.setdefault(n, "回報完成時伺服器仍找不到這個檔案")
    return S3UploadResult(
        stored=sorted(stored), missing=todo, rounds=rounds_run, presign_calls=counts["presign"],
        put_ok=counts["ok"], put_failed=counts["failed"], bytes_put=counts["bytes"],
        failures={n: failures[n] for n in sorted(failures)},
    )


def upload_media_s3(
    client: Client,
    store: str,
    day: str,
    media_dir: Path | None = None,
    *,
    files: Sequence[Path] | None = None,
    max_file_bytes: int | None = None,
    settings: S3Settings | None = None,
) -> S3UploadResult:
    """v1.2 每日媒體上傳：直傳 S3。所有輪數跑完仍有缺少就拋 MediaIncompleteError。"""
    if files is None:
        if media_dir is None:
            raise ConfigError("upload_media_s3 需要 media_dir 或 files")
        limit = env_max_file_bytes(default=S3_MAX_FILE_BYTES) if max_file_bytes is None else max_file_bytes
        files = scan_media(media_dir, day, limit)
    if not files:
        log.warning("%s 在 %s 底下沒有任何媒體檔", day, media_dir)
        return S3UploadResult(stored=[], missing=[], rounds=0)

    st = settings or S3Settings.resolve()
    total = sum(p.stat().st_size for p in files)
    log.info("待上傳媒體（v1.2 直傳 S3）：%d 檔 / %s，每次取網址 ≤ %d 檔，最多 %d 輪",
             len(files), human(total), st.presign_batch, st.rounds)
    result = run_s3_rounds(
        files,
        presign=lambda names: client.post_json("/api/ingest/media/presign", media_payload(store, day, names)),
        put=lambda path, u: s3_put(client.s3, path, u, timeout=(CONNECT_TIMEOUT, st.put_timeout)),
        complete=lambda names: client.post_json("/api/ingest/media/complete", media_payload(store, day, names)),
        settings=st,
    )
    if result.missing:
        raise MediaIncompleteError(
            f"{result.rounds} 輪後仍有 {len(result.missing)} 檔不在 S3：{', '.join(result.missing[:10])}"
            + (" …" if len(result.missing) > 10 else ""),
            result.stored, result.missing, result.rounds,
        )
    log.info("媒體完成（v1.2）：%d 檔全部在 S3，%d 輪、取網址 %d 次", len(result.stored), result.rounds,
             result.presign_calls)
    return result


# ── 媒體 v1.1：經 API 伺服器 multipart 轉存（Deprecated）──────────────────
def make_batches(files: Sequence[Path], max_files: int, max_bytes: int) -> Iterator[list[Path]]:
    """v1.1（Deprecated）用的分批。"""
    batch: list[Path] = []
    size = 0
    for p in files:
        n = p.stat().st_size
        if batch and (len(batch) >= max_files or size + n > max_bytes):
            yield batch
            batch, size = [], 0
        batch.append(p)
        size += n
    if batch:
        yield batch


def upload_media_batch(client: Client, store: str, day: str, batch: Sequence[Path]) -> int:
    """.. deprecated:: v1.2  請改用 upload_media_s3()（直傳 S3）。POST /api/ingest/media 正式上線後停用。"""
    warn_deprecated("upload_media_batch()／POST /api/ingest/media")
    return _upload_media_batch_v11(client, store, day, batch)


def _upload_media_batch_v11(client: Client, store: str, day: str, batch: Sequence[Path]) -> int:
    total = sum(p.stat().st_size for p in batch)
    log.info("上傳媒體批次（v1.1，Deprecated）：%d 檔 / %s", len(batch), human(total))
    try:
        data = client.post(
            "/api/ingest/media",
            data={"store_code": store, "biz_date": day},
            parts=[Part("files[]", p) for p in batch],
        )
    except TooLargeError as e:
        if len(batch) == 1:
            raise LocalCheckError(f"單一檔案被伺服器拒收（過大）：{batch[0].name} {human(total)} — {e}") from None
        mid = len(batch) // 2
        log.warning("批次過大（%s），拆成 %d + %d 兩批重送", e, mid, len(batch) - mid)
        return _upload_media_batch_v11(client, store, day, batch[:mid]) + _upload_media_batch_v11(
            client, store, day, batch[mid:]
        )

    stored = data.get("stored") or []
    if len(stored) != len(batch):  # v1.1 §4.3 請確認 stored 數量等於本批送出的檔案數
        missing = sorted(set(p.name for p in batch) - set(stored))
        raise VerifyError(f"本批送出 {len(batch)} 檔，伺服器只回報 {len(stored)} 檔；缺：{', '.join(missing)}")
    log.info("批次成功：已存 %d 檔", len(stored))
    return len(stored)


def upload_media_v11(
    client: Client,
    store: str,
    day: str,
    media_dir: Path,
    max_files: int = BATCH_MAX_FILES,
    max_bytes: int = BATCH_MAX_BYTES,
    files: Sequence[Path] | None = None,
    max_file_bytes: int | None = None,
) -> int:
    """.. deprecated:: v1.2  經 API 伺服器以 multipart 分批轉存。請改用 upload_media_s3()。"""
    warn_deprecated("upload_media_v11()／POST /api/ingest/media")
    if files is None:
        limit = env_max_file_bytes(default=MEDIA_MAX_FILE_BYTES) if max_file_bytes is None else max_file_bytes
        files = scan_media(media_dir, day, limit)
    if not files:
        log.warning("%s 在 %s 底下沒有任何媒體檔", day, media_dir)
        return 0
    total = sum(p.stat().st_size for p in files)
    log.info("待上傳媒體（v1.1，Deprecated）：%d 檔 / %s", len(files), human(total))

    stored = 0
    for i, batch in enumerate(make_batches(files, max_files, max_bytes), start=1):
        log.info("── 第 %d 批 ──", i)
        stored += _upload_media_batch_v11(client, store, day, batch)
    log.info("媒體完成：共 %d 檔 / %s", stored, human(total))
    return stored


# ── 媒體：依上傳方式分派 ──────────────────────────────────────────────────
def upload_media(
    client: Client,
    store: str,
    day: str,
    media_dir: Path,
    max_files: int = BATCH_MAX_FILES,
    max_bytes: int = BATCH_MAX_BYTES,
    files: Sequence[Path] | None = None,
    max_file_bytes: int | None = None,
    *,
    media_api: str | None = None,
    s3: S3Settings | None = None,
) -> int:
    """上傳一天的媒體，回傳已存的檔案數。預設 v1.2 直傳 S3；media_api="v1.1" 走 Deprecated 的舊方式。

    max_files / max_bytes 只對 v1.1 有意義（v1.2 一支檔案一個 PUT，沒有分批的概念）。
    """
    if resolve_media_api(media_api) == MEDIA_API_V11:
        return upload_media_v11(client, store, day, media_dir, max_files, max_bytes, files, max_file_bytes)
    return len(upload_media_s3(client, store, day, media_dir, files=files,
                               max_file_bytes=max_file_bytes, settings=s3).stored)


# ── 設備截圖（§5）──────────────────────────────────────────────────────────
def upload_screenshot(client: Client, store: str, image: Path, captured_at: str | None = None) -> dict:
    if not STORE_CODE_RE.match(store):
        raise LocalCheckError(f"店代號格式不符：{store}")
    if not image.is_file():
        raise LocalCheckError(f"找不到圖檔：{image}")
    if image.suffix.lower() not in SHOT_EXTS:
        raise LocalCheckError(f"截圖只接受 .jpg / .jpeg / .png，實際為 {image.suffix}")
    size = image.stat().st_size
    if size > SHOT_MAX_BYTES:
        raise LocalCheckError(f"截圖超過 10 MB 上限：{human(size)}")

    if captured_at is None:
        # §5.1 建議帶實際拍攝時間；檔案修改時間是最貼近的自動來源，補傳時才不會記成上傳時間
        captured_at = datetime.fromtimestamp(image.stat().st_mtime, JST).strftime("%Y-%m-%dT%H:%M:%S")
        log.info("captured_at 未指定，改用檔案修改時間（日本時間）：%s", captured_at)
    if not JST_TIME_RE.match(captured_at):
        raise LocalCheckError(f"captured_at 必須是 YYYY-MM-DDTHH:mm:ss（日本時間）：{captured_at}")

    # §5.3 這支不是冪等的：每次 200 都會新增一張。逾時／5xx 時重送可能留下兩張重複截圖。
    data = client.post(
        "/api/ingest/screenshot",
        data={"store_code": store, "captured_at": captured_at},
        parts=[Part("file", image)],
    )
    log.info("截圖上傳成功：id=%s captured_at=%s(UTC)", data.get("id"), data.get("captured_at"))
    return data


# ── 每日流程（§1：CSV 成功才送影片）────────────────────────────────────────
def upload_day(
    client: Client,
    csv_path: Path,
    media_dir: Path,
    max_files: int = BATCH_MAX_FILES,
    max_bytes: int = BATCH_MAX_BYTES,
    verify: bool = True,
    max_file_bytes: int | None = None,
    *,
    media_api: str | None = None,
    s3: S3Settings | None = None,
    log_path: Path | None = None,
    log_max_bytes: int | None = None,
) -> None:
    started = time.monotonic()
    api = resolve_media_api(media_api)
    inspected = inspect_csv(csv_path)
    day, store, _ = inspected
    limit = env_max_file_bytes(default=default_max_file_bytes(api)) if max_file_bytes is None else max_file_bytes
    files = scan_media(media_dir, day, limit)  # 先驗證媒體，免得 CSV 進去了影片卻整個請求被拒

    upload_csv(client, csv_path, verify=verify, inspected=inspected, log_path=log_path, log_max_bytes=log_max_bytes)
    stored = upload_media(client, store, day, media_dir, max_files, max_bytes, files=files,
                          media_api=api, s3=s3)
    log.info("每日上傳完成：店=%s 日期=%s 影片/截圖 %d 檔（%s）耗時 %.0f 秒",
             store, day, stored, api, time.monotonic() - started)


def dry_run(csv_path: Path, media_dir: Path, max_files: int, max_bytes: int,
            max_file_bytes: int | None = None, *, media_api: str | None = None,
            s3: S3Settings | None = None, log_path: Path | None = None,
            log_max_bytes: int | None = None) -> None:
    api = resolve_media_api(media_api)
    day, store, rows = inspect_csv(csv_path)
    if log_path is not None:
        log.info("[dry-run] 附上 log 檔：%s（%s）", log_path.name, human(inspect_log(log_path, day, log_max_bytes)))
    limit = env_max_file_bytes(default=default_max_file_bytes(api)) if max_file_bytes is None else max_file_bytes
    files = scan_media(media_dir, day, limit)
    total = sum(p.stat().st_size for p in files)
    log.info("[dry-run] 店=%s 日期=%s CSV %d 列；媒體 %d 檔 / %s（單檔上限 %s，上傳方式 %s）",
             store, day, rows, len(files), human(total), human(limit), api)
    if api == MEDIA_API_V11:
        log.warning("[dry-run] v1.1（POST /api/ingest/media）已 Deprecated，正式上線後停用")
        for i, batch in enumerate(make_batches(files, max_files, max_bytes), start=1):
            log.info("[dry-run] 第 %d 批：%d 檔 / %s", i, len(batch), human(sum(p.stat().st_size for p in batch)))
    else:
        st = s3 or S3Settings.resolve()
        n = (len(files) + st.presign_batch - 1) // st.presign_batch if files else 0
        log.info("[dry-run] v1.2 直傳 S3：取網址 %d 次（每次 ≤ %d 檔），每支一個 PUT，缺少的最多重來 %d 輪",
                 n, st.presign_batch, st.rounds)
    log.info("[dry-run] 本機檢查全部通過，未送出任何請求")


# ── CLI ────────────────────────────────────────────────────────────────────
# ── API 金鑰：環境變數 → 設定檔 ────────────────────────────────────────────
API_KEY_ENV = "YOSHINOYA_API_KEY"
ENV_FILE_DEFAULT = "/etc/yoshinoya_ai/cloud_upload.env"  # 雲端上傳的機密設定（不進版控）


def read_env_file(path: str | Path) -> dict[str, str]:
    """讀 KEY=VALUE 格式的設定檔（例：cloud_upload.env）。

    空行與 # 開頭的行略過；可有 export 前綴；= 兩側與值前後的空白、成對的引號都會去掉；
    沒加引號的值遇到「 #」視為行尾註解。
    """
    out: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for n, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("export "):
                s = s[len("export "):].lstrip()
            if "=" not in s:
                log.warning("設定檔 %s 第 %d 行不是 KEY=VALUE 格式，已略過", path, n)
                continue
            k, v = (x.strip() for x in s.split("=", 1))
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            elif " #" in v:
                v = v.split(" #", 1)[0].rstrip()
            out[k] = v
    return out


def load_api_key(env_file: str | Path | None = None, explicit: str | None = None) -> tuple[str, str]:
    """載入 API 金鑰，回傳 (金鑰, 來源說明)。找不到就拋 ConfigError。

    順序：明確給的 explicit → 環境變數 YOSHINOYA_API_KEY → 設定檔 env_file（預設 /etc/yoshinoya_ai/cloud_upload.env）。
    每次呼叫都重新讀，所以設定檔換了金鑰，下次載入就會生效。來源說明不含金鑰本身，可以放心寫進 log。
    """
    if explicit and explicit.strip():
        return explicit.strip(), "參數"
    value = (os.environ.get(API_KEY_ENV) or "").strip()
    if value:
        return value, f"環境變數 {API_KEY_ENV}"
    path = Path(env_file or ENV_FILE_DEFAULT)
    if not path.is_file():
        raise ConfigError(
            f"找不到 API 金鑰：環境變數 {API_KEY_ENV} 未設定，設定檔 {path} 也不存在。"
            f"金鑰由系統管理者提供，請勿寫進程式碼或版控。"
        )
    try:
        value = (read_env_file(path).get(API_KEY_ENV) or "").strip()
    except OSError as e:  # 例如權限不足：/etc 下的機密檔通常只有特定帳號能讀
        raise ConfigError(f"無法讀取設定檔 {path}：{e}") from None
    if not value:
        raise ConfigError(f"設定檔 {path} 裡沒有 {API_KEY_ENV}（或值是空的）")
    return value, f"設定檔 {path}"


CA_ENV = "YOSHINOYA_CA"


def ca_source(explicit: str | None, env_file: str | Path | None = None) -> tuple[str, str]:
    """決定 API 伺服器的憑證路徑，回傳 (路徑, 來源說明)；檔案不存在就拋 ConfigError。

    順序：明確給的 → 環境變數 YOSHINOYA_CA → 設定檔 env_file 裡的 YOSHINOYA_CA
    （預設 /etc/yoshinoya_ai/cloud_upload.env）→ 本檔旁邊的 prod-57.181.173.247.crt。
    設定檔裡寫相對路徑時，以設定檔所在的資料夾為基準。設定檔不存在或讀不到就跳過這一層。
    """
    ca, source = explicit, "參數"
    if not ca:
        ca, source = (os.environ.get(CA_ENV) or "").strip(), f"環境變數 {CA_ENV}"
    if not ca:
        path = Path(env_file or ENV_FILE_DEFAULT)
        try:
            value = read_env_file(path).get(CA_ENV, "").strip() if path.is_file() else ""
        except OSError as e:
            log.warning("讀不到設定檔 %s，憑證改用預設位置：%s", path, e)
            value = ""
        if value:
            p = Path(value)
            ca, source = str(p if p.is_absolute() or p.root else path.parent / p), f"設定檔 {path}"
    if not ca:
        ca, source = str(Path(__file__).with_name(CA_FILE_DEFAULT)), "預設位置（本檔旁邊）"
    if not Path(ca).is_file():
        raise ConfigError(
            f"找不到伺服器憑證：{ca}（來源：{source}）\n"
            f"伺服器使用自簽憑證，必須以 {CA_FILE_DEFAULT} 驗證。請向系統管理者索取後放在本檔旁邊，"
            f"或用環境變數 {CA_ENV}、設定檔 {Path(env_file or ENV_FILE_DEFAULT)} 的 {CA_ENV} 指定路徑。\n"
            f"（規格明文禁止關閉憑證驗證：那會讓上傳內容與 API 金鑰暴露於中間人攻擊。）"
        )
    return ca, source


def resolve_ca(explicit: str | None, env_file: str | Path | None = None) -> str:
    """同 ca_source()，只回傳路徑。"""
    return ca_source(explicit, env_file)[0]


def make_client(args: argparse.Namespace) -> Client:
    key, source = load_api_key(getattr(args, "env_file", None))
    log.info("API 金鑰來源：%s", source)
    ca, ca_from = ca_source(args.ca, getattr(args, "env_file", None))
    log.info("憑證：%s（來源：%s）", ca, ca_from)
    base = args.url or os.environ.get("YOSHINOYA_URL") or BASE_URL_DEFAULT
    return Client(base, ca, key)


def s3_settings_from_args(args: argparse.Namespace) -> S3Settings:
    return S3Settings.resolve(
        presign_batch=getattr(args, "presign_batch", None),
        rounds=getattr(args, "media_rounds", None),
        put_timeout=getattr(args, "put_timeout", None),
        refresh_margin=getattr(args, "presign_refresh", None),
        abort_after=getattr(args, "put_abort_after", None),
        s3_url=getattr(args, "s3_url", None),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aibox_upload.py",
        description="AI Box → 吉野家 Dashboard 上傳工具（API v1.2）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--url", help=f"Base URL（預設 {BASE_URL_DEFAULT}）")
    p.add_argument("--ca", help=f"伺服器憑證路徑（預設 ./{CA_FILE_DEFAULT}）")
    p.add_argument("--s3-url", help=f"health 探測用的 S3 位址（預設讀 {S3_URL_ENV}，再退回 {S3_URL_DEFAULT}）")
    p.add_argument("--env-file", help=f"環境變數 {API_KEY_ENV} 沒設定時，讀取 API 金鑰的設定檔（預設 {ENV_FILE_DEFAULT}）")
    p.add_argument("-v", "--verbose", action="store_true", help="輸出除錯訊息")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_api_opt(sp):
        sp.add_argument("--media-api", choices=[MEDIA_API_V12, MEDIA_API_V11],
                        help=f"媒體上傳方式：v1.2 直傳 S3（預設）或 v1.1（Deprecated）；預設讀 {MEDIA_API_ENV}")

    def add_media_opts(sp):
        add_api_opt(sp)
        sp.add_argument("--max-file-mb", type=int,
                        help=f"單檔上限 MB（預設讀 {MAX_FILE_MB_ENV}，再退回規格值：v1.2 = "
                             f"{S3_MAX_FILE_BYTES // 1024 // 1024}、v1.1 = {MEDIA_MAX_FILE_BYTES // 1024 // 1024}）")
        g = sp.add_argument_group("v1.2 直傳 S3")
        g.add_argument("--presign-batch", type=int,
                       help=f"每次取網址／回報完成的檔案數（預設讀 {PRESIGN_BATCH_ENV}，再退回 {PRESIGN_MAX_FILES}）")
        g.add_argument("--media-rounds", type=int,
                       help=f"缺少的檔案最多重來幾輪（預設讀 {MEDIA_ROUNDS_ENV}，再退回 {MEDIA_ROUNDS}）")
        g.add_argument("--put-timeout", type=float,
                       help=f"單支 PUT 的讀取逾時秒數（預設讀 {PUT_TIMEOUT_ENV}，再退回 {PUT_TIMEOUT}）")
        g.add_argument("--presign-refresh", type=float,
                       help=f"網址剩不到幾秒有效就重新取（預設讀 {PRESIGN_REFRESH_ENV}，再退回 {PRESIGN_REFRESH}）")
        g.add_argument("--put-abort-after", type=int,
                       help=f"連續幾支因連線問題失敗就提前結束本輪，0 = 不提前（預設讀 {PUT_ABORT_AFTER_ENV}，"
                            f"再退回 {PUT_ABORT_AFTER}）")
        g11 = sp.add_argument_group("v1.1（Deprecated）")
        g11.add_argument("--batch-files", type=int, default=BATCH_MAX_FILES,
                         help=f"每批檔案數上限（預設 {BATCH_MAX_FILES}）")
        g11.add_argument("--batch-mb", type=int, default=BATCH_MAX_BYTES // 1024 // 1024,
                         help="每批總大小上限 MB（預設 200）")

    def add_log_opts(sp):
        g = sp.add_argument_group("v1.2.4 log 檔（選填）")
        g.add_argument("--log", type=Path, dest="log_path",
                       help="隨 CSV 附上的當日 AI Box log 檔，檔名必須是 YYYYMMDD.log、日期與 CSV 相同；"
                            "不給就只傳 CSV（與之前相同）")
        g.add_argument("--log-max-mb", type=int,
                       help=f"log 檔上限 MB（預設讀 {LOG_MAX_MB_ENV}，再退回規格的 {LOG_MAX_MB}）")

    h = sub.add_parser("health", help="連線與憑證測試（不需金鑰）；v1.2 另外探測 S3")
    add_api_opt(h)

    d = sub.add_parser("day", help="每日上傳：先 CSV，成功後送媒體")
    d.add_argument("csv_path", type=Path)
    d.add_argument("media_dir", type=Path)
    d.add_argument("--dry-run", action="store_true", help="只做本機檢查與上傳規劃，不連線")
    d.add_argument("--no-verify", action="store_true", help="不核對伺服器回報的列數")
    add_media_opts(d)
    add_log_opts(d)

    c = sub.add_parser("csv", help="只上傳 CSV")
    c.add_argument("csv_path", type=Path)
    c.add_argument("--no-verify", action="store_true", help="不核對伺服器回報的列數")
    add_log_opts(c)

    m = sub.add_parser("media", help="只上傳媒體")
    m.add_argument("store_code")
    m.add_argument("biz_date", help="YYYYMMDD")
    m.add_argument("media_dir", type=Path)
    add_media_opts(m)

    s = sub.add_parser("shot", help="上傳設備截圖（非冪等，成功後不要重送）")
    s.add_argument("store_code")
    s.add_argument("image_path", type=Path)
    s.add_argument("--captured-at", help="拍攝時間，日本時間 YYYY-MM-DDTHH:mm:ss（預設取檔案修改時間）")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

    try:
        api = resolve_media_api(getattr(args, "media_api", None))
        s3 = s3_settings_from_args(args)
        max_file = None
        if args.cmd in ("day", "media"):
            max_file = env_max_file_bytes(args.max_file_mb, default=default_max_file_bytes(api))
        log_path = getattr(args, "log_path", None)
        log_max = env_log_max_bytes(args.log_max_mb) if log_path is not None else None

        if args.cmd == "day" and args.dry_run:
            dry_run(args.csv_path, args.media_dir, args.batch_files, args.batch_mb * 1024 * 1024,
                    max_file, media_api=api, s3=s3, log_path=log_path, log_max_bytes=log_max)
            return 0

        if args.cmd == "health":
            ca, ca_from = ca_source(args.ca, args.env_file)
            log.info("憑證：%s（來源：%s）", ca, ca_from)
            client = Client(args.url or os.environ.get("YOSHINOYA_URL") or BASE_URL_DEFAULT, ca, api_key="")
            log.info("healthz：%s", client.healthz())
            if api == MEDIA_API_V11:
                log.warning("媒體上傳方式為 v1.1（Deprecated），略過 S3 探測")
                return 0
            res = probe_s3(s3.s3_url, session=client.s3)
            if not res["ok"]:
                log.error("S3 連不到：%s —— %s。v1.2 起門市網路需能連出 TCP 443 到 S3（§2.1）",
                          res["url"], res["error"])
                return 1
            log.info("S3 連得到：%s（HTTP %s，403 也算正常）", res["url"], res["status"])
            return 0

        client = make_client(args)
        if args.cmd == "day":
            upload_day(
                client,
                args.csv_path,
                args.media_dir,
                args.batch_files,
                args.batch_mb * 1024 * 1024,
                verify=not args.no_verify,
                max_file_bytes=max_file,
                media_api=api,
                s3=s3,
                log_path=log_path,
                log_max_bytes=log_max,
            )
        elif args.cmd == "csv":
            upload_csv(client, args.csv_path, verify=not args.no_verify, log_path=log_path, log_max_bytes=log_max)
        elif args.cmd == "media":
            if not STORE_CODE_RE.match(args.store_code):
                raise LocalCheckError(f"店代號格式不符：{args.store_code}")
            if not BIZ_DATE_RE.match(args.biz_date):
                raise LocalCheckError(f"biz_date 必須是 YYYYMMDD：{args.biz_date}")
            upload_media(
                client,
                args.store_code,
                args.biz_date,
                args.media_dir,
                args.batch_files,
                args.batch_mb * 1024 * 1024,
                max_file_bytes=max_file,
                media_api=api,
                s3=s3,
            )
        elif args.cmd == "shot":
            upload_screenshot(client, args.store_code, args.image_path, args.captured_at)
        return 0

    except ConfigError as e:
        log.error("設定錯誤：%s", e)
        return 2
    except LocalCheckError as e:
        log.error("本機檢查未通過：%s", e)
        return 1
    except ApiError as e:
        # §6：401 / 400 / 413 都不重試，請依 error 修正後再跑
        log.error("伺服器拒絕請求（不重試）：%s", e)
        return 1
    except TransientError as e:
        log.error("%s。請於下一個排程時段整天重跑。", e)
        return 1
    except MediaIncompleteError as e:
        log.error("媒體未完成：%s。請於下一個排程時段整天重跑（CSV 與媒體重送都是覆蓋）。", e)
        return 1
    except VerifyError as e:
        log.error("上傳結果核對失敗：%s", e)
        return 1
    except KeyboardInterrupt:
        log.error("使用者中斷")
        return 1


if __name__ == "__main__":
    sys.exit(main())
