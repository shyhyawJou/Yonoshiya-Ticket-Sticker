#!/usr/bin/env python3
"""aibox_uploader —— AI Box 每日上傳程式（給 crontab 一天執行一次）。

依 AIBOX_UPLOAD_v1.2.8.md 規格上傳一家店、一個營業日的檔案：CSV 與 log 經 API，影片與高清影像直傳 S3。
要上傳哪些副檔名、到哪些資料夾找，都寫在設定檔（JSON），改設定檔就好，不必改程式。

用法：
    python3 aibox_uploader.py all   --store 07c6 --prefix-path /data --date 20261012 --config /etc/yoshinoya_ai/aibox_uploader.json
    python3 aibox_uploader.py video --store 07c6 --prefix-path /data --date 20261012
    python3 aibox_uploader.py all   --store 07c6 --prefix-path /data --dry-run      # 只列出會上傳哪些檔案，不連線

上傳方式（第一個參數）：
    all    log → CSV → 影片＋高清影像（CSV 成功才送影片，規格 §1；高清影像與 log 不受 CSV 影響，照常送出）
    csv    只送 CSV（POST /api/ingest/csv）
    log    只送當日 log（POST /api/ingest/log，規格 §6）
    video  只送影片（直傳 S3，規格 §4）
    image  只送高清影像（直傳 S3，規格 §4）

遇到錯誤不中斷：每個檔案各自重試，同一個檔案失敗 max_attempts 次（預設 3，可設定）就暫停這個檔案、
換下一個，直到所有檔案都處理完，最後列出結果。規格說不要重試的錯誤（HTTP 4xx）與檔案本身的問題
（不存在、不是 UTF-8、0 列的 CSV…）第一次就暫停，重試也不會成功。暫停只在這次執行有效，下次執行會再試。

失敗日期的自動補傳：某個營業日有檔案沒傳成功（結束代碼 1），就記進補傳清單
（<prefix>/.aibox_uploader/pending_<店>.json）。之後每次執行，先傳 --date 那天，再把清單裡的日期
由舊到新、用當初的模式再傳一次；影片與高清影像會先問伺服器哪些已在 S3，只傳缺的。
傳成功就從清單移除；同一天試了 catch_up.max_runs 次（預設 3，含第一次）、或已超過 catch_up.max_days 天
（預設 7）就放棄並留 ERROR，請人工處理。日期資料夾不存在不列入補傳。--no-catch-up 可以這次不補傳。

參數的優先順序：命令列 > 設定檔（--config）> 環境變數 > 程式內建預設值。
--date 是日期資料夾的名稱（YYYYMMDD），不給就用昨天（本機時間）；也可以寫 today / yesterday。

API 金鑰：環境變數 YOSHINOYA_API_KEY → 設定檔 env_file（預設 /etc/yoshinoya_ai/cloud_upload.env）。
憑證：設定檔 ca_file → 環境變數 YOSHINOYA_CA → env_file 裡的 YOSHINOYA_CA → 程式旁邊的 prod-57.181.173.247.crt。

結束代碼：0 = 全部成功（或當天沒有東西要傳）；1 = 有檔案沒上傳成功；2 = 參數或設定錯誤（什麼都沒傳）。
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aibox_upload as ab  # noqa: E402

log = logging.getLogger("aibox_uploader")

MODES = ("all", "csv", "log", "video", "image")
KINDS = ("csv", "log", "video", "image")
KIND_LABEL = {"csv": "CSV", "log": "log", "video": "影片", "image": "高清影像"}
# 每一類能設定的副檔名受 API 限制：CSV / log 的檔名格式是規格固定的；媒體只收這四種（§4.4），
# 多放一種伺服器會把整批網址請求都拒絕（HTTP 400），所以在啟動時就擋下
KIND_EXTS = {"csv": {".csv"}, "log": {".log"}, "video": set(ab.MEDIA_EXTS), "image": set(ab.MEDIA_EXTS)}

MAX_ATTEMPTS = 3                       # 同一個檔案最多試幾次
MAX_ATTEMPTS_ENV = "YOSHINOYA_MAX_ATTEMPTS"
RETRY_BACKOFF = 10.0                   # 重試間隔（秒），第 n 次重試前等 n 倍
RETRY_BACKOFF_ENV = "YOSHINOYA_RETRY_BACKOFF"
CATCH_UP_RUNS = 3                      # 同一個失敗日期最多試幾次（含第一次）
CATCH_UP_RUNS_ENV = "YOSHINOYA_CATCH_UP_RUNS"
CATCH_UP_DAYS = 7                      # 超過幾天前就放棄補傳（門市本機的檔案通常只保留數天）
CATCH_UP_DAYS_ENV = "YOSHINOYA_CATCH_UP_DAYS"

DEFAULT_CONFIG: dict[str, Any] = {
    "store": None,            # 店代號；命令列 --store 優先
    "prefix_path": None,      # 上傳檔案的根目錄；命令列 --prefix-path 優先
    "env_file": ab.ENV_FILE_DEFAULT,
    "base_url": None,         # None → 環境變數 YOSHINOYA_URL → 規格預設值
    "ca_file": None,          # None → 環境變數 YOSHINOYA_CA → env_file 的 YOSHINOYA_CA → 程式旁邊
    "media_api": "v1.2",      # v1.2 = 直傳 S3；v1.1 = 經 API 轉存（Deprecated）
    "max_file_mb": None,      # 媒體單檔上限；None → YOSHINOYA_MAX_FILE_MB → 規格值（v1.2 = 5120）
    "log_max_mb": None,       # log 檔上限；None → YOSHINOYA_LOG_MAX_MB → 規格的 20
    "retry": {                # 每個檔案的重試；None → 對應的環境變數 → 預設值
        "max_attempts": None,  # 同一個檔案失敗幾次就暫停（YOSHINOYA_MAX_ATTEMPTS，預設 3）
        "backoff": None,       # 重試間隔秒數，第 n 次重試前等 n 倍（YOSHINOYA_RETRY_BACKOFF，預設 10）
    },
    "s3": {                   # 直傳 S3 的參數；None → 對應的 YOSHINOYA_* 環境變數 → 規格預設值
        "presign_batch": None,
        "put_timeout": None,
        "presign_refresh": None,
    },
    "catch_up": {             # 失敗日期的自動補傳
        "enabled": True,
        "max_runs": None,      # 同一天最多試幾次（含第一次）；None → YOSHINOYA_CATCH_UP_RUNS → 3
        "max_days": None,      # 超過幾天前的日期就放棄；None → YOSHINOYA_CATCH_UP_DAYS → 7
        "state_dir": ".aibox_uploader",  # 補傳清單的資料夾；相對路徑接在 <prefix> 底下
        "skip_existing": True,  # 補傳時先問伺服器哪些媒體已在 S3，只傳缺的
    },
    "run_log": {              # 這支程式自己的 log 檔
        "enabled": True,
        "dir": "csv_logs",                       # 相對路徑接在 <prefix>/<date>/ 底下
        "name": "{date}_{store}_upload.log",
    },
    "targets": {              # 每一類要到哪些資料夾找、上傳哪些副檔名
        "csv": {"dirs": ["csv_logs"], "exts": [".csv"]},
        "log": {"dirs": ["stream_log"], "exts": [".log"]},
        "video": {"dirs": ["event"], "exts": [".mp4"]},
        "image": {"dirs": ["full_screen"], "exts": [".jpg", ".jpeg", ".png"]},
    },
}


# ── 設定檔 ──────────────────────────────────────────────────────────────────
def _merge(base: dict, over: dict, where: str = "") -> None:
    """把設定檔的值蓋到預設值上。以 _ 開頭的欄位當成註解；不認識的欄位留 warning（多半是拼錯）。"""
    for k, v in over.items():
        if str(k).startswith("_"):
            continue
        if k not in base:
            log.warning("設定檔有不認識的欄位「%s%s」，已忽略（請確認是否拼錯）", where, k)
            continue
        if isinstance(base[k], dict) and isinstance(v, dict):
            _merge(base[k], v, f"{where}{k}.")
        else:
            base[k] = v


def load_config(path: str | None) -> tuple[dict, str]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    source = "未指定設定檔，使用內建預設值"
    if path:
        p = Path(path)
        if not p.is_file():
            raise ab.ConfigError(f"找不到設定檔：{p}")
        try:
            user = json.loads(p.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise ab.ConfigError(f"設定檔 {p} 不是合法的 JSON：第 {e.lineno} 行第 {e.colno} 欄，{e.msg}") from None
        except (OSError, UnicodeDecodeError) as e:
            raise ab.ConfigError(f"無法讀取設定檔 {p}：{e}") from None
        if not isinstance(user, dict):
            raise ab.ConfigError(f"設定檔 {p} 的最外層必須是 JSON 物件（{{ … }}）")
        _merge(cfg, user)
        source = str(p)
    _validate_targets(cfg)
    return cfg, source


def _validate_targets(cfg: dict) -> None:
    for kind in KINDS:
        t = cfg["targets"].get(kind)
        if not isinstance(t, dict):
            raise ab.ConfigError(f"設定檔 targets.{kind} 必須是物件，例如 {{\"dirs\": [...], \"exts\": [...]}}")
        dirs = [t.get("dirs")] if isinstance(t.get("dirs"), str) else t.get("dirs")
        if not isinstance(dirs, list) or not dirs or not all(isinstance(d, str) and d.strip() for d in dirs):
            raise ab.ConfigError(f"設定檔 targets.{kind}.dirs 必須是至少一個資料夾名稱的清單")
        raw = [t.get("exts")] if isinstance(t.get("exts"), str) else t.get("exts")
        if not isinstance(raw, list) or not raw or not all(isinstance(e, str) and e.strip() for e in raw):
            raise ab.ConfigError(f"設定檔 targets.{kind}.exts 必須是至少一個副檔名的清單，例如 [\".mp4\"]")
        exts = list(dict.fromkeys("." + e.strip().lower().lstrip(".") for e in raw))
        bad = [e for e in exts if e not in KIND_EXTS[kind]]
        if bad:
            raise ab.ConfigError(
                f"設定檔 targets.{kind}.exts 有 API 不接受的副檔名：{', '.join(bad)}"
                f"（{KIND_LABEL[kind]}只能是 {', '.join(sorted(KIND_EXTS[kind]))}）"
            )
        t["dirs"], t["exts"] = [d.strip() for d in dirs], exts


@dataclass
class Settings:
    """由設定檔與環境變數算出來的值。一開始就全部算好：有錯就在上傳任何東西之前停下，不會傳到一半才發現。"""

    media_api: str
    max_file_bytes: int
    log_max_bytes: int
    s3: ab.S3Settings
    max_attempts: int
    backoff: float
    catch_up: bool = True
    catch_up_runs: int = CATCH_UP_RUNS
    catch_up_days: int = CATCH_UP_DAYS
    catch_up_dir: str = ".aibox_uploader"
    skip_existing: bool = True


def resolve_settings(cfg: dict, args: argparse.Namespace) -> Settings:
    api = ab.resolve_media_api(cfg["media_api"])
    retry, s3, cu = cfg["retry"], cfg["s3"], cfg["catch_up"]
    explicit_attempts = args.max_attempts if args.max_attempts is not None else retry.get("max_attempts")
    return Settings(
        catch_up=bool(cu.get("enabled", True)) and not args.no_catch_up,
        catch_up_runs=ab.env_number(CATCH_UP_RUNS_ENV, cu.get("max_runs"), CATCH_UP_RUNS,
                                    label="catch_up.max_runs", unit=" 次"),
        catch_up_days=ab.env_number(CATCH_UP_DAYS_ENV, cu.get("max_days"), CATCH_UP_DAYS,
                                    label="catch_up.max_days", unit=" 天"),
        catch_up_dir=str(cu.get("state_dir") or ".aibox_uploader"),
        skip_existing=bool(cu.get("skip_existing", True)),
        media_api=api,
        max_file_bytes=ab.env_max_file_bytes(cfg["max_file_mb"], default=ab.default_max_file_bytes(api)),
        log_max_bytes=ab.env_log_max_bytes(cfg["log_max_mb"]),
        s3=ab.S3Settings.resolve(presign_batch=s3.get("presign_batch"), put_timeout=s3.get("put_timeout"),
                                 refresh_margin=s3.get("presign_refresh")),
        max_attempts=ab.env_number(MAX_ATTEMPTS_ENV, explicit_attempts, MAX_ATTEMPTS,
                                   label="max_attempts", unit=" 次"),
        backoff=ab.env_number(RETRY_BACKOFF_ENV, retry.get("backoff"), RETRY_BACKOFF,
                              label="retry.backoff", cast=float, minimum=0, unit=" 秒"),
    )


# ── 執行環境 ────────────────────────────────────────────────────────────────
@dataclass
class Ctx:
    mode: str
    store: str
    day: str
    prefix: Path
    cfg: dict
    st: Settings
    dry_run: bool = False
    catching_up: bool = False  # 這次是在補傳先前失敗的日期
    problems: list[str] = field(default_factory=list)  # 沒上傳的檔案與原因（影響結束代碼）

    @property
    def day_dir(self) -> Path:
        return self.prefix / self.day

    def dirs(self, kind: str) -> list[Path]:
        """targets.<kind>.dirs：相對路徑接在 <prefix>/<date>/ 底下；絕對路徑原樣使用（可用 {date}、{store}）。"""
        out = []
        for d in self.cfg["targets"][kind]["dirs"]:
            p = Path(d.replace("{date}", self.day).replace("{store}", self.store))
            out.append(p if (p.is_absolute() or p.root) else self.day_dir / p)
        return out

    def exts(self, kind: str) -> set[str]:
        return set(self.cfg["targets"][kind]["exts"])


@dataclass
class Result:
    item: str
    status: str  # OK / SKIP / FAIL
    detail: str


def resolve_day(text: str | None) -> str:
    t = (text or "yesterday").strip().lower()
    if t == "yesterday":
        return (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    if t == "today":
        return datetime.now().strftime("%Y%m%d")
    try:
        return datetime.strptime(t, "%Y%m%d").strftime("%Y%m%d")
    except ValueError:
        raise ab.ConfigError(f"--date 必須是 YYYYMMDD（日期資料夾名稱）、today 或 yesterday，實際為 {text!r}") from None


# ── 錯誤分類與重試 ──────────────────────────────────────────────────────────
# 重試也不會成功的錯誤：伺服器明確拒絕（4xx，規格 §7 說不要重試），或檔案本身的問題
PERMANENT = (ab.ApiError, ab.LocalCheckError, FileNotFoundError, IsADirectoryError, NotADirectoryError,
             UnicodeDecodeError, csv.Error)


def describe(e: BaseException) -> str:
    if isinstance(e, UnicodeDecodeError):
        return f"檔案不是 UTF-8 編碼（{e.reason}，位置 {e.start}）—— 請存成 UTF-8"
    if isinstance(e, FileNotFoundError):
        return f"檔案不存在（可能已被移走或刪除）：{e.filename or e}"
    if isinstance(e, (ab.ApiError, ab.LocalCheckError, ab.TransientError, ab.VerifyError)):
        return str(e).replace("\n", " ")
    return f"{type(e).__name__}: {e}".replace("\n", " ")


def try_file(ctx: Ctx, label: str, fn: Callable[[], Any]) -> tuple[bool, Any, str]:
    """對一個檔案執行 fn，失敗就重試，最多 max_attempts 次；PERMANENT 類的錯誤第一次就停。

    任何例外都在這裡接住，不會讓整支程式中斷。回傳 (成功與否, fn 的結果, 最後的錯誤說明)。
    """
    n_max, err = ctx.st.max_attempts, ""
    for n in range(1, n_max + 1):
        try:
            return True, fn(), ""
        except PERMANENT as e:
            err = describe(e)
            log.error("%s：%s（重試也不會成功，暫停這一項，繼續下一個）", label, err)
            return False, None, err
        except Exception as e:  # noqa: BLE001 —— 網路、5xx、核對不符、或預期外的錯誤：都算一次失敗
            err = describe(e)
            if n < n_max:
                delay = ctx.st.backoff * n
                log.warning("%s：第 %d/%d 次失敗 —— %s；%.0f 秒後重試", label, n, n_max, err, delay)
                time.sleep(delay)
            else:
                log.error("%s：失敗 %d 次 —— %s；暫停這一項，繼續下一個", label, n_max, err)
    return False, None, err


# ── 找檔案（資料夾讀不到也不中斷）──────────────────────────────────────────
def _list_dir(ctx: Ctx, kind: str, d: Path) -> list[Path]:
    if not d.is_dir():
        log.warning("找不到%s資料夾：%s —— 略過", KIND_LABEL[kind], d)
        return []
    try:
        return sorted(d.iterdir())
    except OSError as e:
        ctx.problems.append(f"{d}：讀不到資料夾（{describe(e)}）")
        log.error("讀不到%s資料夾 %s：%s —— 略過", KIND_LABEL[kind], d, describe(e))
        return []


def find_named(ctx: Ctx, kind: str, want: str) -> Path | None:
    """CSV / log 的檔名是規格固定的（YYYYMMDD_<店>.csv、YYYYMMDD.log），在設定的資料夾裡依序找第一個。"""
    for d in ctx.dirs(kind):
        for p in _list_dir(ctx, kind, d):
            # 店代號不分大小寫（規格 v1.2.6），20261005_07B2.csv 也算 07b2 的
            if p.suffix.lower() in ctx.exts(kind) and p.name.lower() == want.lower() and p.is_file():
                return p
    return None


def find_media(ctx: Ctx, kinds: list[str]) -> dict[str, list[Path]]:
    """依設定找出各類媒體檔：副檔名符合、檔名以營業日開頭。有問題的檔案收進 ctx.problems，不送出。

    §4.4：一次請求裡只要有一個檔名不合規，整批都會被拒，所以在本機先擋下，其他檔案照常上傳。
    """
    limit = ctx.st.max_file_bytes
    seen: dict[str, Path] = {}
    found: dict[str, list[Path]] = {k: [] for k in kinds}
    for kind in kinds:
        for d in ctx.dirs(kind):
            n = 0
            for p in _list_dir(ctx, kind, d):
                if p.suffix.lower() not in ctx.exts(kind) or not p.name.startswith(ctx.day):
                    continue
                try:
                    if not p.is_file():
                        continue
                    size = p.stat().st_size
                except OSError as e:
                    ctx.problems.append(f"{p}：{describe(e)}")
                    continue
                if p.name in seen:
                    if seen[p.name].resolve() != p.resolve():  # 同一個檔案被兩類設定都選到就算了
                        ctx.problems.append(f"{p}：與 {seen[p.name]} 同名（在伺服器上是同一個位置，會互相覆蓋）")
                    continue
                if not ab.MEDIA_NAME_RE.match(p.name):
                    ctx.problems.append(f"{p}：檔名只能用英數、'.'、'-'、'_'，且不超過 200 字元")
                elif size == 0:
                    ctx.problems.append(f"{p}：空檔")
                elif size > limit:
                    ctx.problems.append(f"{p}：{ab.human(size)}，超過單檔上限 {ab.human(limit)}")
                else:
                    seen[p.name] = p
                    found[kind].append(p)
                    n += 1
            if d.is_dir():
                log.info("掃描%s資料夾 %s：%d 檔（副檔名 %s）", KIND_LABEL[kind], d, n,
                         "、".join(ctx.cfg["targets"][kind]["exts"]))
    return found


# ── 上傳：log、CSV ─────────────────────────────────────────────────────────
def make_client(ctx: Ctx) -> ab.Client:
    cfg = ctx.cfg
    key, key_from = ab.load_api_key(cfg["env_file"])
    log.info("API 金鑰來源：%s", key_from)  # 只記來源，不記金鑰本身
    base = (cfg["base_url"] or os.environ.get("YOSHINOYA_URL") or ab.BASE_URL_DEFAULT).rstrip("/")
    if urlparse(base).scheme == "https":
        ca, ca_from = ab.ca_source(cfg["ca_file"], cfg["env_file"])
        log.info("憑證：%s（來源：%s）", ca, ca_from)
    else:
        log.warning("Base URL 不是 https（%s），跳過憑證驗證 —— 僅限本機測試使用", base)
        ca = True  # type: ignore[assignment]
    return ab.Client(base, ca, key)


def step_log(ctx: Ctx, client: ab.Client | None) -> Result:
    path = find_named(ctx, "log", f"{ctx.day}.log")
    if path is None:
        return Result("log", "SKIP" if ctx.mode == "all" else "FAIL",
                      f"找不到 {ctx.day}.log（找過 {', '.join(str(d) for d in ctx.dirs('log'))}）")
    if ctx.dry_run:
        size = ab.inspect_log(path, ctx.day, ctx.st.log_max_bytes)
        return Result("log", "OK", f"[dry-run] 會上傳 {path}（{ab.human(size)}）")

    def send() -> str:
        size = ab.inspect_log(path, ctx.day, ctx.st.log_max_bytes)  # 每次都重讀：檔案可能還在寫
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        log.info("上傳 log（/api/ingest/log）：%s（%s）", path, ab.human(size))
        data = client.post("/api/ingest/log", data={"store_code": ctx.store},
                           parts=[ab.Part("file", path)], attempts=1)  # 重試由 try_file 計次
        # §6.2：bytes / sha256 是伺服器實際收到的內容
        if data.get("bytes") is not None and data["bytes"] != size:
            raise ab.VerifyError(f"log 大小不符：本機 {size} bytes，伺服器回報 {data['bytes']} bytes")
        if data.get("sha256") and str(data["sha256"]).lower() != digest:
            raise ab.VerifyError(f"log 內容不符：本機 sha256 {digest}，伺服器回報 {data['sha256']}")
        return f"{path.name}（{ab.human(size)}，sha256 {digest[:12]}… 與伺服器一致）"

    ok, detail, err = try_file(ctx, f"log {path.name}", send)
    return Result("log", "OK" if ok else "FAIL", detail if ok else f"{path.name}：{err}")


def step_csv(ctx: Ctx, client: ab.Client | None) -> Result:
    path = find_named(ctx, "csv", f"{ctx.day}_{ctx.store}.csv")
    if path is None:
        return Result("csv", "SKIP" if ctx.mode == "all" else "FAIL",
                      f"找不到 {ctx.day}_{ctx.store}.csv（找過 {', '.join(str(d) for d in ctx.dirs('csv'))}）")

    def check() -> tuple[str, str, int]:
        day, store, rows = ab.inspect_csv(path)  # 空檔、缺必要欄位、不是 UTF-8 都會在這裡擋下
        if day != ctx.day:
            raise ab.LocalCheckError(f"檔名的日期 {day} 與 --date {ctx.day} 不符")
        if rows == 0:
            # 只有標題列的 CSV 會被伺服器當成「這天 0 列」，把 Dashboard 上這家店這天的訂單整批移除
            raise ab.LocalCheckError("沒有任何資料列（0 列）：不上傳，以免伺服器把這天的訂單整批移除")
        return day, store, rows

    def send() -> str:
        day, store, rows = check()  # 每次都重新檢查：檔案可能在重試之間被換掉
        # 不附 log：log 由 step_log 用 /api/ingest/log 獨立送出（v1.2.8）
        data = client.post("/api/ingest/csv", parts=[ab.Part("file", path)], attempts=1)
        for w in data.get("warnings") or []:
            log.warning("CSV 格式警告：%s", w)  # §3.5 該列仍寫入，但壞欄位存成空值
        if data.get("rows") != rows:  # §3.3 請與本機列數核對
            raise ab.VerifyError(f"列數不符：本機 {rows} 列，伺服器回報 {data.get('rows')} 列")
        return (f"{path.name}：{data.get('rows')} 列（新增 {data.get('upserted')}／變動 {data.get('modified')}／"
                f"移除 {data.get('removed')}），warnings {len(data.get('warnings') or [])}")

    if ctx.dry_run:
        ok, rows_info, err = try_file(ctx, f"CSV {path.name}", check)
        return Result("csv", "OK" if ok else "FAIL",
                      f"[dry-run] 會上傳 {path}（{rows_info[2]} 列）" if ok else f"{path.name}：{err}")
    ok, detail, err = try_file(ctx, f"CSV {path.name}", send)
    return Result("csv", "OK" if ok else "FAIL", detail if ok else f"{path.name}：{err}")


# ── 上傳：媒體（逐檔計次）──────────────────────────────────────────────────
@dataclass
class FileState:
    path: Path
    attempts: int = 0       # 失敗次數
    done: bool = False      # 伺服器 complete 已確認在 S3
    paused: bool = False    # 失敗達上限或無法上傳，這次執行不再試
    error: str = ""


def _fail(ctx: Ctx, s: FileState, err: str, permanent: bool = False) -> None:
    s.attempts += 1
    s.error = err
    if permanent or s.attempts >= ctx.st.max_attempts:
        s.paused = True
        why = "重試也不會成功" if permanent else f"失敗 {s.attempts} 次"
        log.error("%s：%s —— %s；暫停這個檔案的上傳，繼續下一個", s.path.name, err, why)
    else:
        log.warning("%s：第 %d/%d 次失敗 —— %s（稍後重試）", s.path.name, s.attempts, ctx.st.max_attempts, err)


def upload_media_files(ctx: Ctx, client: ab.Client, files: list[Path]) -> list[FileState]:
    """v1.2 §4：取網址 → 逐支 PUT 到 S3 → 回報完成。每個檔案各自計算失敗次數，任何一支出錯都不影響其他支。

    一輪裡每個還沒完成的檔案都試一次；PUT 失敗、或回報完成時伺服器找不到，都算該檔失敗一次。
    取網址或回報完成這兩支 API 本身失敗時，這一輪涉及的檔案各算失敗一次。
    """
    st = ctx.st
    states = [FileState(p) for p in files]
    payload = lambda names: ab.media_payload(ctx.store, ctx.day, names)  # noqa: E731
    if ctx.catching_up and st.skip_existing:
        # 補傳：上次可能已經傳了一部分，先問伺服器哪些已在 S3，只傳缺的
        stored = _already_stored(ctx, client, [s.path.name for s in states], payload)
        for s in states:
            s.done = s.path.name in stored
        skipped = sum(s.done for s in states)
        if skipped:
            log.info("補傳：%d 檔已在 S3，略過；%d 檔需要上傳", skipped, len(states) - skipped)
    rnd = 0
    while True:
        pending = [s for s in states if not s.done and not s.paused]
        if not pending:
            break
        rnd += 1
        if rnd > 1:
            delay = st.backoff * (rnd - 1)
            log.info("── 第 %d 輪：%d 檔待重試，%.0f 秒後開始 ──", rnd, len(pending), delay)
            time.sleep(delay)
        else:
            log.info("── 第 1 輪：%d 檔 ──", len(pending))
        put_ok: set[str] = set()
        for i in range(0, len(pending), st.s3.presign_batch):
            chunk = pending[i:i + st.s3.presign_batch]
            uploads, expires, issued = _presign(ctx, client, chunk, payload)
            if uploads is None:
                continue  # 取網址失敗：這批各算失敗一次，下一輪再來
            for j, s in enumerate(chunk):
                if s.paused:
                    continue
                if not s.path.is_file():
                    _fail(ctx, s, "檔案已不存在（可能已被移走或刪除）", permanent=True)
                    continue
                margin = min(st.s3.refresh_margin, expires / 2)
                if time.monotonic() - issued >= expires - margin:  # 網址快過期：剩下的重新取
                    log.info("上傳網址快過期，重新取得剩下 %d 檔的網址", len(chunk) - j)
                    uploads, expires, issued = _presign(ctx, client, chunk[j:], payload)
                    if uploads is None:
                        break
                u = uploads.get(s.path.name)
                if not u or not u.get("url"):
                    _fail(ctx, s, "取得上傳網址的回應裡沒有這個檔案")
                    continue
                t0 = time.monotonic()
                try:
                    size = s.path.stat().st_size
                    out = ab.s3_put(client.s3, s.path, u, timeout=(ab.CONNECT_TIMEOUT, st.s3.put_timeout))
                except OSError as e:  # 檔案讀到一半被刪除、權限不足
                    _fail(ctx, s, describe(e), permanent=isinstance(e, FileNotFoundError))
                    continue
                except Exception as e:  # noqa: BLE001
                    _fail(ctx, s, describe(e))
                    continue
                if out.ok:
                    put_ok.add(s.path.name)
                    log.info("S3 上傳：%s（%s，%.1f 秒）", s.path.name, ab.human(size), time.monotonic() - t0)
                else:
                    _fail(ctx, s, out.detail)
        # §4.3：回報這一輪所有還在進行的檔名（伺服器會實際到 S3 確認，傳到了就算數）
        names = [s.path.name for s in pending if not s.paused]
        confirmed = _complete(ctx, client, names, payload)
        for s in pending:
            if s.paused:
                continue
            if confirmed is None:
                if s.path.name in put_ok:
                    _fail(ctx, s, "回報完成（/api/ingest/media/complete）失敗，無法確認是否已在 S3")
            elif s.path.name in confirmed:
                s.done = True
            elif s.path.name in put_ok:
                _fail(ctx, s, "回報完成時伺服器在 S3 找不到這個檔案")
        log.info("第 %d 輪結束：已確認 %d/%d 檔，暫停 %d 檔", rnd, sum(s.done for s in states), len(states),
                 sum(s.paused for s in states))
    return states


def _presign(ctx: Ctx, client: ab.Client, chunk: list[FileState], payload) -> tuple[dict | None, float, float]:
    names = [s.path.name for s in chunk if not s.paused]
    try:
        data = client.post_json("/api/ingest/media/presign", payload(names), attempts=1)
    except Exception as e:  # noqa: BLE001
        err = f"取得上傳網址失敗：{describe(e)}"
        log.warning("%s（%d 檔）", err, len(names))
        for s in chunk:
            if not s.paused:  # 4xx（例如 401 金鑰錯誤）依規格不重試
                _fail(ctx, s, err, permanent=isinstance(e, ab.ApiError))
        return None, 0.0, 0.0
    uploads = {u.get("name"): u for u in (data.get("uploads") or []) if isinstance(u, dict)}
    expires = float(data.get("expires_in") or 3600)
    log.info("取得上傳網址：%d 檔（有效 %.0f 秒）", len(names), expires)
    return uploads, expires, time.monotonic()


def _complete(ctx: Ctx, client: ab.Client, names: list[str], payload) -> set[str] | None:
    """回報完成。失敗時重試（回報本身不會造成重複），仍失敗則回 None。"""
    if not names:
        return set()
    confirmed: set[str] = set()
    for i in range(0, len(names), ctx.st.s3.presign_batch):
        part = names[i:i + ctx.st.s3.presign_batch]
        ok, data, err = try_file(ctx, f"回報完成（{len(part)} 檔）",
                                 lambda: client.post_json("/api/ingest/media/complete", payload(part), attempts=1))
        if not ok:
            return None
        confirmed |= set(data.get("stored") or [])
    return confirmed


def _already_stored(ctx: Ctx, client: ab.Client, names: list[str], payload) -> set[str]:
    """補傳前的檢查：complete 會到 S3 實際確認，回報 stored／missing，不會改動任何東西。

    問不到（網路、伺服器錯誤）就當成都還沒傳，全部照常上傳 —— 多傳一次只是覆蓋同一個位置。
    """
    stored: set[str] = set()
    for i in range(0, len(names), ctx.st.s3.presign_batch):
        part = names[i:i + ctx.st.s3.presign_batch]
        try:
            data = client.post_json("/api/ingest/media/complete", payload(part), attempts=1)
        except Exception as e:  # noqa: BLE001
            log.warning("補傳：無法確認哪些檔案已在 S3（%s）—— 全部重新上傳", describe(e))
            return set()
        stored |= set(data.get("stored") or [])
    return stored


def upload_media_files_v11(ctx: Ctx, client: ab.Client, files: list[Path]) -> list[FileState]:
    """v1.1（Deprecated）：經 API 伺服器轉存。一檔一個請求，各自計次。"""
    ab.warn_deprecated("aibox_uploader media_api=v1.1／POST /api/ingest/media", logger=log)
    states = []
    for p in files:
        s = FileState(p)

        def send(p: Path = p) -> None:
            data = client.post("/api/ingest/media", data={"store_code": ctx.store, "biz_date": ctx.day},
                               parts=[ab.Part("files[]", p)], attempts=1)
            if p.name not in (data.get("stored") or []):
                raise ab.VerifyError("伺服器沒有回報已存")

        ok, _, err = try_file(ctx, p.name, send)
        s.done, s.paused, s.error = ok, not ok, err
        states.append(s)
    return states


def step_media(ctx: Ctx, client: ab.Client | None, kinds: list[str], csv_ok: bool = True) -> list[Result]:
    """上傳媒體，回傳每一類的結果。

    CSV 沒成功時（all 模式）：影片不送（規格 §1：CSV 失敗不要送影片），高清影像照送。
    兩類一起掃描，兩個資料夾出現同名檔案時才查得出來（在伺服器上是同一個位置，會互相覆蓋）。
    """
    found = find_media(ctx, kinds)
    blocked = [k for k in kinds if k == "video" and not csv_ok]
    send = [k for k in kinds if k not in blocked]
    results = []
    for k in blocked:
        n = len(found[k])
        results.append(Result(KIND_LABEL[k], "FAIL" if n else "SKIP",
                              f"CSV 沒有成功上傳，依規格不送影片（{n} 檔待傳）" if n else "沒有要上傳的檔案（影片 0 檔）"))
    if not send:
        return results
    item = "＋".join(KIND_LABEL[k] for k in send)
    files = [p for k in send for p in found[k]]
    counts = "、".join(f"{KIND_LABEL[k]} {len(found[k])} 檔" for k in send)
    if not files:
        return results + [Result(item, "SKIP", f"沒有要上傳的檔案（{counts}）")]
    total = sum(p.stat().st_size for p in files if p.exists())
    if ctx.dry_run:
        return results + [Result(item, "OK", f"[dry-run] 會上傳 {counts} / {ab.human(total)}")]
    upload = upload_media_files_v11 if ctx.st.media_api == ab.MEDIA_API_V11 else upload_media_files
    states = upload(ctx, client, files)
    paused = [s for s in states if not s.done]
    for s in paused:
        ctx.problems.append(f"{s.path}：{s.error}（失敗 {s.attempts} 次，已暫停）")
    if paused:
        return results + [Result(item, "FAIL", f"{counts}：{len(states) - len(paused)} 檔完成，"
                                               f"{len(paused)} 檔暫停（見下方清單）")]
    return results + [Result(item, "OK", f"{counts} / {ab.human(total)}，全部確認在 S3")]


def guarded(item: str, fn, *args) -> list[Result]:
    """最後一道防線：步驟裡任何沒預料到的例外都轉成 FAIL，其他步驟照常進行、結果一定印出。"""
    try:
        out = fn(*args)
        return out if isinstance(out, list) else [out]
    except Exception as e:  # noqa: BLE001
        log.exception("%s：發生預期外的錯誤", item)
        return [Result(item, "FAIL", describe(e))]


def run(ctx: Ctx, client: ab.Client | None) -> list[Result]:
    if ctx.mode == "csv":
        return guarded("csv", step_csv, ctx, client)
    if ctx.mode == "log":
        return guarded("log", step_log, ctx, client)
    if ctx.mode in ("video", "image"):
        return guarded(KIND_LABEL[ctx.mode], step_media, ctx, client, [ctx.mode])
    # all：log 與 CSV／媒體無關，先送；CSV 成功才送影片（§1），高清影像不受 CSV 影響
    results = guarded("log", step_log, ctx, client)
    csv_result = guarded("csv", step_csv, ctx, client)
    results += csv_result
    csv_ok = csv_result[0].status == "OK"
    results += guarded("影片＋高清影像", step_media, ctx, client, ["video", "image"], csv_ok)
    return results


# ── 失敗日期的自動補傳 ──────────────────────────────────────────────────────
class Pending:
    """補傳清單：記下哪些營業日沒傳完、用哪個模式，下次執行時自動再傳一次。

    檔案在 <prefix>/<catch_up.state_dir>/pending_<店>.json，每家店一個：
        {"store": "07c6", "days": {"20261012": {"modes": ["all"], "runs": 1, "first_failed": "…",
                                                 "last_tried": "…", "last_error": "…"}}}
    讀寫失敗只留 warning，不影響上傳；內容壞掉就改名成 .bad，重新開始記錄。
    dry-run 只讀不寫。
    """

    def __init__(self, ctx: Ctx):
        st = ctx.st
        folder = Path(st.catch_up_dir.replace("{store}", ctx.store))
        if not (folder.is_absolute() or folder.root):
            folder = ctx.prefix / folder
        self.path = folder / f"pending_{ctx.store.lower()}.json"  # 店代號不分大小寫（規格 v1.2.6）
        self.store, self.max_runs, self.max_days = ctx.store, st.catch_up_runs, st.catch_up_days
        self.readonly = ctx.dry_run
        self.days: dict[str, dict] = {}
        self.gave_up: list[str] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            days = data.get("days") if isinstance(data, dict) else None
            if not isinstance(days, dict):
                raise ValueError("缺少 days 物件")
            for day, e in days.items():
                modes = [m for m in MODES if m in (e.get("modes") or [])] if isinstance(e, dict) else []
                if re.fullmatch(r"\d{8}", str(day)) and modes:
                    self.days[str(day)] = {**e, "modes": modes, "runs": int(e.get("runs") or 0)}
        except OSError as e:
            log.warning("讀不到補傳清單 %s：%s —— 這次不補傳先前的日期", self.path, describe(e))
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError) as e:
            self.days = {}
            bad = self.path.with_name(self.path.name + ".bad")
            log.warning("補傳清單 %s 內容損壞（%s）—— 改名為 %s，重新開始記錄", self.path, describe(e), bad.name)
            if not self.readonly:
                try:
                    os.replace(self.path, bad)
                except OSError as e2:
                    log.warning("無法改名損壞的補傳清單：%s", describe(e2))

    def save(self) -> None:
        if self.readonly:
            return
        try:
            if not self.days:
                if self.path.exists():
                    self.path.unlink()
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            body = {"_說明": "aibox_uploader 的補傳清單（程式自動維護）", "store": self.store, "days": self.days}
            tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)  # 先寫暫存檔再換掉，寫到一半斷電也不會留下壞檔
        except OSError as e:
            log.warning("無法寫入補傳清單 %s：%s —— 這些日期下次不會自動補傳：%s", self.path, describe(e),
                        "、".join(sorted(self.days)) or "（無）")

    @staticmethod
    def age(day: str) -> int:
        return (datetime.now().date() - datetime.strptime(day, "%Y%m%d").date()).days

    def give_up(self, day: str, why: str) -> None:
        e = self.days.pop(day, None) or {}
        self.gave_up.append(day)
        log.error("營業日 %s：%s —— 放棄自動補傳，請人工處理（模式 %s，最後的錯誤：%s）", day, why,
                  "、".join(e.get("modes") or []) or "-", e.get("last_error") or "-")

    def settle(self, day: str, outcomes: dict[str, str | None], folder_missing: bool = False) -> None:
        """一個營業日處理完後更新清單。outcomes：{模式: None（成功）或錯誤說明}。"""
        e = self.days.get(day)
        if folder_missing:  # 補傳時日期資料夾不見了（多半是本機已清檔），再試也沒用
            if e:
                self.give_up(day, "找不到日期資料夾（檔案可能已被清除）")
            return
        for mode, err in outcomes.items():
            if err is None and e:
                e["modes"] = [] if mode == "all" else [m for m in e["modes"] if m != mode]
        failed = {m: err for m, err in outcomes.items() if err is not None}
        if not failed:
            if e is not None and not e["modes"]:
                del self.days[day]
                if e.get("runs"):
                    log.info("營業日 %s 已補傳完成，從補傳清單移除", day)
            return
        if e is None:
            if self.age(day) > self.max_days:
                log.warning("營業日 %s 已超過 %d 天，不列入自動補傳，請人工處理", day, self.max_days)
                return
            e = self.days[day] = {"modes": [], "runs": 0,
                                  "first_failed": datetime.now().isoformat(timespec="seconds")}
        modes = set(e["modes"]) | set(failed)
        e["modes"] = ["all"] if "all" in modes else [m for m in MODES if m in modes]
        e["runs"] = int(e.get("runs") or 0) + 1
        e["last_tried"] = datetime.now().isoformat(timespec="seconds")
        e["last_error"] = next(iter(failed.values()))[:500]
        if e["runs"] >= self.max_runs:
            self.give_up(day, f"已試 {e['runs']} 次仍未成功")
        else:
            log.warning("營業日 %s 有檔案沒傳成功，已列入補傳清單（第 %d/%d 次），下次執行時自動補傳",
                        day, e["runs"], self.max_runs)


# ── log 與主程式 ────────────────────────────────────────────────────────────
FORMAT ="%(asctime)s %(levelname)-7s %(name)s %(message)s"


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
    root.addHandler(h)


def attach_run_log(ctx: Ctx) -> tuple[Path | None, logging.Handler | None]:
    """程式自己的 log 檔：預設 <prefix>/<date>/csv_logs/<date>_<店>_upload.log，同一天重跑會接在後面。

    每個營業日（含補傳的日期）各寫進自己那天的 log 檔；處理完要用 detach_run_log 拿掉。
    開不了檔（權限、路徑被佔用）只留 warning、改成只輸出到畫面，不影響上傳。
    """
    rl = ctx.cfg["run_log"]
    if not rl.get("enabled", True):
        return None, None
    try:
        name = str(rl.get("name") or "{date}_{store}_upload.log").format(date=ctx.day, store=ctx.store)
    except (KeyError, IndexError, ValueError) as e:
        raise ab.ConfigError(f"設定檔 run_log.name 只能用 {{date}} 與 {{store}}：{e}") from None
    folder = Path(str(rl.get("dir") or "."))
    if not (folder.is_absolute() or folder.root):
        folder = ctx.day_dir / folder if ctx.day_dir.is_dir() else Path.cwd()  # 日期資料夾不在：不另外建立
    path = folder / name
    try:
        folder.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(path, encoding="utf-8")
    except OSError as e:
        log.warning("無法寫入 log 檔 %s：%s —— 改成只輸出到畫面，照常上傳", path, describe(e))
        return None, None
    h.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
    logging.getLogger().addHandler(h)
    return path, h


def detach_run_log(h: logging.Handler | None) -> None:
    if h is not None:
        logging.getLogger().removeHandler(h)
        h.close()


def report(ctx: Ctx, results: list[Result], elapsed: float) -> int:
    log.info("==== 上傳結果：店 %s，營業日 %s，模式 %s%s%s，耗時 %.1f 秒 ====",
             ctx.store, ctx.day, ctx.mode, "（自動補傳）" if ctx.catching_up else "",
             "（dry-run，未連線）" if ctx.dry_run else "", elapsed)
    for r in results:
        (log.error if r.status == "FAIL" else log.info)("%-8s %-4s %s", r.item, r.status, r.detail)
    for p in ctx.problems:
        log.error("沒有上傳：%s", p)
    code = 1 if ctx.problems or any(r.status == "FAIL" for r in results) else 0
    log.info("結束代碼：%d", code)
    return code


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aibox_uploader.py", description="AI Box 每日上傳程式（API v1.2.8）",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("mode", choices=MODES, help="上傳方式：all / csv / log / video / image")
    p.add_argument("--store", help="店家代碼（優先於設定檔的 store）")
    p.add_argument("--prefix-path", help="上傳檔案的根目錄，日期資料夾在它底下（優先於設定檔的 prefix_path）")
    p.add_argument("--date", help="日期資料夾名稱 YYYYMMDD，也可以 today / yesterday（預設 yesterday，本機時間）")
    p.add_argument("--config", help="設定檔路徑（JSON）；命令列參數優先於設定檔")
    p.add_argument("--max-attempts", type=int,
                   help=f"同一個檔案失敗幾次就暫停（優先於設定檔 retry.max_attempts，再來是 {MAX_ATTEMPTS_ENV}，預設 {MAX_ATTEMPTS}）")
    p.add_argument("--dry-run", action="store_true",
                   help="只列出會上傳哪些檔案（含補傳清單裡的日期），不連線、不需要金鑰、不改補傳清單")
    p.add_argument("--no-catch-up", action="store_true", help="這次不補傳先前失敗的日期（清單保留，下次照常補傳）")
    p.add_argument("-v", "--verbose", action="store_true", help="輸出除錯訊息")
    return p


class Conn:
    """API 連線：第一次需要時才建立（讀金鑰、憑證），之後每個營業日共用。dry-run 不建立。"""

    def __init__(self, ctx: Ctx):
        self.ctx, self.client = ctx, None

    def get(self) -> ab.Client | None:
        if self.ctx.dry_run:
            return None
        if self.client is None:
            self.client = make_client(self.ctx)
        return self.client


def process_date(ctx: Ctx, conn: Conn, cfg_from: str) -> tuple[int, str | None, bool]:
    """上傳一個營業日（一個模式），寫進那天的 log 檔。回傳 (結束代碼, 失敗原因或 None, 日期資料夾不存在)。"""
    t0 = time.monotonic()
    run_log, handler = attach_run_log(ctx)
    try:
        log.info("aibox_uploader 開始：模式 %s，店 %s，營業日 %s，根目錄 %s%s", ctx.mode, ctx.store, ctx.day,
                 ctx.prefix, "（自動補傳）" if ctx.catching_up else "")
        log.info("設定檔：%s；log 檔：%s；每個檔案最多試 %d 次（間隔 %.0f 秒起）", cfg_from,
                 run_log or "（無，只輸出到畫面）", ctx.st.max_attempts, ctx.st.backoff)
        if not ctx.day_dir.is_dir():
            msg = f"找不到日期資料夾 {ctx.day_dir}（請確認 --prefix-path / --date）"
            return report(ctx, [Result(ctx.mode, "FAIL", msg)], time.monotonic() - t0), msg, True
        try:
            client = conn.get()
        except ab.ConfigError as e:
            log.error("設定錯誤：%s（沒有上傳任何檔案）", e)
            e.logged = True  # type: ignore[attr-defined]
            raise
        results = run(ctx, client)
        code = report(ctx, results, time.monotonic() - t0)
        err = next((f"{r.item}：{r.detail}" for r in results if r.status == "FAIL"), None) or \
            (ctx.problems[0] if ctx.problems else None)
        return code, (err or "有檔案沒上傳") if code else None, False
    finally:
        detach_run_log(handler)


def catch_up(base: Ctx, conn: Conn, pending: Pending, cfg_from: str) -> list[int]:
    """補傳清單裡的其他日期（由舊到新），每個日期用當初失敗的模式再跑一次。"""
    days = sorted(d for d in pending.days if d != base.day)
    if not days:
        return []
    log.info("==== 自動補傳：清單裡有 %d 個營業日（%s）%s ====", len(days), "、".join(days),
             "（dry-run，只列出、不改清單）" if base.dry_run else "")
    codes = []
    for day in days:
        e = pending.days[day]
        if pending.age(day) > pending.max_days:
            if base.dry_run:
                log.warning("[dry-run] 營業日 %s 已超過 %d 天，實際執行時會放棄補傳", day, pending.max_days)
            else:
                pending.give_up(day, f"已超過 {pending.max_days} 天")
                pending.save()
            codes.append(1)
            continue
        log.info("── 補傳營業日 %s：模式 %s，第 %d/%d 次 ──", day, "、".join(e["modes"]), e["runs"] + 1,
                 pending.max_runs)
        outcomes: dict[str, str | None] = {}
        missing = False
        for mode in list(e["modes"]):
            ctx = replace(base, mode=mode, day=day, catching_up=True, problems=[])
            code, err, missing = process_date(ctx, conn, cfg_from)
            codes.append(code)
            if missing:
                break
            outcomes[mode] = err
        if not base.dry_run:
            pending.settle(day, outcomes, folder_missing=missing)
            pending.save()  # 每個日期處理完就存，中途被中斷也不會忘記
    return codes


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        # ── 設定一次檢查完：有錯就在上傳任何東西之前停下（結束代碼 2）
        cfg, cfg_from = load_config(args.config)
        store = args.store or cfg.get("store")
        if not store:
            raise ab.ConfigError("需要店家代碼：請給 --store，或在設定檔寫 store")
        store = str(store).strip()
        if not ab.STORE_CODE_RE.match(store):
            raise ab.ConfigError(f"店家代碼只能用英數、-、_，1–32 字元：{store!r}")
        prefix = args.prefix_path or cfg.get("prefix_path")
        if not prefix:
            raise ab.ConfigError("需要上傳檔案的根目錄：請給 --prefix-path，或在設定檔寫 prefix_path")
        ctx = Ctx(args.mode, store, resolve_day(args.date), Path(prefix), cfg, resolve_settings(cfg, args),
                  args.dry_run)
        conn = Conn(ctx)
        if not ctx.st.catch_up:
            return process_date(ctx, conn, cfg_from)[0]

        pending = Pending(ctx)
        ctx.catching_up = ctx.day in pending.days  # 手動重跑清單裡的日期：一樣先略過已在 S3 的檔案
        code, err, missing = process_date(ctx, conn, cfg_from)
        if not ctx.dry_run:
            # 日期資料夾不存在（--date 打錯、那天沒營業）不列入補傳；本來就在清單裡的則放棄
            pending.settle(ctx.day, {ctx.mode: err}, folder_missing=missing)
            pending.save()
        codes = [code] + catch_up(ctx, conn, pending, cfg_from)
        if len(codes) > 1 or pending.gave_up or pending.days:
            log.info("==== 補傳清單：%s；這次放棄 %s；結束代碼 %d ====",
                     "、".join(f"{d}（{'、'.join(e['modes'])}，已試 {e['runs']} 次）"
                              for d, e in sorted(pending.days.items())) or "已清空",
                     "、".join(pending.gave_up) or "無", max(codes))
        return max(codes)
    except ab.ConfigError as e:
        if not getattr(e, "logged", False):
            log.error("設定錯誤：%s（沒有上傳任何檔案）", e)
        return 2
    except KeyboardInterrupt:
        log.error("使用者中斷")
        return 1


if __name__ == "__main__":
    sys.exit(main())
