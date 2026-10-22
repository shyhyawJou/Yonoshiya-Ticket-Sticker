"""
cloud_uploader.py
=================
把同仁開發的 aibox_uploader/（Dashboard API + S3 直傳）接進主程式：
每天 cfg.at 上傳「昨天」的 CSV 與影片，到 at + window_min 一律中止。

設計原則
--------
1. **上傳永遠不能拖住核對站。** 核對餐點是產品本體，雲端上傳是附加功能，
   所以任何檢查失敗都只停用上傳、記 ERROR，主程式照常跑；連線類的問題
   在背景重試。start() 不會阻塞（同仁模組的 start() 失敗時會重試約 30 秒）。

2. **機密不在版控裡。** API 金鑰與伺服器憑證由人工放到
   /etc/yoshinoya_ai/。cloud_upload.env 由這裡自己讀進環境變數，不靠 run_app.sh
   —— app 不一定是經由 run_app.sh 啟動的（手動 python3 app.py、App 的啟動鈕），
   靠啟動腳本載入的話，換一種啟動方式金鑰就靜默消失。
   缺了什麼，log 會直接講缺哪個檔、放到哪裡、看哪份文件，現場不必翻程式碼。
   log 裡絕不印出金鑰的值。

3. **硬性截止。** crontab 的 rclone 會在截止後把同一批日期資料夾 move 到
   OneDrive（move = 上傳後刪除本機檔），兩者重疊就是在刪一個正在上傳的檔案。
   所以截止時間到了就 cancel_all()，不等同仁模組自己結束。

已知限制
--------
- 開機時驗證不了金鑰本身對不對：start() 打的 /api/healthz 不需要金鑰。
  金鑰錯誤要到上傳時拿到 401 才會知道，那時會記一行明確的 ERROR。
- 某天沒傳完（截止或斷線），那些檔案會被 rclone 搬走，之後不會再補傳。
- 同仁模組還在開發中，這裡只用它公開的介面
  （Options / aibox_aws / start / stop / add_job / register_pipeline /
  run_job_now / cancel_all / upload_day），不改它的程式碼。
"""
from __future__ import annotations

import importlib
import logging
import os
import socket
import sys
import threading
from datetime import datetime, time as time_of_day, timedelta
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

SECRETS_FILE = "/etc/yoshinoya_ai/cloud_upload.env"
DOC_HINT = "docs/cloud_upload.md"
JOB_NAME = "每日上傳"
PIPELINE_NAME = "nightly_upload"

# 開機時若已在上傳時段內，剩下的時間少於這個就不補跑（傳不完只會被截止中斷）
MIN_CATCH_UP = timedelta(minutes=10)

_TAG = "[cloud_upload]"


def load_secrets_file(path: str = SECRETS_FILE) -> list:
    """把 KEY=VALUE 格式的 env 檔讀進環境變數，回傳讀到的變數名（不含值）。

    已經存在的環境變數優先、不覆蓋，所以測試時仍可用 export 臨時換值。
    支援 `#` 註解、`export KEY=VALUE` 與加引號的值。
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.warning(f"{_TAG} 無法讀取 {path}（{e}）")
        return []

    loaded = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


class _LoguruBridge(logging.Handler):
    """同仁模組用標準 logging（logger 名稱 aibox_aws / aibox），轉進 loguru，
    這樣上傳紀錄會落在當天的 stream_log 裡，跟其他 log 放在一起。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.log(level, f"{_TAG}[{record.name}] {record.getMessage()}")


class CloudUploader:
    def __init__(self, cfg, csv_log, project_dir: Path):
        self.cfg = cfg
        self.csv_log = csv_log
        self.project_dir = Path(project_dir)

        self._aws = None          # 同仁的 aibox_aws 模組
        self._ab = None           # 同仁的 aibox_upload 模組
        self._box = None
        self._at: Optional[time_of_day] = None
        self._window = timedelta(minutes=max(1, int(cfg.window_min)))
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """不阻塞：檢查設定，通過才在背景連線與計時。"""
        if not self.cfg.enabled:
            logger.info(f"{_TAG} 未啟用（config.yaml 的 cloud_upload.enabled = false）")
            return

        problem = self._preflight()
        if problem:
            logger.error(
                f"{_TAG} 已啟用但{problem}。本次不上傳，主程式照常運作"
                f"（設定方式見 {DOC_HINT}）"
            )
            return

        try:
            self._box = self._build_box()
        except Exception as e:
            logger.error(f"{_TAG} 上傳模組初始化失敗：{e}。本次不上傳，主程式照常運作")
            self._box = None
            return

        threading.Thread(target=self._connect_loop, name="cloud_upload-connect", daemon=True).start()
        threading.Thread(target=self._deadline_loop, name="cloud_upload-deadline", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._box is not None:
            try:
                self._box.stop()
            except Exception as e:
                logger.warning(f"{_TAG} 停止上傳模組時發生錯誤（已忽略）：{e}")

    # ------------------------------------------------------------------
    # 啟動前檢查：只檢查「重試也沒用」的設定問題
    # ------------------------------------------------------------------
    def _preflight(self) -> Optional[str]:
        try:
            self._at = datetime.strptime(self.cfg.at, "%H:%M").time()
        except ValueError:
            return f"上傳時間 cloud_upload.at = {self.cfg.at!r} 格式錯誤（要 HH:MM，例如 \"01:00\"）"

        module_dir = (self.project_dir / self.cfg.module_dir).resolve()
        if not module_dir.is_dir():
            return f"找不到上傳模組資料夾 {module_dir}"

        loaded = load_secrets_file(SECRETS_FILE)
        if loaded:
            logger.info(f"{_TAG} 已從 {SECRETS_FILE} 載入 {', '.join(loaded)}")

        if not os.environ.get("YOSHINOYA_API_KEY", "").strip():
            where = "該檔案不存在" if not os.path.isfile(SECRETS_FILE) else "檔案裡沒有這個變數，或值是空的"
            return f"找不到 API 金鑰：請在 {SECRETS_FILE} 設定 YOSHINOYA_API_KEY（{where}）"

        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))
        try:
            self._ab = importlib.import_module("aibox_upload")
            self._aws = importlib.import_module("aibox_aws")
        except Exception as e:   # 例如沒裝 requests
            return f"無法載入上傳模組（{type(e).__name__}: {e}）；相依套件見 {module_dir / 'requirements.txt'}"

        ca = os.environ.get("YOSHINOYA_CA") or str(module_dir / self._ab.CA_FILE_DEFAULT)
        if not os.path.isfile(ca):
            return (
                f"找不到伺服器憑證 {ca}：請把 .crt 放到 /etc/yoshinoya_ai/，"
                f"並在 {SECRETS_FILE} 用 YOSHINOYA_CA 指定路徑"
            )
        return None

    def _build_box(self):
        for name in ("aibox_aws", "aibox"):
            lg = logging.getLogger(name)
            if not any(isinstance(h, _LoguruBridge) for h in lg.handlers):
                lg.addHandler(_LoguruBridge())
            lg.setLevel(logging.INFO)
            lg.propagate = False

        store = self.cfg.store_code or socket.gethostname()[-4:]   # 與 CsvWriter 的檔名同一個來源
        opt = self._aws.Options(
            prefix_path=self.csv_log.root_dir,
            csv_folder=self.csv_log.folder,
            folder_names=list(self.cfg.folders),
            store_code=store,
            log_to_stdout=False,
        )
        box = self._aws.aibox_aws(opt)
        box.register_pipeline(PIPELINE_NAME, self._nightly)
        box.add_job(JOB_NAME, PIPELINE_NAME, at=self.cfg.at, every="day")
        return box

    # ------------------------------------------------------------------
    # 背景執行緒
    # ------------------------------------------------------------------
    def _connect_loop(self) -> None:
        """連上為止。連不上不影響主程式，也不影響排程的時間（start() 成功才開始計時）。"""
        while not self._stop.is_set():
            try:
                self._box.start()
            except self._ab.ConfigError as e:
                logger.error(f"{_TAG} 設定錯誤：{e}。本次不上傳，主程式照常運作（見 {DOC_HINT}）")
                return
            except Exception as e:
                logger.warning(
                    f"{_TAG} 連不上上傳伺服器（{e}），{self.cfg.retry_sec} 秒後重試；主程式不受影響"
                )
                if self._stop.wait(max(30, int(self.cfg.retry_sec))):
                    return
                continue

            start, end = self._window_around(datetime.now())
            logger.info(
                f"{_TAG} 就緒：伺服器與憑證驗證通過、S3 已探測。店代號 {self._box.opt.store_code}，"
                f"每天 {start:%H:%M} 上傳昨天的 CSV 與 {','.join(self.cfg.folders)}，{end:%H:%M} 截止"
            )
            logger.warning(
                f"{_TAG} 注意：API 金鑰只確認「有設定」，還沒驗證對不對 —— 連線檢查的 /api/healthz "
                f"不需要金鑰。金鑰錯誤要到 {start:%H:%M} 上傳時伺服器回 401 才會知道"
            )
            self._catch_up_if_in_window()
            return

    def _deadline_loop(self) -> None:
        """每天 at + window_min 中止所有未完成的上傳。

        分段等待（最多 60 秒一段）再對照牆上時鐘，NTP 校時或系統時間被改時
        也不會錯過截止點。
        """
        while not self._stop.is_set():
            deadline = self._next_deadline(datetime.now())
            while True:
                remaining = (deadline - datetime.now()).total_seconds()
                if remaining <= 0:
                    break
                if self._stop.wait(min(60.0, remaining)):
                    return
            try:
                n = self._box.cancel_all(reason="超過上傳時段")
            except Exception as e:
                logger.error(f"{_TAG} 截止時中止上傳失敗：{e}")
                n = 0
            if n:
                logger.warning(
                    f"{_TAG} 已到 {deadline:%H:%M} 截止時間，中止 {n} 個未完成的上傳，"
                    f"避免與之後 rclone 搬檔重疊；沒傳完的檔案這次不會再補傳"
                )

    def _nightly(self, box, **_kwargs):
        """排程觸發時跑的 pipeline：上傳昨天的資料。"""
        try:
            res = box.upload_day(day="yesterday")
        except self._ab.ApiError as e:
            if getattr(e, "status", None) == 401:
                logger.error(
                    f"{_TAG} 伺服器回 401：API 金鑰錯誤或已失效，"
                    f"請確認 {SECRETS_FILE} 的 YOSHINOYA_API_KEY 後重啟 app"
                )
            raise
        csv_rows = (res.get("csv") or {}).get("rows")
        logger.info(
            f"{_TAG} 昨日資料上傳完成：CSV {csv_rows} 列、媒體 {len(res.get('stored') or [])} 檔、"
            f"略過 {len(res.get('skipped') or [])} 檔，耗時 {res.get('elapsed', 0):.1f} 秒"
        )
        return res

    # ------------------------------------------------------------------
    # 時段計算
    # ------------------------------------------------------------------
    def _window_around(self, now: datetime) -> Tuple[datetime, datetime]:
        """最近一次（含現在）開始的上傳時段 [start, end)。跨午夜的時段也成立。"""
        start = datetime.combine(now.date(), self._at)
        if start > now:
            start -= timedelta(days=1)
        return start, start + self._window

    def _next_deadline(self, now: datetime) -> datetime:
        _, end = self._window_around(now)
        return end if end > now else end + timedelta(days=1)

    def _catch_up_if_in_window(self) -> None:
        """開機（或 app 重啟）時已經在上傳時段內：同仁模組的排程只會排到明天，
        今晚這次就整個漏掉，所以補跑一次。重送是覆蓋（見 aibox_upload.py §6 的說明），
        就算今晚稍早已經傳過一次也不會重複。"""
        now = datetime.now()
        start, end = self._window_around(now)
        if not (start <= now < end):
            return
        if end - now < MIN_CATCH_UP:
            logger.warning(f"{_TAG} 啟動時已接近 {end:%H:%M} 截止時間，今晚不補跑")
            return
        logger.warning(f"{_TAG} 啟動時已在上傳時段內（{start:%H:%M}~{end:%H:%M}），立刻補跑一次")
        try:
            self._box.run_job_now(JOB_NAME)
        except Exception as e:
            logger.error(f"{_TAG} 補跑失敗：{e}")
