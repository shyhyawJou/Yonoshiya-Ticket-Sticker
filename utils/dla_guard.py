"""
dla_guard.py
============
DLA 模型載入守衛：分辨「APU 沒起來」與「模型檔本身有問題」，只有前者才重開機。

為什麼需要這個
--------------
現場（不是開發機，開發機從來沒發生過）遇過邊緣平台開機時 APU 沒正常起來，
第一個 DLA 模型就載不進去，而且不 reboot 不會自己好。2026-09-18 現場 log：

    File "/home/root/ocr_mmr_dla/mmr_engine.py", line 21, in __init__
        self.model = RTMDET(path)
    RuntimeError: Load /home/root/ocr_mmr_dla/tasks/ocr/mmr_11cls_v01_02.dla failed !

`app.py::main` 的 except 會記下 traceback，然後 `yoshinoya_ai.service` 的
`Restart=always` 每秒重試一次 —— 但每次都一樣失敗，機器就這樣空轉到有人去現場。

為什麼不能無條件 reboot
-----------------------
2026-09-29 在機器上實測，原生模組（AI.so）對三種失敗丟的是**完全一樣**的
例外，型別、訊息都分不出來：

    路徑不存在   -> RuntimeError: Load /tmp/definitely_missing.dla failed !
    內容是垃圾   -> RuntimeError: Load /tmp/garbage.dla failed !
    檔案被截斷   -> RuntimeError: Load /tmp/trunc.dla failed !

（底層 stderr 有 "Cannot read the file" / "The header of DLA is invalid" /
"Fail to read the DLA file" 的差別，但那不在例外裡，攔 stderr 太脆弱。）

所以「該不該 reboot」只能由我們自己檢查檔案來判斷：

  * 檔案不存在 / 不是一般檔案 / 讀不到 / 大小為 0
        -> 只記 ERROR，**絕不 reboot**。這是部署或路徑設錯，重開一百次也不會好。
  * 檔案看起來正常，載入卻失敗
        -> 判定為 APU 沒起來，reboot。
  * 檔案讀得到，但內容跟「上次成功載入時」不一樣
        -> 疑似換上了壞模型，只記 CRITICAL **不 reboot**，免得一個壞掉的
           部署把機器變成無限重開機。

另外兩層防呆：

  * 連續 reboot 次數上限（預設 2 次）。超過就不再重開，讓機器停在「服務一直
    重啟」這個還能 SSH 進去的狀態等人處理，而不是反覆重開機。
  * 任何一次全部模型載入成功就把狀態清掉（`mark_ok()`），所以偶發一次的 APU
    問題永遠有完整的 reboot 額度，不會被幾個月前的計數擋住。

驗證用
------
`DLA_GUARD_DRYRUN=1` 環境變數：照常判斷、照常寫 log 與狀態檔，但不真的重開機。
現場之外要測整條路徑時用這個（開發機上重現不了真正的 APU 失敗，只能用
「把 .dla 換成垃圾內容」之類的方式模擬，見 README 的說明）。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# 狀態檔：要能跨重開機存活，所以放專案目錄（/mnt/reserved 有可能沒掛上）。
# zip 部署是解壓覆蓋，不會刪掉壓縮檔裡沒有的檔案，所以這個檔案也活得下來。
_DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / ".dla_guard_state.json"

# 連續重開上限。超過就只記 CRITICAL，不再重開。
_DEFAULT_MAX_REBOOTS = 2

# 決定重開之後、真正下指令之前等幾秒。留一個讓人來得及 SSH 進來攔的空窗，
# 也讓 log 有時間落地。
_DEFAULT_DELAY_SEC = 1

# 上次重開距今超過這麼久，就當成新的一次事件，計數歸零。萬一 mark_ok()
# 沒被呼叫到（例如每次都在載入階段就掛掉），這是第二道恢復額度的保險。
_DEFAULT_RESET_AFTER_SEC = 6 * 3600

# 指紋只取頭尾各 1 MiB + 檔案大小：足以抓到「換了模型」或「檔案被截斷」，
# 又不必每次啟動去 md5 五個 25 MB 的模型。
_FINGERPRINT_CHUNK = 1024 * 1024


class _Settings:
    enabled: bool = True
    max_reboots: int = _DEFAULT_MAX_REBOOTS
    delay_sec: int = _DEFAULT_DELAY_SEC
    reset_after_sec: int = _DEFAULT_RESET_AFTER_SEC
    state_path: Path = _DEFAULT_STATE_PATH


_settings = _Settings()

# 這個 process 裡成功載入過的模型路徑，mark_ok() 用它寫指紋
_loaded_paths: List[str] = []


def configure(
    enabled: Optional[bool] = None,
    max_reboots: Optional[int] = None,
    delay_sec: Optional[int] = None,
    reset_after_sec: Optional[int] = None,
    state_path: Optional[Any] = None,
) -> None:
    """由呼叫端（TaskContext.load_task）在載入模型之前套用設定。

    全部參數都可以不給 —— 沒給就用模組預設值，所以機器上的 config.yaml
    不需要新增任何一行就能運作（config.yaml 在執行期會被相機參數覆寫，
    是狀態而不只是輸入，能不動就不動）。
    """
    if enabled is not None:
        _settings.enabled = bool(enabled)
    if max_reboots is not None:
        _settings.max_reboots = int(max_reboots)
    if delay_sec is not None:
        _settings.delay_sec = int(delay_sec)
    if reset_after_sec is not None:
        _settings.reset_after_sec = int(reset_after_sec)
    if state_path is not None:
        _settings.state_path = Path(state_path)


# ---------------------------------------------------------------------------
# 狀態檔
# ---------------------------------------------------------------------------

def _read_state() -> Dict[str, Any]:
    try:
        with open(_settings.state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if isinstance(state, dict):
            return state
        logger.warning(f"[dla_guard] 狀態檔格式不對（不是 dict），當成空的：{_settings.state_path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        # 狀態檔壞掉不可以讓 app 起不來，也不可以因此就去重開機
        logger.warning(f"[dla_guard] 讀不到狀態檔 {_settings.state_path}（{e}），當成空的")
    return {}


def _write_state(state: Dict[str, Any]) -> None:
    try:
        tmp = _settings.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _settings.state_path)
    except Exception as e:
        logger.warning(f"[dla_guard] 寫不進狀態檔 {_settings.state_path}（{e}）")


def _fingerprint(path: str) -> Optional[str]:
    """檔案指紋：大小 + 頭尾各 1 MiB 的 md5。讀不到回 None。"""
    try:
        size = os.path.getsize(path)
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(_FINGERPRINT_CHUNK))
            if size > _FINGERPRINT_CHUNK * 2:
                f.seek(-_FINGERPRINT_CHUNK, os.SEEK_END)
                h.update(f.read(_FINGERPRINT_CHUNK))
        return f"{size}:{h.hexdigest()}"
    except Exception as e:
        logger.warning(f"[dla_guard] 算不出指紋 {path}（{e}）")
        return None


# ---------------------------------------------------------------------------
# 檔案體檢
# ---------------------------------------------------------------------------

def _inspect_file(path: str) -> Optional[str]:
    """檔案本身有問題就回一句人看得懂的原因；看起來正常回 None。"""
    if not path:
        return "設定裡的模型路徑是空的"
    p = Path(path)
    if not p.exists():
        return "檔案不存在"
    if not p.is_file():
        return "路徑不是一般檔案（是目錄或裝置？）"
    try:
        size = p.stat().st_size
    except OSError as e:
        return f"讀不到檔案狀態（{e}）"
    if size == 0:
        return "檔案大小是 0"
    try:
        with open(path, "rb") as f:
            if not f.read(1):
                return "檔案讀不出任何內容"
    except OSError as e:
        return f"檔案打不開（{e}）"
    return None


# ---------------------------------------------------------------------------
# 對外主要介面
# ---------------------------------------------------------------------------

def load(factory: Callable[[str], Any], path: str, label: str) -> Any:
    """載入一個 DLA 模型；失敗時按上面的規則決定「只記 log」還是「重開機」。

    Args:
        factory: 原生建構子，例如 AI.RTMDET / AI.PaddleOCR_Det。
        path:    .dla 檔路徑。
        label:   log 裡好認的名字，例如 "物件偵測 RTMDET"。

    Returns:
        原生模型物件。

    Raises:
        原本的例外一律往外拋（reboot 的情況通常拋不到呼叫端就重開了，但
        dry-run、超過重開上限、判定是檔案問題時都會拋，行為跟改動前一致 ——
        app.py::main 照樣記 traceback、service 照樣重啟。）
    """
    file_problem = _inspect_file(path)
    if file_problem is not None:
        # 先擋掉檔案問題，連載入都不用試 —— 這一條永遠不會 reboot
        logger.error(
            f"[dla_guard] {label} 模型檔有問題，不重開機：{file_problem}。path={path}"
            f"（請檢查部署與 config.yaml 的 runtime.model 路徑）"
        )
        raise FileNotFoundError(f"{label} 模型檔有問題（{file_problem}）: {path}")

    try:
        model = factory(path)
    except Exception as e:
        # 檔案體檢過了卻載不進去 -> 高度懷疑 APU 沒起來
        _handle_load_failure(label, path, e)
        raise

    _loaded_paths.append(path)
    logger.info(f"[dla_guard] {label} 模型載入成功：{path}")
    return model


def mark_ok() -> None:
    """全部模型都載入成功時呼叫：清掉重開計數，並記下這一版的檔案指紋。

    指紋是下一次失敗時判斷「檔案有沒有被換過」的依據；計數清零是為了讓
    偶發的 APU 問題永遠有完整的 reboot 額度。
    """
    fingerprints = {}
    for path in _loaded_paths:
        fp = _fingerprint(path)
        if fp:
            fingerprints[path] = fp

    state = _read_state()
    previous = int(state.get("reboot_count", 0) or 0)
    _write_state({
        "reboot_count": 0,
        "last_ok_ts": time.time(),
        "last_ok_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprints": fingerprints,
    })
    if previous:
        logger.info(
            f"[dla_guard] 模型全部載入成功，重開計數從 {previous} 歸零"
            f"（前一次重開機看來是有效的）"
        )


def _handle_load_failure(label: str, path: str, err: Exception) -> None:
    state = _read_state()

    # 1. 檔案跟上次成功載入時不一樣 -> 疑似壞模型，不重開
    known_fp = (state.get("fingerprints") or {}).get(path)
    if known_fp:
        current_fp = _fingerprint(path)
        if current_fp and current_fp != known_fp:
            logger.critical(
                f"[dla_guard] {label} 載入失敗，但這個檔案跟上次成功載入時**不一樣**"
                f"（指紋 {known_fp} -> {current_fp}），判定是模型檔的問題而不是 APU，"
                f"不重開機。path={path} 原始錯誤：{err}"
            )
            return

    # 2. 沒開啟自動重開
    if not _settings.enabled:
        logger.critical(
            f"[dla_guard] {label} 載入失敗，檔案看起來正常，疑似 APU 沒起來；"
            f"但自動重開機已停用（dla_guard.enabled=false），只記錄不處理。"
            f"path={path} 原始錯誤：{err}"
        )
        return

    # 3. 連續重開次數上限
    count = int(state.get("reboot_count", 0) or 0)
    last_ts = float(state.get("last_reboot_ts", 0) or 0)
    if count and last_ts and (time.time() - last_ts) > _settings.reset_after_sec:
        logger.info(
            f"[dla_guard] 上次自動重開是 {(time.time() - last_ts) / 3600:.1f} 小時前，"
            f"超過 {_settings.reset_after_sec / 3600:.1f} 小時，重開計數從 {count} 歸零"
        )
        count = 0

    if count >= _settings.max_reboots:
        logger.critical(
            f"[dla_guard] {label} 載入失敗，檔案看起來正常（疑似 APU 沒起來），"
            f"但已經連續自動重開 {count} 次（上限 {_settings.max_reboots}）仍然失敗，"
            f"**停止自動重開**，請人工到現場檢查。"
            f"path={path} 原始錯誤：{err}"
        )
        return

    # 4. 真的要重開了
    count += 1
    now = time.time()
    state.update({
        "reboot_count": count,
        "last_reboot_ts": now,
        "last_reboot_time_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_reboot_label": label,
        "last_reboot_path": path,
        "last_reboot_error": str(err),
    })
    _write_state(state)

    dry_run = os.environ.get("DLA_GUARD_DRYRUN", "") not in ("", "0", "false", "False")
    logger.critical(
        f"[dla_guard] {label} 載入失敗，但模型檔體檢正常 -> 判定 APU 未正常啟動，"
        f"這種狀況不重開機不會恢復。{_settings.delay_sec} 秒後執行系統重開機"
        f"（第 {count}/{_settings.max_reboots} 次）。"
        f"{'【DRY-RUN：不會真的重開】' if dry_run else ''}"
        f" path={path} 原始錯誤：{err}"
    )

    if dry_run:
        return

    # log 落地再重開，否則重開後查不到原因
    try:
        logger.complete()
        os.sync()
    except Exception:
        pass

    time.sleep(_settings.delay_sec)
    _do_reboot()


def _do_reboot() -> None:
    """依序試可用的重開指令。機器上實測 /usr/sbin/reboot 與 systemctl 都有。"""
    for cmd in (["/usr/sbin/reboot"], ["systemctl", "reboot"], ["/usr/sbin/shutdown", "-r", "now"]):
        try:
            logger.critical(f"[dla_guard] 執行重開機指令：{' '.join(cmd)}")
            subprocess.run(cmd, timeout=30, check=False)
            time.sleep(15)  # 指令下去了，等系統自己關機
        except Exception as e:
            logger.error(f"[dla_guard] 重開機指令 {' '.join(cmd)} 失敗：{e}")
    logger.critical("[dla_guard] 所有重開機指令都失敗，放棄（app 會照原本流程往外拋錯）")
