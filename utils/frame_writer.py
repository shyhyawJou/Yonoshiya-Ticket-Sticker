import queue
import re
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import logging
from loguru import logger as loguru_logger

logger = logging.getLogger(__name__)


def full_screen_tag(task_type: str) -> str:
    """OCR 任務類型 -> full_screen 檔名標籤：ticket / sticker / ticket_extra -> ticket / sticker / ticket-extra。"""
    return task_type.replace("_", "-")


def frame_filename(current_time: float, tag: str = "") -> str:
    """存圖檔名：<YYYYMMDD_HHMMSS>_<mmm>[_<標籤>].jpg

    標籤 = 這張圖「為什麼存」（見 app.py 的 full_screen：ticket / sticker / ticket-extra / sauce）。
    不帶標籤時維持舊格式 <YYYYMMDD_HHMMSS>_<mmm>.jpg。
    標籤只留英數與連字號：aibox_uploader 的檔名規則是 ^[A-Za-z0-9._-]{1,200}$（不能有空白與中日文）。
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
    msec = int(current_time * 1000) % 1000
    tag = re.sub(r"[^A-Za-z0-9-]", "-", tag or "")
    return f"{timestamp}_{msec:03d}_{tag}.jpg" if tag else f"{timestamp}_{msec:03d}.jpg"


class AsyncFrameWriter:
    """
    背景執行緒非同步寫檔（含建立資料夾），主迴圈只負責丟資料進 queue。
    """

    def __init__(self, max_queue_size: int = 30):
        self._queue: "queue.Queue[tuple]" = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._dropped_count = 0
        self._known_dirs = set()  # 快取已建立過的資料夾，避免每張圖都呼叫 mkdir
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def save(self, frame, save_dir: Path, folder_name: str, tag: str = "") -> Optional[str]:
        """
        非阻塞。frame 必須先 copy()，因為主迴圈的 buffer 可能馬上被下一幀覆寫。
        timestamp 在這裡（呼叫當下）產生，才能反映真正拍攝的時間，
        而不是背景執行緒實際寫檔的時間（queue 若有堆積，兩者可能差好幾百毫秒）。

        tag：這張圖為什麼存，會成為檔名後綴（見 frame_filename）。
        回傳預定的檔名；佇列已滿被丟棄時回傳 None。丟棄的行為與計數、警告完全沒變，只是把結果回報給呼叫端。
        """
        current_time = time.time()
        try:
            self._queue.put_nowait((frame, save_dir, folder_name, current_time, tag))
        except queue.Full:
            self._dropped_count += 1
            if self._dropped_count % 50 == 1:
                logger.warning(f"AsyncFrameWriter queue 已滿，累積丟棄 {self._dropped_count} 幀")
            return None
        return frame_filename(current_time, tag)

    def _worker(self):
        while self._running:
            try:
                frame, save_dir, folder_name, current_time, tag = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                save_path_dir = save_dir / folder_name

                if folder_name not in self._known_dirs:
                    save_path_dir.mkdir(parents=True, exist_ok=True)
                    self._known_dirs.add(folder_name)

                filename = frame_filename(current_time, tag)

                save_path = save_path_dir / filename
                t_write = time.perf_counter()
                ok = cv2.imwrite(str(save_path), frame)

                # 寫圖成功才記（imwrite 失敗會回傳 False，這種情況不算成功、維持原本不記）。
                # 用 loguru：上面那個 logger 是標準 logging，App 沒有把它轉到 loguru 的輸出，
                # 寫在那邊的訊息不一定會進你的 log 檔。記錄本身出錯不能把「已經寫好的圖」當成失敗。
                if ok:
                    try:
                        loguru_logger.info(
                            f"[存圖] {Path(folder_name).name}/{filename} 寫圖成功 "
                            f"{save_path.stat().st_size / 1e6:.2f}MB 寫檔 {(time.perf_counter() - t_write) * 1000:.0f}ms "
                            f"佇列剩 {self._queue.qsize()}"
                        )
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"儲存影像失敗 (Folder: {folder_name}): {e}")
            finally:
                self._queue.task_done()

    def stop(self, wait: bool = True, timeout: float = 5.0):
        self._running = False
        if wait:
            self._thread.join(timeout=timeout)