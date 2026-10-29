import queue
import threading
import time
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)


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

    def save(self, frame, save_dir: Path, folder_name: str) -> None:
        """
        非阻塞。frame 必須先 copy()，因為主迴圈的 buffer 可能馬上被下一幀覆寫。
        timestamp 在這裡（呼叫當下）產生，才能反映真正拍攝的時間，
        而不是背景執行緒實際寫檔的時間（queue 若有堆積，兩者可能差好幾百毫秒）。
        """
        current_time = time.time()
        try:
            self._queue.put_nowait((frame, save_dir, folder_name, current_time))
        except queue.Full:
            self._dropped_count += 1
            if self._dropped_count % 50 == 1:
                logger.warning(f"AsyncFrameWriter queue 已滿，累積丟棄 {self._dropped_count} 幀")

    def _worker(self):
        while self._running:
            try:
                frame, save_dir, folder_name, current_time = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                save_path_dir = save_dir / folder_name

                if folder_name not in self._known_dirs:
                    save_path_dir.mkdir(parents=True, exist_ok=True)
                    self._known_dirs.add(folder_name)

                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
                msec = int(current_time * 1000) % 1000
                filename = f"{timestamp}_{msec:03d}.jpg"

                save_path = save_path_dir / filename
                cv2.imwrite(str(save_path), frame)

            except Exception as e:
                logger.error(f"儲存影像失敗 (Folder: {folder_name}): {e}")
            finally:
                self._queue.task_done()

    def stop(self, wait: bool = True, timeout: float = 5.0):
        self._running = False
        if wait:
            self._thread.join(timeout=timeout)