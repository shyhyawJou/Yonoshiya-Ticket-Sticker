from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _natural_sort_key(path: Path):
    """
    讓 frame_2.jpg 排在 frame_10.jpg 前面,而不是照字串排到 frame_10 < frame_2。
    """
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


class OfflineSource:
    """
    離線影像來源,提供跟 cv2.VideoCapture 相容的介面 (read / isOpened / release)。

    傳入一個資料夾路徑,會自動判斷內容:
      - 資料夾內有影片檔 (.mp4/.avi/...) -> 影片模式。
        若有多支影片,會依檔名排序後依序播放,一支播完自動接下一支。
      - 資料夾內沒有影片但有圖片 (.jpg/.png/...) -> 圖片序列模式。
        會依檔名做 natural sort 後,一張一張當作影格吐出來。
      - 兩者都沒有 -> 開啟失敗 (isOpened() 回傳 False)。

    也支援直接傳入單一影片檔案路徑 (非資料夾),方便單支影片測試。

    Args:
        path: 資料夾路徑 (或單一影片檔案路徑)。
        loop: 播完/讀完是否從頭重來，預設 False (讀完即結束，read() 回傳 (False, None))。
        fps: 圖片序列模式下，模擬送出影格的節奏 (每秒幾張)。
             影片模式下不使用 (影片本身有自己的節奏，讀多快算多快，
             如需要跟真實時間同步，請自行在外層依 fps 做 sleep)。
    """

    def __init__(self, path: str, loop: bool = False, fps: float = 15.0):
        self.path = Path(path)
        self.loop = loop
        self.target_fps = fps if fps and fps > 0 else 0.0
        self._frame_interval = (1.0 / self.target_fps) if self.target_fps else 0.0
        self._last_read_time: Optional[float] = None

        self._mode: Optional[str] = None  # "video" or "image"
        self._opened = False

        # --- 影片模式用 ---
        self._video_files: List[Path] = []
        self._video_idx = 0
        self._current_cap: Optional[cv2.VideoCapture] = None

        # --- 圖片模式用 ---
        self._image_paths: List[Path] = []
        self._image_idx = 0

        self._build_source()

    # ---------- 初始化 ----------
    def _build_source(self) -> None:
        if not self.path.exists():
            logger.error(f"[OfflineSource] 路徑不存在: {self.path}")
            return

        # 允許直接指定單一影片檔案
        if self.path.is_file():
            if self.path.suffix.lower() in VIDEO_EXTS:
                self._mode = "video"
                self._video_files = [self.path]
                self._opened = self._open_next_video()
            else:
                logger.error(f"[OfflineSource] 不支援的檔案類型: {self.path}")
            return

        # 資料夾模式:掃描內容
        video_files = sorted(
            (f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS),
            key=_natural_sort_key,
        )
        image_files = sorted(
            (f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS),
            key=_natural_sort_key,
        )

        if video_files:
            self._mode = "video"
            self._video_files = video_files
            logger.info(f"[OfflineSource] 影片模式，共 {len(video_files)} 支影片: "
                        f"{[f.name for f in video_files]}")
            self._opened = self._open_next_video()

        elif image_files:
            self._mode = "image"
            self._image_paths = image_files
            logger.info(f"[OfflineSource] 圖片序列模式，共 {len(image_files)} 張影像")
            self._opened = True

        else:
            logger.error(f"[OfflineSource] 資料夾內找不到可用的影片或圖片: {self.path}")

    def _open_next_video(self) -> bool:
        """
        開啟 self._video_idx 指向的下一支影片。成功回傳 True 並把游標往後移一格，
        找不到可開啟的影片 (含跳過壞檔後仍找不到) 則回傳 False。
        """
        while self._video_idx < len(self._video_files):
            vf = self._video_files[self._video_idx]
            self._video_idx += 1
            cap = cv2.VideoCapture(str(vf))
            if cap.isOpened():
                self._current_cap = cap
                logger.info(f"[OfflineSource] 開始播放: {vf.name}")
                return True
            logger.warning(f"[OfflineSource] 無法開啟影片，略過: {vf.name}")
            cap.release()

        self._current_cap = None
        return False

    # ---------- 對外介面 ----------
    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None

        self._throttle()

        if self._mode == "video":
            return self._read_video()
        elif self._mode == "image":
            return self._read_image()
        return False, None

    def release(self) -> None:
        if self._current_cap is not None:
            self._current_cap.release()
            self._current_cap = None
        self._opened = False
        logger.info("[OfflineSource] 已釋放。")

    # ---------- 內部細節 ----------
    def _throttle(self) -> None:
        """
        圖片序列模式下用來模擬 fps 節奏 (避免整批圖片被瞬間讀完)。
        影片模式不需要，因為 VideoCapture.read() 本身就有解碼耗時。
        """
        if self._mode != "image" or self._frame_interval <= 0:
            return
        now = time.time()
        if self._last_read_time is not None:
            elapsed = now - self._last_read_time
            remain = self._frame_interval - elapsed
            if remain > 0:
                time.sleep(remain)
        self._last_read_time = time.time()

    def _read_video(self) -> Tuple[bool, Optional[np.ndarray]]:
        while True:
            if self._current_cap is None:
                return False, None

            ok, frame = self._current_cap.read()
            if ok:
                return True, frame

            # 這支影片讀完了，接下一支
            self._current_cap.release()
            if self._open_next_video():
                continue

            # 全部影片都播完了
            if self.loop:
                self._video_idx = 0
                if self._open_next_video():
                    continue
            return False, None

    def _read_image(self) -> Tuple[bool, Optional[np.ndarray]]:
        while True:
            if self._image_idx >= len(self._image_paths):
                if self.loop:
                    self._image_idx = 0
                else:
                    return False, None

            img_path = self._image_paths[self._image_idx]
            self._image_idx += 1

            frame = cv2.imread(str(img_path))
            if frame is None:
                logger.warning(f"[OfflineSource] 讀取失敗，略過: {img_path.name}")
                continue

            return True, frame