from __future__ import annotations

import numpy as np
from loguru import logger

from manager.camera_controller import CameraController
from utils.offline_source import OfflineSource


class OfflineCameraController(CameraController):
    """
    離線測試專用，繼承 CameraController，只覆寫「取像來源」跟「入列策略」
    兩處，其餘 (worker thread 生命週期、reload、queue 管理) 完全沿用
    父類別邏輯 —— 這樣離線工具永遠自動跟著 CameraController 的行為走，
    不用擔心兩邊邏輯之後會分岔。

    ⚠️ 內部工具，正式環境的 StreamManager 不會用到這支程式。
    """

    def __init__(
        self,
        cfg,
        bus,
        source_path: str,
        loop: bool = False,
        fps: float = 15.0,
        accuracy_mode: bool = False,
        video_recorder=None,
        camera_alone: bool = True,
        video_record_size: tuple[int, int] = (1280, 960),
    ):
        """
        accuracy_mode:
            False (預設) -> 重現線上行為，佇列滿了會丟舊幀，consumer 跟不上就跳幀。
                            適合拿預錄影片模擬線上運行、測穩定性/reset 邏輯。
            True         -> 逐幀驗證模式，佇列滿了會阻塞等待 consumer 消化完，
                            保證每一幀都會進到 mmr/logic/ocr。適合對預錄資料
                            做準確率驗證 (不可漏幀)。
        """
        super().__init__(
            cfg=cfg,
            bus=bus,
            video_recorder=video_recorder,
            camera_alone=camera_alone,
            video_record_size=video_record_size,
        )
        self._source_path = source_path
        self._loop = loop
        self._offline_fps = fps
        self._accuracy_mode = accuracy_mode

    # ---------- 覆寫:取像來源 ----------
    def _open(self) -> None:
        self.capture = OfflineSource(
            self._source_path, loop=self._loop, fps=self._offline_fps
        )
        if not self.capture.isOpened():
            raise ValueError(f"Failed to open offline source: {self._source_path}")
        logger.success(
            f"opened offline source [{self._source_path}] "
            f"(loop={self._loop}, accuracy_mode={self._accuracy_mode})"
        )

    # ---------- 覆寫:入列策略 ----------
    def _enqueue_frame(self, frame: np.ndarray) -> None:
        if not self._accuracy_mode:
            # 重現線上行為：沿用父類別的「丟舊幀」策略
            super()._enqueue_frame(frame)
            return

        # 逐幀驗證模式：maxsize=1，滿了就阻塞在這裡等 consumer 拿走，
        # 保證每一幀都會被 stream_frames() 主迴圈消費到，不會漏偵測。
        self.frame_queue.put(frame)