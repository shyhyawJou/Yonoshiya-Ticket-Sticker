from __future__ import annotations

import sys
import signal
import argparse
import traceback
from pathlib import Path

from loguru import logger

from app import StreamManager
from manager.camera_controller import CameraController
from manager.offline_camera_controller import OfflineCameraController


class OfflineStreamManager(StreamManager):
    """
    離線測試專用，繼承 StreamManager，只覆寫「要建立哪一種
    CameraController」，其餘 (mmr / logic / ocr / overlay / streamer)
    完全沿用父類別邏輯。

    ⚠️ 內部工具，客戶端 config/app.py 不會用到這支程式。
    """

    def __init__(
        self,
        task: str,
        ver: str,
        source_path: str,
        loop: bool = False,
        fps: float = 15.0,
        accuracy_mode: bool = False,
    ):
        """
        accuracy_mode:
            False (預設) -> 重現線上行為，佇列滿了會丟舊幀，consumer 跟不上就跳幀。
                            適合拿預錄影片模擬線上運行、測穩定性/reset 邏輯。
            True         -> 逐幀驗證模式，佇列滿了會阻塞等待 consumer 消化完，
                            保證每一幀都會進到 mmr/logic/ocr。適合對預錄資料
                            做準確率驗證 (不可漏幀)。
        """
        self._source_path = source_path
        self._loop = loop
        self._offline_fps = fps
        self._accuracy_mode = accuracy_mode
        super().__init__(task, ver)

    # ---------- 覆寫:相機控制器改用 OfflineCameraController ----------
    def _build_camera_controller(self) -> CameraController:
        return OfflineCameraController(
            cfg=self.task.cfg,
            bus=self.task.bus,
            source_path=self._source_path,
            loop=self._loop,
            fps=self._offline_fps,
            accuracy_mode=self._accuracy_mode,
            video_recorder=self.video_recorder,
            camera_alone=self.camera_alone,
            video_record_size=self.video_record_size,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Offline 離線測試工具 (內部限定使用)")
    parser.add_argument(
        "source_path",
        type=str,
        help="離線影像來源資料夾路徑 (內含影片或圖片序列)，或單一影片檔案路徑",
    )
    parser.add_argument(
        "--task", type=str, default="ocr", help="任務名稱 (對應 tasks/<task>/config.yaml)"
    )
    parser.add_argument("--version", type=str, default="0.0", help="版本號 (僅顯示用)")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="讀完/播完後是否從頭重來 (預設不 loop，讀完即結束)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="圖片序列模式下模擬送幀的節奏 (每秒幾張)，影片模式不使用此參數",
    )
    parser.add_argument(
        "--accuracy-mode",
        action="store_true",
        help="逐幀驗證模式：不丟幀，佇列滿了會阻塞等待，確保每一幀都被處理 "
        "(適合對預錄資料驗證準確率；不加此參數則重現線上跳幀行為)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_dir = Path(__file__).resolve().parent / "logs" / "offline_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(str(log_dir / "{time:YYYYMMDD}.log"), level="INFO", rotation="00:00")
    logger.add(sys.stderr, level="INFO")

    manager = None

    def graceful_exit(signum, frame):
        logger.info(f"收到訊號 {signum}，準備退出...")
        if manager:
            manager._running = False

    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGINT, graceful_exit)

    try:
        manager = OfflineStreamManager(
            task=args.task,
            ver=args.version,
            source_path=args.source_path,
            loop=args.loop,
            fps=args.fps,
            accuracy_mode=args.accuracy_mode,
        )
        manager.start_camera()
        manager.stream_frames()
    except:
        logger.error(traceback.format_exc())
    finally:
        logger.info("執行清理作業 (Cleanup)...")
        if manager:
            try:
                manager.cleanup()
            except Exception as e:
                logger.error(f"Cleanup 失敗: {e}")


if __name__ == "__main__":
    main()