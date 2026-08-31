from __future__ import annotations

import sys
import signal
import time
import threading
import traceback
from pathlib import Path

import cv2
from loguru import logger

from utils.clip_video import Clip, PeriodicClipRecorder
from utils.frame_writer import AsyncFrameWriter

from stream_vision.streamer import Mjpeg_Streamer

from manager.camera_controller import CameraController
from manager.task_context import TaskContext
from manager.overlay_renderer import OverlayRenderer
from manager.command_dispatcher import CommandDispatcher


class StreamManager:
    """
    協調者：組裝 CameraController / TaskContext / OverlayRenderer /
    CommandDispatcher，並負責跑主迴圈 (stream_frames)。

    不再自己實作相機 thread、指令分派、畫面繪製的細節 —— 那些邏輯
    都下沉到 manager/ 底下對應的類別裡，這裡只保留「串起整條流水線」
    的協調邏輯。
    """

    def __init__(self, task: str, ver: str):
        self.base_dir = Path(__file__).resolve().parent

        self.data_lock = threading.RLock()

        # --- Flag vars ---
        self.show_box: bool = True
        self.show_fps: bool = True
        self.camera_alone: bool = True
        self._running: bool = False

        # --- 主迴圈共享狀態 ---
        self.ocr_data = None
        self.tmp_frame = None

        # --- 錄影 ---
        self.save_dir = Path("/mnt/reserved/record/stream")
        self.video_record_size = (1280, 960)
        self.event_record_max_seconds = 600 # 600 秒如果 event 都沒觸發 stop 就主動觸發
        self._event_record_start_time = None
        self.event_recorder = Clip(
            root_dir=str(self.save_dir),      # 跟 fullscreen/ocr 同一層基準路徑
            enable=True,
            tag="event",
            category="event",
            crf=23,
            bitrate=4,
            fps=15,
            frame_size=None,
            preroll_frames=20,
            postroll_frames=20,
        )

        self.continuous_recorder = PeriodicClipRecorder(
            clip=Clip(
                root_dir=str(self.save_dir),  # 同一個 root_dir
                enable=True,
                tag="continuous",
                category="continuous",
                crf=23,
                bitrate=4,
                fps=15,
                frame_size=None,
                preroll_frames=0,
                postroll_frames=0,
            ),
            periods=[
                (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
                (16, 17), (17, 18), (18, 19), (19, 20),
            ],
        )

        # --- 存圖 ---
        self.frame_writer = AsyncFrameWriter(max_queue_size=30)

        # --- 任務相關物件 (cfg/mmr/logic/ocr/bus) ---
        self.task = TaskContext(self.base_dir)
        self.task.load_task(
            task,
            on_recording_start=self.start_event_recording,
            on_recording_stop=self.stop_event_recording,
        )
        self.task.bus.on_command(self._handle_cmd)

        # --- 相機 ---
        self.camera = self._build_camera_controller()

        # --- 畫面繪製 ---
        self.overlay = OverlayRenderer(self.task)

        # --- 指令分派 ---
        self.commands = CommandDispatcher(self)

        # --- 串流 ---
        self.stream_mode = "detect"
        self.stream_size = self.task.cfg.runtime.stream.stream_size
        self.streamer = Mjpeg_Streamer(
            route='/meal',
            port=self.task.cfg.runtime.stream.port,
            size=self.stream_size,
            quality=40
        )
        self.streamer.start()

        logger.info(f"StreamManager v{ver} 初始化成功!")

    def _build_camera_controller(self) -> CameraController:
        """
        建立要用的 CameraController。

        獨立成一個方法是為了讓子類別（例如離線測試用的
        OfflineStreamManager）可以覆寫成回傳不同的 CameraController
        子類別（例如換掉取像來源），不用整個重寫 __init__。
        """
        return CameraController(
            cfg=self.task.cfg,
            bus=self.task.bus,
            camera_alone=self.camera_alone,
            video_record_size=self.video_record_size,
        )
    
    def start_event_recording(self, reason: str = "") -> None:
        with self.data_lock:
            self.event_recorder.start()
        logger.info(f"[event_recorder] 開始錄影 ({reason})")

    def stop_event_recording(self, save_video: bool = True, reason: str = "") -> None:
        with self.data_lock:
            self.event_recorder.request_stop(save_video=save_video)
        logger.info(f"[event_recorder] 請求結束錄影 ({reason}), save_video={save_video}")

    def _handle_cmd(self, cmd: str, payload: dict) -> None:
        self.commands.handle(cmd, payload)

    def _save_frame(self, frame, folder_name: str) -> None:
        try:
            self.frame_writer.save(frame.copy(), self.save_dir, folder_name)

        except Exception as e:
            logger.error(f"儲存影像失敗 (Folder: {folder_name}): {e}")

    def _check_event_record_timeout(self, now_ts):
        if not self.event_recorder.is_running:
            self._event_record_start_time = None
            return

        if self._event_record_start_time is None:
            self._event_record_start_time = now_ts
            return

        elapsed = now_ts - self._event_record_start_time

        if elapsed >= self.event_record_max_seconds:
            logger.error(
                f"[EVENT RECORDER] "
                f"錄影超過 {self.event_record_max_seconds}s，"
                "啟動 watchdog 強制停止！"
            )

            try:
                self.stop_event_recording(reason=f"event time out !")
            except Exception:
                logger.error(
                    f"[EVENT RECORDER] watchdog stop failed:\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                self._event_record_start_time = None

    def start_camera(self) -> None:
        self.camera.start()
        self._running = True
        logger.info("Camera started. Streaming...")

    def stream_frames(self) -> None:
        """Main loop"""

        fps_counter = 0
        display_fps = "fps: 0.0"
        time_1 = time.time()

        # --- 熱路徑快取 ---
        # self.task.mmr / self.task.logic / self.task.ocr 這種多層屬性鏈
        # 在 CPython 裡每一層 "." 都是一次 LOAD_ATTR，主迴圈每秒跑十幾二十次、
        # 每次又用到好幾次，累積起來會比原本 self.mmr 這種單層查詢慢。
        # 這裡在迴圈外先快取成區域變數 (LOAD_FAST，比 LOAD_ATTR 快很多)。
        # 注意：如果之後 no_tray_setting 切換模式會重建 self.task.logic，
        # 這裡的 logic 參照就會過期，所以每次迴圈開頭都重新快取一次
        # （比每次都走三層屬性鏈划算，因為單層 self.task 查詢只需要一次）。
        camera = self.camera
        overlay = self.overlay

        while self._running:

            task_ctx = self.task
            mmr = task_ctx.mmr
            logic = task_ctx.logic
            ocr = task_ctx.ocr

            if camera.is_reloading:
                time.sleep(0.1)
                continue

            if self.camera_alone and not camera.is_alive():
                logger.error("偵測到 Camera Thread 已停止，主程式即將退出...")
                self._running = False
                break

            try:
                if self.camera_alone:
                    frame = camera.get_frame(timeout=1.0)
                    if frame is None:
                        continue
                    self.tmp_frame = frame.copy()
                else:
                    ok, frame = camera.read()
                    if not ok:
                        logger.error("Failed to capture frame.")
                        break

                record_frame = cv2.resize(frame, self.video_record_size)
                now_ts = time.time()
                self.event_recorder.feed(record_frame, now_ts) # 事件觸發影片
                self.continuous_recorder.feed(record_frame, now_ts) # 時間內持續寫入影片
                self._check_event_record_timeout(now_ts)

                for ocr_result in task_ctx.drain_ocr_results():
                    metadata = ocr_result["metadata"]
                    with self.data_lock:
                        self.ocr_data = ocr_result
                        logic.apply_ocr_result(
                            tray_id=metadata["tray_id"],
                            item_type=metadata["type"],
                            frame_crop=ocr_result["frame_crop"],
                            dt_boxes=ocr_result["dt_boxes"],
                            rec_res=ocr_result["rec_res"],
                            is_flip=ocr_result["is_flip"],
                            task_bbox=metadata["bbox"]
                        )

                out, detections = mmr.detect(frame, self.show_box)

                with self.data_lock:
                    tasks, movements = logic.update(frame, detections)

                for task in tasks:
                    if not ocr.is_busy:

                        cx, cy, w, h, r = task["xywhr"]
                        warped_img, M_inv, M = mmr.crop_by_angle(frame, cx, cy, w, h, r)
                        if warped_img.size == 0 or w < 10 or h < 10:
                            continue

                        self._save_frame(warped_img, f"{str(self.save_dir)}/{self.event_recorder.date}/ocr_cut")
                        self._save_frame(self.tmp_frame, f"{str(self.save_dir)}/{self.event_recorder.date}/full_screen")

                        metadata = {
                            "tray_id": task["tray_id"],
                            "type": task["type"],
                            "M": M,
                            "M_inv": M_inv,
                            "bbox": task["bbox"],
                            "cls_name": task['cls_name']
                        }

                        ocr.request_ocr(frame_crop=warped_img, metadata=metadata)

                        with self.data_lock:
                            logic.set_ocr_busy(tray_id=task["tray_id"], item_type=task["type"], bbox=task["bbox"])

                        break

                current_1 = time.time()
                fps_counter += 1
                time_diff = current_1 - time_1
                if time_diff > 1.0:
                    fps = fps_counter / time_diff
                    display_fps = f"fps: {fps:.1f}"
                    fps_counter = 0
                    time_1 = current_1

                with self.data_lock:
                    current_ocr_data = self.ocr_data

                out, keep_ocr_data = overlay.draw(out, current_ocr_data)
                if not keep_ocr_data:
                    with self.data_lock:
                        self.ocr_data = None

                if self.show_fps:
                    cv2.putText(out, display_fps, (self.task.cfg.runtime.camera.width-650, self.task.cfg.runtime.camera.height-350), cv2.FONT_HERSHEY_SIMPLEX, 4, (194, 244, 255), 4)

                if self.stream_mode == "detect":
                    stream_out = cv2.resize(out, self.stream_size)
                    self.streamer.push_frame(stream_out)
                else:
                    self.streamer.push_frame(out)

            except Exception:
                logger.error(traceback.format_exc())
                break

    def stop_stream(self):
        if self.streamer is not None:
            try:
                self.streamer.stop()
                logger.info("MJPEG streamer stopped.")
            except Exception as e:
                logger.error(f"Stop streamer error: {e}")
            except BaseException as e:
                if type(e).__name__ == "GracefulExit":
                    logger.info("捕獲到 aiohttp GracefulExit，這是正常的關閉流程。")
                else:
                    logger.error(f"Stop streamer BaseException: {e}")
                    raise e
            finally:
                self.streamer = None
                logger.info("MJPEG streamer -> None.")
        else:
            pass

    def cleanup(self) -> None:
        """Release resources and disconnect MQTT cleanly."""
        self._running = False
        time.sleep(0.5)

        if self.frame_writer:
            self.frame_writer.stop(wait=True)
        if self.event_recorder.is_running:
            with self.data_lock:
                self.event_recorder.stop(save_video=True)
            logger.info("[event_recorder] cleanup 強制停止錄影")

        self.continuous_recorder.stop(save_video=True, wait_for_encode=True, encode_timeout=5.0,)
        logger.info("[continuous_recorder] cleanup 強制停止錄影 (最多等 5 秒)")

        self.task.stop()
        self.camera.stop()
        self.stop_stream()
        logger.info("Cleaned up.")


def main():

    log_dir = Path(__file__).resolve().parent / "logs" / "stream_log"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        str(log_dir / "{time:YYYYMMDD}.log"),
        level='INFO',
        rotation="00:00"
    )
    logger.add(
        sys.stderr,
        level='INFO'
    )

    task = "ocr"
    version = "0.0"

    manager = None

    def graceful_exit(signum, frame):
        logger.info(f"收到訊號 {signum}，準備退出...")
        if manager:
            manager._running = False

    signal.signal(signal.SIGTERM, graceful_exit)
    signal.signal(signal.SIGINT, graceful_exit)

    try:
        manager = StreamManager(task, version)
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