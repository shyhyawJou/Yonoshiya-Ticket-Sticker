from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Optional

import cv2
from loguru import logger

from stream_vision.hikcam import HikCamera


class CameraController:
    """
    封裝相機的開啟/取像/重啟/關閉。

    對外只暴露：start / stop / reload / is_alive / get_frame / read，
    呼叫端 (StreamManager) 不需要知道底層是「獨立 thread + queue」
    還是「主迴圈同步 read()」。

    - camera_alone=True  : 內部開一條 worker thread 持續取像，frame 放進
                            queue，呼叫端用 get_frame() 拿最新一幀（非阻塞
                            佇列滿了就丟舊幀）。
    - camera_alone=False : 不開 thread，呼叫端自己在主迴圈用 read() 同步
                            拉幀（沿用舊版 else 分支的行為）。
    """

    def __init__(self, cfg, bus, camera_alone: bool = True,
                 video_record_size: tuple[int, int] = (1280, 960)):
        self.cfg = cfg
        self.bus = bus
        self.camera_alone = camera_alone
        self.video_record_size = video_record_size

        self.capture: Any = None
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=1)

        self._stop_event = threading.Event()
        self._running = False
        self.is_reloading = False  # 單純的公開 bool 屬性，不用 @property。
        # 主迴圈每一輪都會檢查這個值（在抓 frame 之前就先檢查），
        # @property 會多一次 descriptor protocol 的函式呼叫，在較弱的
        # edge CPU 上這種高頻率的小開銷值得省下來。

    # ---------- 對外狀態 ----------

    def is_alive(self) -> bool:
        """相機目前是否還在正常運作（camera_alone 看 thread，否則看 capture）"""
        if self.camera_alone:
            return self.capture_thread is not None and self.capture_thread.is_alive()
        return self.capture is not None and self.capture.isOpened()

    # ---------- 啟動 / 停止 ----------

    def start(self) -> None:
        self._running = True
        if self.camera_alone:
            self.capture_thread = threading.Thread(target=self._worker, daemon=True)
            self.capture_thread.start()
        else:
            self._open()

    def stop(self) -> None:
        """停止 worker thread（若有）並釋放相機資源"""
        self._running = False
        self._stop_event.set()

        if self.camera_alone and self.capture_thread is not None:
            self.capture_thread.join(timeout=5)
            if self.capture_thread.is_alive():
                logger.warning("[CAMERA] camera worker thread 未在時限內結束")

        if self.capture is not None:
            try:
                self.capture.release()
                logger.info("Camera stopped.")
            except Exception as e:
                logger.error(f"Stop camera error: {e}")
            finally:
                self.capture = None

    def reload(self) -> None:
        """
        安全地重啟相機（只重啟 worker thread + capture，
        呼叫端的其他資源 (MQTT / OCR / streamer / 主迴圈) 不受影響）。
        """
        if self.is_reloading:
            logger.warning("[CAMERA] 目前正在重啟相機中，略過重複請求")
            return

        self.is_reloading = True
        logger.info("[CAMERA] 準備重啟相機以套用新參數...")

        try:
            if self.camera_alone and self.capture_thread is not None:
                self._stop_event.set()
                self.capture_thread.join(timeout=5)
                if self.capture_thread.is_alive():
                    logger.warning("[CAMERA] camera worker thread 未在時限內結束")

            if self.capture is not None:
                try:
                    self.capture.release()
                except Exception as e:
                    logger.error(f"[CAMERA] 釋放舊相機失敗: {e}")
                finally:
                    self.capture = None

            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self._stop_event.clear()
            if self.camera_alone:
                self.capture_thread = threading.Thread(target=self._worker, daemon=True)
                self.capture_thread.start()
            else:
                self._open()

            logger.success("[CAMERA] 相機已成功重啟，新參數已套用")

        except Exception:
            logger.error(f"[CAMERA] 相機重啟失敗: {traceback.format_exc()}")
        finally:
            self.is_reloading = False

    # ---------- 取像 ----------

    def get_frame(self, timeout: float = 1.0):
        """camera_alone 模式專用：從 queue 拿最新一幀，逾時回傳 None"""
        try:
            return self.frame_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def read(self):
        """非 camera_alone 模式專用：直接同步呼叫 capture.read()"""
        if self.capture is None:
            return False, None
        return self.capture.read()

    # ---------- 硬體參數（透傳給底層 capture）----------

    def reset_camera_parameters(self) -> None:
        if self.capture is not None:
            self.capture.reset_camera_parameters()

    def set_camera_parameters(self, payload: dict, save: bool = True) -> None:
        if self.capture is not None:
            self.capture.set_camera_parameters(payload, save)

    def save_camera_config(self) -> None:
        if self.capture is not None:
            self.capture._save_camera_config()

    # ---------- 內部 ----------

    def _open(self) -> None:
        if self.cfg.runtime.camera.device == 'hik':
            logger.info("使用工業相機 (HIK)...")
            self.capture = HikCamera(self.cfg.runtime.camera.source, self.cfg, self.bus)
        elif self.cfg.runtime.camera.device == 'aravis':
            ctrl_dict = self.cfg.camera_params['aravis']
            features_str = " ".join([f"{k}={v}" for k, v in ctrl_dict.items()])
            logger.info(f"使用工業相機 (Aravis)..., feature str: {features_str}")
            gst_str = (
                f'aravissrc features="{features_str}" ! '
                'bayer2rgb ! '
                'queue ! '
                'videoconvert ! '
                f'video/x-raw,format=BGR,width={self.cfg.runtime.camera.width},height={self.cfg.runtime.camera.height} ! '
                'queue ! '
                'appsink drop=true max-buffers=1 sync=false'
            )
            self.capture = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        else:
            raise ValueError('valid camera are "aravis" and "hik"')

        if not self.capture.isOpened():
            raise ValueError("Failed to open camera.")

        logger.success(f'opened [{self.cfg.runtime.camera.device}] camera !')

    def _enqueue_frame(self, frame) -> None:
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def _worker(self) -> None:
        """獨立的相機取像執行緒 (Producer)"""
        try:
            self._open()
            logger.info("Camera Worker 啟動")

            consecutive_errors = 0
            MAX_RETRIES = 30

            while (
                self._running
                and not self._stop_event.is_set()
                and self.capture is not None
                and self.capture.isOpened()
            ):
                try:
                    ok, frame = self.capture.read()
                    if not ok:
                        consecutive_errors += 1
                        if consecutive_errors > MAX_RETRIES:
                            logger.error("相機連續讀取失敗次數過多，已斷線")
                            self._running = False
                            break

                        time.sleep(0.1)
                        continue

                    consecutive_errors = 0
                    self._enqueue_frame(frame)

                    time.sleep(0.03)

                except Exception:
                    logger.error(f"Camera worker error: {traceback.format_exc()}")
                    time.sleep(0.1)
        except Exception:
            logger.error(f"Camera worker error: {traceback.format_exc()}")
        finally:
            if self.capture is not None:
                self.capture.release()
            logger.info("Camera Worker 結束")