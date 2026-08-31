from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Optional

import cv2
import numpy as np
from loguru import logger

try:
    from pyorbbecsdk import (
        Config,
        OBFormat,
        OBSensorType,
        Pipeline,
    )
    _ORBBEC_SDK_AVAILABLE = True
except ImportError:
    # 相機環境（pyorbbecsdk）還沒裝好時，先讓這支檔案可以被 import，
    # 實際呼叫 start() 時才會丟出清楚的錯誤訊息，而不是在 import 階段就整包壞掉。
    _ORBBEC_SDK_AVAILABLE = False


class OrbbecDepthCameraError(RuntimeError):
    """Orbbec 深度相機相關的錯誤（開啟失敗、SDK 未安裝、逾時...等）"""


class OrbbecDepthCamera:
    """
    封裝奧比中光 (Orbbec) Gemini-EW 深度相機的開啟 / 取像 / 重啟 / 關閉，
    以及深度圖 -> 偽彩色 RGB 的可視化渲染。

    設計上刻意跟現有 RGB 相機的 CameraController 保持相同的對外介面
    (start / stop / reload / is_alive / get_frame)，方便之後要接進同一套
    StreamManager 或跟 RGB 相機並行運作時，呼叫端不需要另外分支處理。

    目前只負責「把相機開起來、拿到深度圖 (mm)、渲染成可視化的 BGR 圖」。
    之後 RGB bbox -> 深度相機座標轉換、bbox 內深度統計 (平均/中位數/離群值
    剔除) 這些功能，等你決定好要放在哪支 py 檔後再接上來 —— 這支 class
    只提供最基礎的 get_frame() 拿到的深度 mm 陣列，那些統計功能可以直接
    重用 depth_bbox_stats_tool.py 裡 compute_bbox_stats() 那套邏輯，
    輸入格式是一樣的 (H, W) float32 mm、0 代表無效值。

    Parameters
    ----------
    width, height, fps:
        深度串流解析度與幀率。不指定 (None) 的話會用相機回報的預設 profile。
    min_depth_mm, max_depth_mm:
        渲染成偽彩色圖時的顯示範圍（同時也用來過濾明顯不合理的深度值）。
        不指定的話，渲染當下會用 auto_range() 依實際深度分佈自動決定。
    colormap:
        cv2 的 colormap，預設 COLORMAP_JET（跟你之前 depth_bbox_stats_tool.py
        用的一樣，方便報告圖風格統一）。
    frame_timeout_ms:
        pipeline.wait_for_frames() 的逾時時間 (ms)。
    """

    def __init__(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        min_depth_mm: Optional[float] = 20.0,
        max_depth_mm: Optional[float] = 10000.0,
        colormap: int = cv2.COLORMAP_JET,
        frame_timeout_ms: int = 100,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.min_depth_mm = min_depth_mm
        self.max_depth_mm = max_depth_mm
        self.colormap = colormap
        self.frame_timeout_ms = frame_timeout_ms

        self.pipeline: Any = None
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=1)

        self._stop_event = threading.Event()
        self._running = False
        self.is_reloading = False

    # ---------- 對外狀態 ----------

    def is_alive(self) -> bool:
        return self.capture_thread is not None and self.capture_thread.is_alive()

    # ---------- 啟動 / 停止 ----------

    def start(self) -> None:
        if not _ORBBEC_SDK_AVAILABLE:
            raise OrbbecDepthCameraError(
                "找不到 pyorbbecsdk，請先安裝 Orbbec 官方 Python SDK "
                "(https://github.com/orbbec/pyorbbecsdk) 再啟動這台相機。"
            )
        self._running = True
        self.capture_thread = threading.Thread(target=self._worker, daemon=True)
        self.capture_thread.start()

    def stop(self) -> None:
        """停止 worker thread 並釋放相機資源"""
        self._running = False
        self._stop_event.set()

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=5)
            if self.capture_thread.is_alive():
                logger.warning("[ORBBEC] depth camera worker thread 未在時限內結束")

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                logger.info("Orbbec depth camera stopped.")
            except Exception as e:
                logger.error(f"Stop orbbec depth camera error: {e}")
            finally:
                self.pipeline = None

    def reload(self) -> None:
        """安全地重啟深度相機（同 RGB 相機的 CameraController.reload 邏輯）"""
        if self.is_reloading:
            logger.warning("[ORBBEC] 目前正在重啟深度相機中，略過重複請求")
            return

        self.is_reloading = True
        logger.info("[ORBBEC] 準備重啟深度相機...")

        try:
            if self.capture_thread is not None:
                self._stop_event.set()
                self.capture_thread.join(timeout=5)
                if self.capture_thread.is_alive():
                    logger.warning("[ORBBEC] depth camera worker thread 未在時限內結束")

            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    logger.error(f"[ORBBEC] 釋放舊 pipeline 失敗: {e}")
                finally:
                    self.pipeline = None

            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break

            self._stop_event.clear()
            self.capture_thread = threading.Thread(target=self._worker, daemon=True)
            self.capture_thread.start()

            logger.success("[ORBBEC] 深度相機已成功重啟")

        except Exception:
            logger.error(f"[ORBBEC] 深度相機重啟失敗: {traceback.format_exc()}")
        finally:
            self.is_reloading = False

    # ---------- 取像 ----------

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        從 queue 拿最新一幀深度圖。

        Returns
        -------
        depth_mm: (H, W) float32 ndarray，單位 mm，0 代表無效值。逾時回傳 None。
        """
        try:
            return self.frame_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    # ---------- 深度圖 -> 偽彩色 RGB 可視化 ----------

    @staticmethod
    def auto_range(depth_mm: np.ndarray, low_percentile: float = 0, high_percentile: float = 100):
        """依實際深度分佈（排除 0）自動抓顯示用的 min/max range，單位 mm。"""
        valid = depth_mm[depth_mm > 0]
        if valid.size == 0:
            raise ValueError("沒有有效的深度值 (全部是 0)")
        lo = float(np.percentile(valid, low_percentile))
        hi = float(np.percentile(valid, high_percentile))
        return lo, hi

    def colorize(self, depth_mm: np.ndarray, min_depth_mm: Optional[float] = None,
                 max_depth_mm: Optional[float] = None) -> np.ndarray:
        """
        把 (H, W) float32 mm 的深度圖轉成 BGR 偽彩色圖 (uint8)。
        無效值 (<=0) 強制顯示黑色，跟 depth_bbox_stats_tool.py 裡的邏輯一致。

        min_depth_mm / max_depth_mm 不指定的話，會用建構子傳入的值；
        兩者都沒有的話，用 auto_range() 依這一幀實際分佈自動決定。
        """
        lo = min_depth_mm if min_depth_mm is not None else self.min_depth_mm
        hi = max_depth_mm if max_depth_mm is not None else self.max_depth_mm
        if lo is None or hi is None:
            lo, hi = self.auto_range(depth_mm)

        invalid_mask = depth_mm <= 0
        clipped = np.clip(depth_mm, lo, hi)
        denom = max(hi - lo, 1e-6)
        normalized = ((clipped - lo) / denom * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(normalized, self.colormap)
        colored_bgr[invalid_mask] = (0, 0, 0)
        return colored_bgr

    def get_colorized_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """方便直接拿「最新一幀的偽彩色可視化圖」，內部就是 get_frame() + colorize()。"""
        depth_mm = self.get_frame(timeout=timeout)
        if depth_mm is None:
            return None
        return self.colorize(depth_mm)

    # ---------- 內部 ----------

    def _open(self) -> None:
        config = Config()
        pipeline = Pipeline()

        # TODO: 多台 Orbbec 相機時，需要用 Context().query_devices() 依序號
        # 選定特定裝置再建立 Pipeline(device)。目前先假設只接一台，用預設
        # 抓到的第一台裝置。等實機環境架好、確認你安裝的 pyorbbecsdk 版本
        # 的裝置列舉 API 後再補上。

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if self.width and self.height:
            fps = self.fps or 30
            depth_profile = depth_profile_list.get_video_stream_profile(
                self.width, self.height, OBFormat.Y16, fps
            )
        else:
            depth_profile = depth_profile_list.get_default_video_stream_profile()

        config.enable_stream(depth_profile)
        pipeline.start(config)

        self.pipeline = pipeline
        logger.success(
            f"opened Orbbec Gemini-EW depth camera "
            f"[{depth_profile.get_width()}x{depth_profile.get_height()}] !"
        )

    def _enqueue_frame(self, frame: np.ndarray) -> None:
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def _worker(self) -> None:
        """獨立的深度相機取像執行緒 (Producer)"""
        try:
            self._open()
            logger.info("Orbbec Depth Camera Worker 啟動")

            consecutive_errors = 0
            MAX_RETRIES = 30

            while self._running and not self._stop_event.is_set() and self.pipeline is not None:
                try:
                    frames = self.pipeline.wait_for_frames(self.frame_timeout_ms)
                    if frames is None:
                        consecutive_errors += 1
                        if consecutive_errors > MAX_RETRIES:
                            logger.error("深度相機連續讀取失敗次數過多，已斷線")
                            self._running = False
                            break
                        continue

                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        continue

                    if depth_frame.get_format() != OBFormat.Y16:
                        logger.warning("[ORBBEC] depth frame format 不是 Y16，略過此幀")
                        continue

                    w = depth_frame.get_width()
                    h = depth_frame.get_height()
                    scale = depth_frame.get_depth_scale()

                    depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((h, w))
                    depth_mm = depth_raw.astype(np.float32) * scale

                    # 過濾掉明顯不合理的深度值（沿用你 tool 裡的無效值=0 慣例）
                    if self.min_depth_mm is not None and self.max_depth_mm is not None:
                        valid = (depth_mm > self.min_depth_mm) & (depth_mm < self.max_depth_mm)
                        depth_mm = np.where(valid, depth_mm, 0).astype(np.float32)

                    consecutive_errors = 0
                    self._enqueue_frame(depth_mm)

                except Exception:
                    logger.error(f"Orbbec depth camera worker error: {traceback.format_exc()}")
                    time.sleep(0.1)
        except Exception:
            logger.error(f"Orbbec depth camera worker error: {traceback.format_exc()}")
        finally:
            if self.pipeline is not None:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            logger.info("Orbbec Depth Camera Worker 結束")


if __name__ == "__main__":
    # 簡單的手動測試腳本：開相機、拿一幀、存一張偽彩色圖看結果。
    cam = OrbbecDepthCamera()
    cam.start()
    try:
        depth_mm = cam.get_frame(timeout=5.0)
        if depth_mm is None:
            print("沒拿到深度圖，請確認相機連線與 pyorbbecsdk 安裝狀況")
        else:
            colored = cam.colorize(depth_mm)
            cv2.imwrite("orbbec_depth_preview.png", colored)
            print(f"depth shape={depth_mm.shape}, 已存檔 orbbec_depth_preview.png")
    finally:
        cam.stop()