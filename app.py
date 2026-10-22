from __future__ import annotations

import fcntl
import os
import sys
import signal
import time
import threading
import traceback
from pathlib import Path
import numpy as np

import cv2
from loguru import logger

from utils.clip_video import Clip, PeriodicClipRecorder
from utils.frame_writer import AsyncFrameWriter
from utils import clock_guard
from utils.disk_guard import DiskGuard
from utils.perf_stats import LOOP_PROFILER, ocr_done_line

from stream_vision.streamer import Mjpeg_Streamer
from depth import ORBBEC_CAMERA

from manager.camera_controller import CameraController
from manager.task_context import TaskContext
from manager.overlay_renderer import OverlayRenderer
from manager.command_dispatcher import CommandDispatcher
from manager.cloud_uploader import CloudUploader
from logic.auto_roi_calibrator import AutoRoiCalibrator
from config import PresetRoiCfg

PROJECT_DIR = Path(__file__).resolve().parent

# 所有落地資料的共同基準路徑：影片、全螢幕/裁切圖、CSV 與 log 都掛在
# <RECORD_ROOT>/<YYYYMMDD>/<類別>/ 底下，同一筆訂單的素材才會待在一起。
RECORD_ROOT = Path("/mnt/reserved/record/stream")

# log 在日期資料夾底下的子目錄名。刻意跟 csv_logs 一樣用獨立子目錄，
# 因為 rclone 夜間搬檔要能用 --exclude "<這個名字>/**" 把「今天還開著的檔」
# 排除掉（見 install_crontab.sh 的說明）。
LOG_SUBDIR = "stream_log"


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
        self.save_dir = RECORD_ROOT
        self.video_record_size = (1280, 960)
        self.event_record_max_seconds = 600 # 600 秒如果 event 都沒觸發 stop 就主動觸發（載入設定後會依 order_guard 調整）
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
                (11, 12), (12, 13), (13, 14), (17, 18), (18, 19), (19, 20),
            ],
        )

        # 人工錄影
        self.manual_recorder = Clip(
            root_dir=str(self.save_dir),
            enable=True,
            tag="record",
            category="record",
            crf=23,
            bitrate=4,
            fps=15,
            frame_size=None,
            preroll_frames=0,
            postroll_frames=0,
        )

        # --- 殘留錄影掃描 ---
        # 沒成功轉成影片的 JPEG 資料夾（被 kill、斷電、encode 被跳過）以前會一直
        # 卡在當日資料夾裡等人工清。這裡開機先掃一次，之後每隔一段時間在背景補掃。
        self.orphan_scan_interval = 1800    # 30 分鐘
        self._last_orphan_scan = None    # time.monotonic() 的值；None = 還沒掃過
        self._orphan_scan_thread = None

        # --- 存圖 ---
        self.frame_writer = AsyncFrameWriter(max_queue_size=30)

        # --- 任務相關物件 (cfg/mmr/logic/ocr/bus) ---
        self.task = TaskContext(self.base_dir)
        self.task.load_task(
            task,
            on_recording_start=self.start_event_recording,
            on_recording_stop=self.stop_event_recording,
            get_video_filename=self.get_event_video_filename,
        )
        self.task.bus.on_command(self._handle_cmd)

        # 錄影 watchdog 只是保險：訂單本身有 order_guard.timeout_sec 超時收尾
        # （寫 CSV + 重置 + 停錄影），這裡刻意多留 30 秒，正常情況一定是訂單先收。
        # 以前只有這一道、而且只停影片不收訂單，超時後的訂單全部進不了 CSV。
        order_timeout = float(self.task.cfg.order_guard.timeout_sec or 0)
        if order_timeout > 0:
            self.event_record_max_seconds = order_timeout + 30

        # --- 相機 ---
        self.camera = self._build_camera_controller()
        self.depth_cam = ORBBEC_CAMERA(self.task.cfg.runtime.depth_cam.cfg)

        # --- preset_roi 自動校正（開機一次性，見 logic/auto_roi_calibrator.py）---
        self.roi_calibrators = self._build_roi_calibrators()

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

        # 開機先補一次上次沒收尾乾淨的錄影（丟到背景做，不擋啟動）
        self._schedule_orphan_scan(time.monotonic())

        # --- 雲端上傳（config 的 cloud_upload；不阻塞，失敗只停用上傳）---
        self.cloud_uploader = CloudUploader(
            self.task.cfg.cloud_upload, self.task.cfg.csv_log, self.base_dir
        )
        self.cloud_uploader.start()

        # --- 磁碟用量監控：/mnt/reserved 滿了 log/影片/CSV 會一起寫不進去，快滿先警告 ---
        self.disk_guard = DiskGuard(mount_path=RECORD_ROOT.parents[1], record_root=RECORD_ROOT)
        self.disk_guard.start()

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
    
    def _build_roi_calibrators(self) -> dict:
        """
        為設定檔裡每一個 preset_roi 區域各準備一個自動校正器。

        該 ROI 的 source == "auto" 時開機就 arm，一次性跑完就停；
        source == "manual" 時仍然建立物件（只是不啟動），這樣現場想臨時
        自動對一次，發 cmd/preset_roi {"action": "auto", "name": "..."}
        就能觸發，不用改設定檔重開。

        每個 ROI 各自一份、各自盯自己設定的類別（auto.class_name），
        所以以後 packing_roi 想用別的標記物自動對位，不用動這裡的程式。
        """
        preset_roi = getattr(self.task.cfg, "preset_roi", None)
        if preset_roi is None:
            return {}

        calibrators = {}
        for roi_name in PresetRoiCfg.ROI_FIELDS:
            roi = getattr(preset_roi, roi_name, None)
            if roi is None:
                continue

            calibrators[roi_name] = AutoRoiCalibrator(
                frame_width=self.task.cfg.runtime.camera.width,
                frame_height=self.task.cfg.runtime.camera.height,
                class_name=roi.auto.class_name,
                roi_name=roi_name,
                padding_ratio=roi.auto.padding_ratio,
                confirm_frames=roi.auto.confirm_frames,
                stable_iou=roi.auto.stable_iou,
                timeout_frames=roi.auto.timeout_frames,
            )
            if roi.source == "auto":
                calibrators[roi_name].arm()
            else:
                logger.info(f"[AUTO ROI] {roi_name}.source = manual，沿用設定檔既有的座標")

        return calibrators

    def apply_preset_roi(self, name: str, points, color=None, persist: bool = True,
                         source: str = "mqtt") -> bool:
        """
        套用一組 preset_roi 座標（App 的 cmd/preset_roi 或自動校正都走這裡）。

        實際邏輯在 TaskContext（它才擁有 cfg 與 logic），這裡只負責上
        data_lock —— 呼叫端可能是 MQTT callback thread，而 logic 正被主迴圈用著。
        """
        with self.data_lock:
            roi = self.task.apply_preset_roi(
                name=name, points=points, color=color, persist=persist, source=source
            )
        return roi is not None

    def _feed_roi_calibrators(self, detections) -> None:
        """把這一幀的偵測結果餵給還在跑的校正器；定案後直接套用並寫回設定檔。"""
        for roi_name, calibrator in self.roi_calibrators.items():
            if not calibrator.is_active:
                continue

            try:
                points = calibrator.feed(detections)
            except Exception:
                logger.error(
                    f"[AUTO ROI] {roi_name} 校正失敗，停止自動校正:\n{traceback.format_exc()}"
                )
                calibrator.cancel()
                continue

            if points is None:
                continue

            roi = getattr(self.task.cfg.preset_roi, roi_name, None)
            self.apply_preset_roi(
                name=roi_name,
                points=points,
                persist=roi.auto.persist if roi is not None else True,
                source=f"auto:{calibrator.class_name}",
            )

    def arm_roi_calibration(self, name: str = "ticket_roi", reason: str = "") -> None:
        calibrator = self.roi_calibrators.get(name)
        if calibrator is None:
            logger.error(
                f"[AUTO ROI] 設定檔的 preset_roi 底下沒有 '{name}'，無法自動校正"
                f"（現有的: {sorted(self.roi_calibrators)}）"
            )
            return
        logger.info(f"[AUTO ROI] 收到 {name} 重新校正要求 ({reason})")
        calibrator.arm()

    def cancel_roi_calibration(self, name: str = "ticket_roi", reason: str = "") -> None:
        calibrator = self.roi_calibrators.get(name)
        if calibrator is not None and calibrator.is_active:
            calibrator.cancel(reason=reason)

    def start_manual_recording(self, reason: str = "user") -> None:
        with self.data_lock:
            if self.manual_recorder.is_running:
                logger.warning("[RECORD] 已經在錄影中，忽略 start")
                return
            self.manual_recorder.start()

        start_ts = time.time()
        logger.info(f"[manual_recorder] 開始錄影 ({reason})")
        # path 這時候還不確定 (實際檔名要等第一幀寫入時才會定案，
        # 見 Clip._run)，所以只帶 start_time，App 端用 start_time 算
        # elapsed 就好，不依賴 path。
        self.task.bus.publish_record_status(recording=True, start_time=start_ts, reason=reason)

    def stop_manual_recording(self, reason: str = "user") -> None:
        with self.data_lock:
            if not self.manual_recorder.is_running:
                logger.warning("[RECORD] 目前沒有在錄影，忽略 stop")
                return
            self.manual_recorder.stop(save_video=True)
            video_path = self.manual_recorder.video_path  # stop 之後這個一定已經確定了

        logger.info(f"[manual_recorder] 結束錄影 ({reason})")
        self.task.bus.publish_record_status(
            recording=False,
            path=str(video_path) if video_path else None,
            reason=reason,
        )

    def start_event_recording(self, reason: str = "") -> None:
        with self.data_lock:
            # 一定要用 start() 的回傳值：錄影被拒絕(上一支還在 encode)時它會回
            # None，而 get_current_session_id() 會給出上一段的 session，CSV 就會
            # 指到上一單的影片。
            session_id = self.event_recorder.start()
        logger.info(f"[event_recorder] 開始錄影 ({reason}), session={session_id}")
        return session_id
    
    def stop_event_recording(self, save_video: bool = True, reason: str = "") -> None:
        with self.data_lock:
            self.event_recorder.request_stop(save_video=save_video)
        logger.info(f"[event_recorder] 請求結束錄影 ({reason}), save_video={save_video}")
    
    def get_event_video_filename(self, session_id: str) -> str:
        with self.data_lock:
            return self.event_recorder.get_video_filename_for_session(session_id)

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

    def _recover_orphan_clips(self) -> None:
        """
        逐一請每個 recorder 清理 / 補救自己 category 底下沒轉成影片的殘留資料夾。

        每個 Clip 只掃自己的 category（event / record / continuous），所以
        full_screen、ocr_cut 這種純圖片資料夾不會被碰到；正在錄影或正在
        encode 的 recorder 會自己跳過，不會跟現役的 ffmpeg 搶硬體編碼器。
        """
        for recorder in (
            self.event_recorder,
            self.manual_recorder,
            self.continuous_recorder.clip,
        ):
            try:
                recorder.recover_orphan_clips()
            except Exception:
                logger.error(
                    f"[{recorder.tag}] 殘留錄影掃描失敗:\n{traceback.format_exc()}"
                )

    def _schedule_orphan_scan(self, now_ts: float) -> None:
        """時間到了就在背景跑一次殘留錄影掃描（上一輪還沒跑完就跳過這次）"""
        # None = 還沒掃過：開機那一次一定要跑（monotonic 從開機起算，不能拿 0 當起點比）
        if (self._last_orphan_scan is not None
                and now_ts - self._last_orphan_scan < self.orphan_scan_interval):
            return

        if self._orphan_scan_thread is not None and self._orphan_scan_thread.is_alive():
            return

        self._last_orphan_scan = now_ts
        self._orphan_scan_thread = threading.Thread(
            target=self._recover_orphan_clips,
            name="orphan-clip-scan",
            daemon=True,
        )
        self._orphan_scan_thread.start()

    def start_camera(self) -> None:
        self.camera.start()
        self.depth_cam.start()
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
        depth_cam = self.depth_cam
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
                # 每段耗時 (ms)，彙總成每 60 秒一行的 [效能]（utils/perf_stats.py）
                t_loop = t_prev = time.perf_counter()
                stage = {}

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
                t_now = time.perf_counter(); stage["camera"] = 1000 * (t_now - t_prev); t_prev = t_now

                # 深度相機
                depth_frame = depth_cam.get_frame(timeout=0.1)
                t_now = time.perf_counter(); stage["depth"] = 1000 * (t_now - t_prev); t_prev = t_now

                record_frame = cv2.resize(frame, self.video_record_size)
                now_ts = time.time()
                # 經過時間一律用 monotonic：跳電後 RTC 回到預設時間、chrony 校時把時鐘
                # 往前撥一年多時，用牆上時鐘算會讓 watchdog 以為已錄了一年、立刻強制收尾
                # （2026-10-02 現場：start_time=2025-05-30 的那筆 NGtoOK）。
                mono_ts = time.monotonic()
                self.event_recorder.feed(record_frame, now_ts) # 事件觸發影片
                self.manual_recorder.feed(record_frame, now_ts)   # 人工觸發錄影
                # self.continuous_recorder.feed(record_frame, now_ts) # 時間內持續寫入影片
                self._check_event_record_timeout(mono_ts) # 檢查事件觸發錄影是否超時 (10分鐘)
                self._schedule_orphan_scan(mono_ts) # 檢查有沒有壞檔, 有的話嘗試補救
                t_now = time.perf_counter(); stage["record"] = 1000 * (t_now - t_prev); t_prev = t_now

                for ocr_result in task_ctx.drain_ocr_results():
                    metadata = ocr_result["metadata"]
                    drained_mono = time.monotonic()
                    # 數量欄 rec 在 apply 裡面（主執行緒）跑，前後相減才知道這次用了幾次
                    rec_pred = getattr(logic, "rec_pred", None)
                    qty0 = (getattr(rec_pred, "prof_calls", 0), getattr(rec_pred, "prof_ms", 0.0))
                    t_apply = time.perf_counter()
                    with self.data_lock:
                        self.ocr_data = ocr_result
                        logic.apply_ocr_result(
                            tray_id=metadata["tray_id"],
                            item_type=metadata["type"],
                            frame_crop=ocr_result["frame_crop"],
                            dt_boxes=ocr_result["dt_boxes"],
                            rec_res=ocr_result["rec_res"],
                            is_flip=ocr_result["is_flip"],
                            task_bbox=metadata["bbox"],
                            target=metadata.get("target"),
                        )
                        # 傳票的菜單區域下界（吉野家 parser 才有），跟著這份 OCR 結果給 overlay 畫
                        pop_line = getattr(logic.order_parser, "pop_menu_bottom_line", None)
                        ocr_result["menu_bottom_line"] = pop_line() if pop_line else None
                    apply_ms = 1000 * (time.perf_counter() - t_apply)
                    logger.info(ocr_done_line(
                        metadata, drained_mono, apply_ms,
                        getattr(rec_pred, "prof_calls", 0) - qty0[0],
                        getattr(rec_pred, "prof_ms", 0.0) - qty0[1],
                    ))
                    trace = metadata.get("trace") or {}
                    if trace.get("start_mono") is not None and trace.get("end_mono") is not None:
                        LOOP_PROFILER.add_ocr(1000 * (trace["end_mono"] - trace["start_mono"]))
                t_now = time.perf_counter(); stage["apply"] = 1000 * (t_now - t_prev); t_prev = t_now

                out, detections = mmr.detect(frame, self.show_box)
                stage.update(getattr(mmr, "last_timing", None) or {})
                t_prev = time.perf_counter()

                # preset_roi 自動校正：只在開機後那一小段（或收到 action=auto）
                # 期間會真的做事，定案/逾時之後 is_active 就永遠是 False，
                # 熱路徑上只剩每個 ROI 一次屬性判斷。
                self._feed_roi_calibrators(detections)

                with self.data_lock:
                    tasks, movements = logic.update(frame, detections)
                t_now = time.perf_counter(); stage["logic"] = 1000 * (t_now - t_prev); t_prev = t_now

                for task in tasks:
                    if not ocr.is_busy:

                        cx, cy, w, h, r = task["xywhr"]
                        warped_img, M_inv, M = mmr.crop_by_angle(frame, cx, cy, w, h, r)
                        if warped_img.size == 0 or w < 10 or h < 10:
                            continue

                        # self._save_frame(warped_img, f"{str(self.save_dir)}/{self.event_recorder.date}/ocr_cut")
                        self._save_frame(self.tmp_frame, f"{str(self.save_dir)}/{self.event_recorder.date}/full_screen")

                        metadata = {
                            "tray_id": task["tray_id"],
                            "type": task["type"],
                            "M": M,
                            "M_inv": M_inv,
                            "bbox": task["bbox"],
                            "cls_name": task['cls_name'],
                            # 計時：出現時間、移動重置次數、第幾次讀取；OCR 線程再補上
                            # 送出 / 開始 / 結束時間與各段耗時（utils/perf_stats.py）
                            "trace": dict(task.get("trace") or {}),
                            # 被讀的那個追蹤物件本身：標記 OCR 中、套用結果都認它，
                            # 不靠框重疊去找（疊在一起的貼紙會配錯，見 logic/readers/base.py::find_task_item）。
                            # OCR 線程不碰它，只是跟著結果帶回主迴圈
                            "target": task.get("target"),
                        }

                        ocr.request_ocr(frame_crop=warped_img, metadata=metadata)

                        with self.data_lock:
                            logic.set_ocr_busy(tray_id=task["tray_id"], item_type=task["type"], bbox=task["bbox"],
                                               target=task.get("target"))

                        break
                t_now = time.perf_counter(); stage["dispatch"] = 1000 * (t_now - t_prev); t_prev = t_now

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

                # 合併
                if out is not None and depth_frame is not None:
                    w = self.task.cfg.runtime.camera.width // 2
                    h = self.task.cfg.runtime.camera.height
                    out_depth = self.depth_cam.draw(depth_frame)
                    out = cv2.resize(out, (w, h))
                    out_depth = cv2.resize(out_depth, (w, h))
                    out = np.concatenate([out, out_depth], 1)

                if self.show_fps:
                    if depth_frame is not None:
                        w = self.task.cfg.runtime.camera.width // 2
                        h = self.task.cfg.runtime.camera.height
                        cv2.putText(out, display_fps, (w - 650, h - 350), cv2.FONT_HERSHEY_SIMPLEX, 4, (194, 244, 255), 4)
                    else:
                        cv2.putText(out, display_fps, (self.task.cfg.runtime.camera.width-650, self.task.cfg.runtime.camera.height-350), cv2.FONT_HERSHEY_SIMPLEX, 4, (194, 244, 255), 4)

                if self.stream_mode == "detect":
                    stream_out = cv2.resize(out, self.stream_size)
                    self.streamer.push_frame(stream_out)
                else:
                    self.streamer.push_frame(out)

                t_now = time.perf_counter(); stage["overlay"] = 1000 * (t_now - t_prev)
                LOOP_PROFILER.add_frame(stage, 1000 * (t_now - t_loop))
                LOOP_PROFILER.maybe_report()

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

        # 先停上傳：進行中的上傳會被中斷，不留背景執行緒拖住關機
        if getattr(self, "cloud_uploader", None) is not None:
            self.cloud_uploader.stop()
        if getattr(self, "disk_guard", None) is not None:
            self.disk_guard.stop()

        if self.frame_writer:
            self.frame_writer.stop(wait=True)
        if self.event_recorder.is_running:
            with self.data_lock:
                self.event_recorder.stop(save_video=True)
            logger.info("[event_recorder] cleanup 強制停止錄影")
        if self.manual_recorder.is_running:
            with self.data_lock:
                self.manual_recorder.stop(save_video=True)
            logger.info("[manual_recorder] cleanup 強制停止錄影")
            self.task.bus.publish_record_status(recording=False, reason="cleanup")

        self.continuous_recorder.stop(save_video=True, wait_for_encode=True, encode_timeout=5.0,)
        logger.info("[continuous_recorder] cleanup 強制停止錄影 (最多等 5 秒)")

        self.task.stop()
        self.camera.stop()
        self.depth_cam.stop()
        self.stop_stream()
        logger.info("Cleaned up.")


def _resolve_log_sink() -> str:
    """決定 log 檔要寫哪裡，回傳給 loguru 的路徑樣板。

    正常情況：
        <RECORD_ROOT>/<YYYYMMDD>/stream_log/<YYYYMMDD>.log
    跟影片 (event/continuous/record)、截圖 (full_screen/ocr_cut) 與
    csv_logs 放在同一個日期資料夾底下，事後要查某一天的狀況不必跨目錄找。

    路徑裡的 {time} 由 loguru 在「每次建立檔案」時重新格式化，**目錄部分也算**，
    所以午夜 rotation 會自動換到隔天的日期資料夾，不需要自己排程搬檔。

    /mnt/reserved 是獨立掛載的分割區。沒掛上時不應該讓整個 app 起不來，
    也不該在 rootfs 上偷偷造一個同名目錄把磁碟寫爆 —— 退回專案內的
    logs/stream_log，並留一行警告。
    """
    pattern = Path("{time:YYYYMMDD}") / LOG_SUBDIR / "{time:YYYYMMDD}.log"
    mount_root = RECORD_ROOT.parents[1]   # /mnt/reserved

    if mount_root.is_dir():
        try:
            RECORD_ROOT.mkdir(parents=True, exist_ok=True)
            return str(RECORD_ROOT / pattern)
        except OSError as e:
            logger.warning(f"[log] 無法寫入 {RECORD_ROOT} ({e})，log 改寫到專案目錄")
    else:
        logger.warning(f"[log] {mount_root} 未掛載，log 改寫到專案目錄")

    fallback_dir = PROJECT_DIR / "logs" / LOG_SUBDIR
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return str(fallback_dir / "{time:YYYYMMDD}.log")


# 單例鎖：同一台機器上只准跑一份 app.py
# ----------------------------------------------------------------------
# /run 是 tmpfs，重開機自動清空；鎖由 kernel 綁在 fd 上，process 結束
# （含當掉、被 kill）就自動釋放，不會留下殘鎖需要手動清。
LOCK_PATH = "/run/yoshinoya_ai.lock"
EXIT_ALREADY_RUNNING = 3   # yoshinoya_ai.service 用 RestartPreventExitStatus 認這個碼

_lock_fd = None  # 必須保持開著：fd 一關，鎖就沒了


def acquire_single_instance_lock() -> bool:
    """搶單例鎖。拿不到就回 False，呼叫端應該直接結束。

    為什麼需要這個
    --------------
    2026-09-29 現場的 MQTT 互踢（`Client client already connected, closing old
    connection.` 一個下午 181 次、app 這邊連續 `reason: 7`）根因是**同時有兩份
    app.py 在跑**：service 起了一份，手動除錯或 App 的 AI 啟動鈕又帶起一份。
    兩份用同一個 MQTT client id，於是在 broker 上互踢。

    把 client id 改成唯一值只會讓症狀消失 —— 兩份會各自連上，然後繼續搶
    HikRobot 相機、DLA/APU 與 /mnt/reserved 的寫入，反而更難發現。所以擋在
    這裡：第二份連相機、模型、MQTT 都不去碰就結束。
    """
    global _lock_fd
    try:
        _lock_fd = open(LOCK_PATH, "a+")
    except OSError as e:
        # 鎖檔開不起來不該讓 app 起不來（例如 /run 不可寫的異常環境）
        logger.warning(f"[單例] 無法開啟鎖檔 {LOCK_PATH}（{e}），跳過單例檢查")
        return True

    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _lock_fd.seek(0)
        other = _lock_fd.read().strip() or "?"
        logger.error(
            f"[單例] 已有另一份 AI 程式在執行 (pid={other})，本次啟動放棄。"
            f" 要手動除錯請先 systemctl stop yoshinoya_ai"
            f"（或 systemctl mask --runtime yoshinoya_ai，重開機後自動失效）。"
        )
        return False

    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
    return True


def main():

    # 先接上 stderr，這樣 _resolve_log_sink() 的退回警告才看得到
    logger.remove()
    logger.add(
        sys.stderr,
        level='INFO'
    )

    # 單例檢查刻意排在「接上檔案 sink 之前」：搶不到鎖的那份只會對 stderr
    # 說一句就走（service 的 journal / 手動執行的終端機都看得到），不會去開
    # 正在被另一份寫入的當日 log 檔，也就不可能在午夜剛好觸發 rotation、
    # 把人家的 log 檔改名。
    if not acquire_single_instance_lock():
        sys.exit(EXIT_ALREADY_RUNNING)

    # 開機時鐘還沒校正就先等（最多 120 秒）。排在接上檔案 sink 之前：
    # 時間不對時 log 路徑的日期也是錯的，會開出一個 20250530/ 的 log 檔。
    clock_ok, clock_waited = clock_guard.wait_for_valid_clock(timeout_sec=120)

    logger.add(
        _resolve_log_sink(),
        level='INFO',
        rotation="00:00"
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["yoshinoya", "suzumo"])
    task = parser.parse_args().profile or "ocr"
    # 目前僅供人工辨識：StreamManager 沒有用這個參數，也沒有經 MQTT 對外發布
    version = "3.3.1"

    # 一次開機的起點，log 裡往上找這行就是這一輪的開頭（模型檔名見後面的 [dla_guard]）
    logger.info(f"{'=' * 20} [程式啟動] v{version} task={task} pid={os.getpid()} {'=' * 20}")
    clock_guard.report(clock_ok, clock_waited)

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
        logger.info(f"{'=' * 20} [程式結束] v{version} pid={os.getpid()} {'=' * 20}")

if __name__ == "__main__":
    main()