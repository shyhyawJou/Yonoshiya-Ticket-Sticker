from __future__ import annotations

import os
import sys
import signal
import time
import yaml
import socket
import queue
import traceback
import threading
import numpy as np
from loguru import logger
from pathlib import Path
import cv2
import threading
import subprocess

from config import load_config
from mqtt_bus import MqttBus, MqttSettings
from ocr_engine import AsyncOCR
from mmr_engine import Rotated_RTMDET
from hikcam import HikCamera
from logic_engine import LogicEngine
from video import Video
from streamer import Mjpeg_Streamer
from logic.tray_tracker import Tray


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


class StreamManager:

    def __init__(self, task: str, ver: str):
        self.base_dir = Path(__file__).resolve().parent

        # --- init thread ---
        self.data_lock = threading.Lock()

        # --- vars ---
        self.cfg = None 
        self.mmr = None
        self.logic = None
        self.ocr = None     

        # --- Capture vars ---
        self.capture = None 
        self.capture_thread = None       
        self.frame_queue = queue.Queue(maxsize=1)

        # --- video recorder ---
        self.save_dir = Path("/mnt/reserved/record/stream")
        self.video_record_size = (1280, 960)
        self.video_recorder = None

        # --- Flag vars ---
        self.show_box: bool = True
        self.show_fps: bool = True
        self.camera_alone: bool = True
        self._running: bool = False
        self._is_reloading_camera: bool = False
        self._is_reloading_stream: bool = False
        self._trigger_capture: bool = False

        # --- OCR Data ---
        self.ocr_result_queue = queue.Queue()
        self.ocr_data = None

        # --- init engine ---
        self._load_task(task)

        # --- init streamer ---
        self.stream_mode = "detect"
        self.stream_size = self.cfg.runtime.stream.stream_size
        self.streamer = Mjpeg_Streamer(
            route='/ai_stream', 
            port=self.cfg.runtime.stream.port, 
            size=self.stream_size, 
            quality=40
        )
        self.streamer.start()

        logger.info(f"StreamManager v{ver} 初始化成功!")

    def _load_task(self, task: str):
        """
        安全地重新載入指定任務的設定檔、AI 模型和邏輯引擎
        """
        logger.info(f"正在為任務 '{task}' 載入設定...")

        config_path = self.base_dir / "tasks" / task / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"任務 '{task}' 的設定檔不存在於 {config_path}")

        self.config_path = config_path

        cfg = load_config(str(config_path))
        class_names = [c.name for c in cfg.classes]
  
        self.cfg = cfg

        # --- init mqtt ---
        self.bus = MqttBus(MqttSettings(
                           host=cfg.mqtt.host, 
                           port=cfg.mqtt.port, 
                           base_topic=cfg.mqtt.base_topic, 
                           session=cfg.mqtt.session))
        self.bus.on_command(self._handle_cmd)       

        # --- init video recorder ---
        self.video_recorder = Video(
                    base_dir=str(self.save_dir),
                    frame_w=self.video_record_size[0],
                    frame_h=self.video_record_size[1],
                    fps=15)

        # --- init mmrotate engine ---
        self.mmr = Rotated_RTMDET(
            path=cfg.runtime.model.object_det, 
            classes=class_names, 
            conf_thresh=cfg.thresholds.ai_conf,
            iou_thresh=cfg.thresholds.ai_iou
        )  

        # --- init logic engine ---                     
        self.logic = LogicEngine(
            cfg=self.cfg, 
            bus=self.bus, 
            mmr=self.mmr,
            rec_path=cfg.runtime.model.ocr_rec,
            dict_path=cfg.runtime.model.text
        )

        # --- init ocr ---
        self.ocr = AsyncOCR(
            det_path=cfg.runtime.model.ocr_det,
            cls_path=cfg.runtime.model.ocr_cls,
            rec_path=cfg.runtime.model.ocr_rec,
            dict_path=cfg.runtime.model.text,
            result_callback=self._on_ocr_callback
        )
        self.ocr.start()

        # --- mqtt connect ---
        self.bus.connect()

        logger.info(f"已成功載入任務: {task}")

    def _save_frame(self, frame: np.ndarray, folder_name: str) -> None:
        try:
            save_path_dir = self.save_dir / folder_name
            save_path_dir.mkdir(parents=True, exist_ok=True)

            current_time = time.time()
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(current_time))
            msec = int(current_time * 1000) % 1000
            filename = f"{timestamp}_{msec:03d}.jpg"
            
            save_path = save_path_dir / filename
            cv2.imwrite(str(save_path), frame)
     
        except Exception as e:
            logger.error(f"儲存影像失敗 (Folder: {folder_name}): {e}")

    def _on_ocr_callback(self, frame_crop, rec_res, dt_boxes, is_flip, time_cost, metadata):
        """
        由 OCR Thread 呼叫的 Callback
        """
        result_pkg = {
            "frame_crop": frame_crop,
            "rec_res": rec_res,
            "dt_boxes": dt_boxes,
            "is_flip": is_flip,
            "metadata": metadata
        }
        self.ocr_result_queue.put(result_pkg)
  
    def start_camera(self) -> None:
        if self.camera_alone:
            self.capture_thread = threading.Thread(target=self._camera_worker, daemon=True)
            self.capture_thread.start()
        else:
            self.init_camera()

        self._running = True

        logger.info("Camera started. Streaming...")

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        針對 crop 後的影像做強化處理。
        適用於 OCR 前的 warped_img。
        """
        def pad_to_square(img):
            h, w = img.shape[:2]
            size = max(h, w)
            pad_img = np.full((size, size, 3), 114, dtype=np.uint8)
            pad_img[:h, :w] = img
            return pad_img

        if frame is None or frame.size == 0:
            return frame    
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # # 2. 降噪（bilateral 保邊又快）
        frame = cv2.bilateralFilter(frame, d=5, sigmaColor=15, sigmaSpace=15)

        # 3. 轉灰階（保留階層，不二值化）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 5. CLAHE（保留梯度，不破壞階層）
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 6. 轉回 BGR 送給 PaddleOCR（它預期 3 通道輸入）
        frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return frame

    def _draw_overlay(self, frame):
        """
        將 OCR 結果畫在 frame 上
        """
        vis_frame = frame 

        current_data = None
        with self.data_lock:
            if self.ocr_data is not None:
                current_data = self.ocr_data 

        if current_data is None:
            return vis_frame

        tray_id = current_data["metadata"]["tray_id"]
        try:
            if tray_id not in self.logic.trays:
                with self.data_lock:
                    self.ocr_data = None
                return vis_frame
        except RuntimeError:
            return vis_frame
        
        dt_boxes = current_data['dt_boxes']
        rec_res = current_data['rec_res']
        metadata = current_data["metadata"]
        M_inv = metadata["M_inv"]
        if len(dt_boxes) > 0 and len(rec_res) > 0:
            for box, (text, score) in zip(dt_boxes, rec_res):
                if score > 0.5:
                    if box.size != 8:   # 4 個點 × (x,y) = 8
                        logger.warning(f"略過異常 box(點數不足), size={box.size}）: {box}")
                        continue
                    pts = np.array(box, dtype="float32").reshape(-1, 1, 2)
                    transformed_pts = cv2.transform(pts, M_inv)
                    transformed_pts = transformed_pts.reshape(4, 2).astype(int)
                    cv2.polylines(vis_frame, [transformed_pts], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # plot cheched stickers
        for sticker in self.logic.trays[tray_id].stickers:
            if sticker.is_checked:
                bbox = np.array(sticker.bbox).astype(int)
                xy = np.array(sticker.xywhr[:2]).astype(int)
                cv2.polylines(vis_frame, bbox.reshape(-1, 4, 2), True, (0, 255, 0), 2)
                cv2.putText(vis_frame, 'OK', xy + [-50, 30], cv2.FONT_HERSHEY_SIMPLEX, 
                            3, (0, 255, 0), 7)
        
        return vis_frame

    def _camera_worker(self):
        """
        獨立的相機取像執行緒 (Producer)
        """
        try:
            self.init_camera()
            logger.info("Camera Worker 啟動")

            consecutive_errors = 0
            MAX_RETRIES = 30 

            while self._running and self.capture is not None and self.capture.isOpened():
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

                    # frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR_VNG)
                    consecutive_errors = 0

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.frame_queue.put(frame)
                    if self.camera_alone:     
                        record_frame = cv2.resize(frame, self.video_record_size)
                        self.video_recorder.write_frame(record_frame)
                    time.sleep(0.03)

                except Exception as e:
                    #logger.error(f"Camera worker error: {e}")
                    logger.error(f"Camera worker error: {traceback.format_exc()}")
                    time.sleep(0.1) 
        except:
            logger.error(f"Camera worker error: {traceback.format_exc()}")
        finally:
            self.capture.release()
            logger.info("Camera Worker 結束")

    def stream_frames(self) -> None:
        """Main loop"""

        fps_counter = 0
        display_fps = "fps: 0.0"
        time_1 = time.time()
        last_processed_id = -1

        while self._running:

            if self._is_reloading_camera or self._is_reloading_stream:
                time.sleep(0.1)
                continue
            
            if self.camera_alone:
                if not self.capture_thread.is_alive():
                    logger.error("偵測到 Camera Thread 已停止，主程式即將退出...")
                    self._running = False
                    break

            try:
                if self.camera_alone:
                    try:
                        frame = self.frame_queue.get(block=True, timeout=1.0)
                        self.tmp_frame = frame.copy()
                    except queue.Empty:
                        continue
                else:   
                    ok, frame = self.capture.read()
                    if not ok:
                        logger.error("Failed to capture frame.")
                        break

                if self._trigger_capture:
                    self._save_frame(frame, "capture")
                    self._trigger_capture = False

                if not self.camera_alone:
                    record_frame = cv2.resize(frame, self.video_record_size)
                    self.video_recorder.write_frame(record_frame)

                while not self.ocr_result_queue.empty():
                    ocr_result = self.ocr_result_queue.get()
                    metadata = ocr_result["metadata"]

                    with self.data_lock:
                        self.ocr_data = ocr_result
                        self.logic.apply_ocr_result(
                            tray_id=metadata["tray_id"], 
                            item_type=metadata["type"], 
                            frame_crop=ocr_result["frame_crop"],
                            dt_boxes=ocr_result["dt_boxes"],
                            rec_res=ocr_result["rec_res"],
                            is_flip=ocr_result["is_flip"],
                            task_bbox=metadata["bbox"]
                        )
                # cv2.imwrite("ori_0.jpg", frame)
                out, detections = self.mmr.detect(frame, self.show_box)

                with self.data_lock:
                    tasks, movements = self.logic.update(frame, detections) 

                # with self.data_lock:
                #     if self.ocr_data is not None:
                #         current_tray_id = self.ocr_data["metadata"].get("tray_id")
                #         if current_tray_id in movements:
                #             dx, dy = movements[current_tray_id]
                #             self.ocr_data["dt_boxes"][:, :, 0] += dx
                #             self.ocr_data["dt_boxes"][:, :, 1] += dy

                for task in tasks:
                    if not self.ocr.is_busy:

                        cx, cy, w, h, r = task["xywhr"]
                        warped_img, M_inv = self.mmr.crop_by_angle(frame, cx, cy, w, h, r)
                        # cv2.imwrite("testori.jpg", warped_img)
                        #warped_img = self.enhance(warped_img)
                        # cv2.imwrite("warped_img.jpg", warped_img)
                        if warped_img.size == 0 or w < 10 or h < 10:
                            continue                     

                        warped_dir = Path(self.video_recorder.today_dir) / "ocr_cut"
                        warped_dir.mkdir(parents=True, exist_ok=True)
                        _t = time.time()
                        _ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(_t))
                        _ms = int(_t * 1000) % 1000
                        cv2.imwrite(str(warped_dir / f"{_ts}_{_ms:03d}.jpg"), warped_img)
                        
                        metadata = {
                            "tray_id": task["tray_id"],
                            "type": task["type"],
                            "M_inv": M_inv,
                            "bbox": task["bbox"],
                            "cls_name": task['cls_name']
                        }

                        self.ocr.request_ocr(frame_crop=warped_img, metadata=metadata)

                        with self.data_lock:
                            self.logic.set_ocr_busy(tray_id=task["tray_id"], item_type=task["type"], bbox=task["bbox"])

                        break 
                
                current_1 = time.time()
                fps_counter += 1
                time_diff = current_1 - time_1
                if time_diff > 1.0:
                    fps = fps_counter / time_diff
                    display_fps = f"fps: {fps:.1f}"
                    fps_counter = 0
                    time_1 = current_1

                _ = self._draw_overlay(out)

                if self.show_fps:
                    cv2.putText(out, display_fps, (20, 2500), cv2.FONT_HERSHEY_SIMPLEX, 5, (194,244,255), 5)

                if self.stream_mode == "detect":
                    stream_out = cv2.resize(out, self.stream_size)
                    self.streamer.push_frame(stream_out)
                else:
                    self.streamer.push_frame(out)

            except Exception:
                logger.error(traceback.format_exc())
                break

    def stop_camera(self):
        if self.capture and self.capture.isOpened():
            try:
                self.capture.release()
                logger.info("Camera stopped.")
            except Exception as e:
                logger.error(f"Stop camera error: {e}")
            finally:
                self.capture = None
        else:
            pass

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

        if hasattr(self, 'video_recorder') and self.video_recorder:
            self.video_recorder.stop()

        if hasattr(self, 'ocr') and self.ocr:
            self.ocr.stop()

        if hasattr(self, 'bus') and self.bus:
            try:
                self.bus.disconnect()
                logger.info("MQTT bus disconnected.")
            except Exception as e:
                logger.error(f"Clean MQTT bus error: {e}")
        
        self.capture_thread.join(5)
        self.stop_stream()
        logger.info("Cleaned up.")

    # ---------- commands ----------
    def _handle_cmd(self, cmd: str, payload: dict) -> None:
        logger.info(f"收到 MQTT 指令: '{cmd}', 內容: {payload}")
        
        try:
            if cmd == "plot_setting":
                if "box" in payload:
                    val = payload["box"]
                    self.show_box = str(val).lower() == 'true'
                    logger.info(f"設定 show_box: {self.show_box} 成功")
                elif "fps" in payload:
                    val = payload["fps"]
                    self.show_fps = str(val).lower() == 'true'
                    logger.info(f"設定 show_box: {self.show_fps} 成功")
                else:
                    logger.warning(f"不支援 {payload} 畫圖設定")

            elif cmd == "mode_setting":
                pass

            elif cmd == "capture":
                self._trigger_capture = True
                logger.info("CAPTURE done.")
        
            elif cmd == "reset":
                reset_type = payload.get("type")
                tray_id = payload.get("tray_id")

                # 用原始畫面 (未經錄影壓縮) 存檔，檔名含日期時間避免覆蓋
                if hasattr(self, "tmp_frame") and self.tmp_frame is not None:
                    self._save_frame(self.tmp_frame, "reset_capture")
                else:
                    logger.warning("[RESET] tmp_frame 尚未就緒，略過截圖")

                logger.info(f"[RESET] 收到重置指令 (type: '{reset_type}')...")
                with self.data_lock:
                    if reset_type == "a":
                        if not tray_id:
                            logger.warning("[RESET] 缺少 tray_id，略過")
                            return
                        self.logic.reset(tray_id=tray_id)
                    elif reset_type == "b":
                        if not tray_id:
                            logger.warning("[RESET] 缺少 tray_id，略過")
                            return
                        self.logic.reset_tray_states(tray_id=tray_id)
                    elif reset_type == "all":
                        self.logic.reset_all()
                    while not self.ocr_result_queue.empty():
                        try:
                            self.ocr_result_queue.get_nowait()
                        except Exception:
                            break

            elif cmd == "hardware_ctrl":
                ctrl_type = payload.get("type")
                ctrl = payload.get('control')
                if ctrl_type == "camera":
                    if ctrl == 'reset_parameter':
                        self.capture.reset_camera_parameters()
                    else:
                        self.capture.set_camera_parameters(payload, True)
                else:
                    logger.warning(f"不支援 {ctrl_type} 硬體設定")

            elif cmd == "no_tray_setting":
                no_tray = payload.get("no_tray")
                new_mode = 'single' if no_tray else 'tray'

                if new_mode not in ("tray", "single"):
                    logger.error(f"[MODE] 不支援的 mode: '{new_mode}'，僅接受 'tray' 或 'single'")
                    return

                if new_mode == self.cfg.mode:
                    logger.warning(f"[MODE] 目前已經是 '{new_mode}' 模式，略過切換")
                    return

                logger.info(f"[MODE] 切換模式: '{self.cfg.mode}' -> '{new_mode}'，重建狀態機...")
                old_mode = self.cfg.mode
                try:
                    self.cfg.mode = new_mode
                    # 重新 new 一份 LogicEngine：舊的 tracker/state machine（連同
                    # 裡面所有 trays 狀態）直接被丟棄，不需要額外做清空動作。
                    self.logic = LogicEngine(
                        cfg=self.cfg,
                        bus=self.bus,
                        mmr=self.mmr,
                        rec_path=self.cfg.runtime.model.ocr_rec,
                        dict_path=self.cfg.runtime.model.text
                    )
                except Exception as e:
                    logger.error(f"[MODE] 切換模式失敗，退回 '{old_mode}': {e}")
                    self.cfg.mode = old_mode
                    return

                    # 舊模式殘留的 OCR 結果/畫面暫存資料一併清掉，避免對到錯的 tray_id
                with self.data_lock:
                    self.ocr_data = None

                while not self.ocr_result_queue.empty():
                    try:
                        self.ocr_result_queue.get_nowait()
                    except Exception:
                        break

                logger.success(f"[MODE] 已切換為 '{new_mode}' 模式")
                #self.bus.publish_system({
                #    "ts": _now(),
                #    "type": "MODE_CHANGED",
                #    "msg": {"mode": new_mode}
                #})

            else:
                logger.warning(f"UNKNOWN_CMD: {cmd}")
        except:
            logger.error(traceback.format_exc())

    def init_camera(self):
        if self.cfg.runtime.camera.device == 'hik':
            logger.info(f"使用工業相機 (HIK)...")
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