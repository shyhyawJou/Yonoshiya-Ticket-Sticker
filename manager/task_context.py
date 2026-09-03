from __future__ import annotations

import queue
from pathlib import Path
from typing import List, Optional

from loguru import logger

from config import load_config
from utils.mqtt_bus import MqttBus, MqttSettings
from logic_engine import LogicEngine
from ocr_engine import AsyncOCR
from mmr_engine import Rotated_RTMDET


class TaskContext:
    """
    管理單一任務所需要的一組物件：cfg / bus / mmr / logic / ocr。

    這組物件在概念上是綁在一起的（同一個任務的設定 + 同一套模型 +
    同一個狀態機），所以把「載入」跟「模式切換時重建」都收斂在這裡，
    StreamManager 不需要知道重建的細節，只需要呼叫 load_task() /
    switch_mode()。
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

        self.cfg = None
        self.bus: Optional[MqttBus] = None
        self.mmr: Optional[Rotated_RTMDET] = None
        self.logic: Optional[LogicEngine] = None
        self.ocr: Optional[AsyncOCR] = None
        self.config_path: Optional[Path] = None
        self._on_recording_start = None
        self._on_recording_stop = None

        self.ocr_result_queue: "queue.Queue" = queue.Queue()

    # ---------- 載入 ----------

    def load_task(self, task: str, on_recording_start=None, on_recording_stop=None, get_video_filename=None) -> None:
        """
        安全地載入指定任務的設定檔、AI 模型和邏輯引擎。
        """
        logger.info(f"正在為任務 '{task}' 載入設定...")

        config_path = self.base_dir / "tasks" / task / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"任務 '{task}' 的設定檔不存在於 {config_path}")

        self.config_path = config_path
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._get_event_video_filename = get_video_filename

        cfg = load_config(str(config_path))
        class_names = [c.name for c in cfg.classes]
        self.cfg = cfg

        # --- init mqtt ---
        self.bus = MqttBus(MqttSettings(
            host=cfg.mqtt.host,
            port=cfg.mqtt.port,
            base_topic=cfg.mqtt.base_topic,
            session=cfg.mqtt.session))

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
            dict_path=cfg.runtime.model.text,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            get_video_filename=self._get_event_video_filename,
        )

        # --- init ocr ---
        self.ocr = AsyncOCR(
            det_path=cfg.runtime.model.ocr_det,
            cls_path=cfg.runtime.model.ocr_cls,
            rec_path=cfg.runtime.model.ocr_rec,
            dict_path=cfg.runtime.model.text,
            result_callback=self._on_ocr_result
        )
        self.ocr.start()

        # --- mqtt connect ---
        self.bus.connect()

        logger.info(f"已成功載入任務: {task}")

    # ---------- OCR 結果 ----------

    def _on_ocr_result(self, frame_crop, rec_res, dt_boxes, is_flip, time_cost, metadata) -> None:
        """由 OCR Thread 呼叫的 callback"""
        self.ocr_result_queue.put({
            "frame_crop": frame_crop,
            "rec_res": rec_res,
            "dt_boxes": dt_boxes,
            "is_flip": is_flip,
            "metadata": metadata,
        })

    def drain_ocr_results(self) -> List[dict]:
        """把目前累積在 queue 裡的 OCR 結果全部非阻塞取出"""
        results = []
        while not self.ocr_result_queue.empty():
            try:
                results.append(self.ocr_result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    # ---------- 模式切換 ----------

    def switch_mode(self, new_mode: str) -> bool:
        """
        切換 tray / single / preset_roi 模式，重建 LogicEngine（連同裡面
        所有 trays 狀態一起丟棄，不需要額外清空動作）。

        回傳是否真的切換成功；切換失敗時會自動退回原本的 mode。
        """
        if new_mode not in ("tray", "single", "preset_roi"):
            logger.error(f"[MODE] 不支援的 mode: '{new_mode}'，僅接受 'tray' 或 'single' 或 'preset_roi'")
            return False

        if new_mode == self.cfg.mode:
            logger.warning(f"[MODE] 目前已經是 '{new_mode}' 模式，略過切換")
            return False

        old_mode = self.cfg.mode
        logger.info(f"[MODE] 切換模式: '{old_mode}' -> '{new_mode}'，重建狀態機...")

        try:
            self.cfg.mode = new_mode
            self.logic = LogicEngine(
                cfg=self.cfg,
                bus=self.bus,
                mmr=self.mmr,
                rec_path=self.cfg.runtime.model.ocr_rec,
                dict_path=self.cfg.runtime.model.text,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                get_video_filename=self._get_event_video_filename,
            )
        except Exception as e:
            logger.error(f"[MODE] 切換模式失敗，退回 '{old_mode}': {e}")
            self.cfg.mode = old_mode
            return False

        # 舊模式殘留的 OCR 結果一併清掉，避免對到錯的 tray_id
        # （畫面暫存的 ocr_data 由呼叫端 (StreamManager) 自行清除，因為那是
        #  主迴圈的狀態，不屬於 TaskContext）
        self.drain_ocr_results()

        logger.success(f"[MODE] 已切換為 '{new_mode}' 模式")
        return True

    # ---------- 清理 ----------

    def stop(self) -> None:
        if self.ocr:
            self.ocr.stop()

        if self.bus:
            try:
                self.bus.disconnect()
                logger.info("MQTT bus disconnected.")
            except Exception as e:
                logger.error(f"Clean MQTT bus error: {e}")