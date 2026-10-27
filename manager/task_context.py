from __future__ import annotations

import queue
import time
import traceback
from pathlib import Path
from typing import List, Optional

from loguru import logger

from utils import dla_guard
from utils.mqtt_bus import MqttBus, MqttSettings
from utils.clip_video import get_now_iso_str
from config import PresetRoiCfg, RoiCfg, save_preset_roi
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

    def load_task(self, task: str, on_recording_start=None, on_recording_stop=None, get_video_filename=None,
                  save_frame=None) -> None:
        """
        安全地載入指定任務的設定檔、AI 模型和邏輯引擎。
        """
        logger.info(f"正在為任務 '{task}' 載入設定...")

        from profile_config import load_profile_config
        cfg = load_profile_config(self.base_dir, None if task == "ocr" else task)
        self.profile = cfg.profile_name
        config_path = Path(cfg.source_config_path)
        logger.info(f"Profile={self.profile}, config={config_path}")

        self.config_path = config_path
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop
        self._get_event_video_filename = get_video_filename
        self._save_frame_cb = save_frame          # save_frame(tag)：存一張全幀（醬料包判定用）

        class_names = [c.name for c in cfg.classes]
        self.cfg = cfg

        # --- init mqtt ---
        # client_id 沒設就不傳，讓 MqttSettings 自己產生 process 專屬的唯一 id
        # （固定 id 會讓重啟時新舊行程在 broker 上互踢，見 utils/mqtt_bus.py）
        mqtt_kwargs = dict(
            host=cfg.mqtt.host,
            port=cfg.mqtt.port,
            base_topic=cfg.mqtt.base_topic,
            session=cfg.mqtt.session,
        )
        if cfg.mqtt.client_id:
            mqtt_kwargs["client_id"] = cfg.mqtt.client_id
        self.bus = MqttBus(MqttSettings(**mqtt_kwargs))

        # --- DLA 載入守衛 ---
        # 必須在任何模型載入之前套用。config.yaml 沒有 dla_guard 那一段時，
        # 這些欄位都是 dataclass 預設值，行為就是「啟用、最多連續重開 2 次」。
        dla_guard.configure(
            enabled=cfg.dla_guard.enabled,
            max_reboots=cfg.dla_guard.max_reboots,
            delay_sec=cfg.dla_guard.delay_sec,
            reset_after_sec=cfg.dla_guard.reset_after_sec,
        )

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
            save_frame=self._save_frame_cb,
        )

        # --- init ocr ---
        self.ocr = AsyncOCR(
            det_path=cfg.runtime.model.ocr_det,
            cls_path=cfg.runtime.model.ocr_cls,
            rec_path=cfg.runtime.model.ocr_rec,
            dict_path=cfg.runtime.model.text,
            result_callback=self._on_ocr_result,
            ocr_routes=cfg.ocr_routes,
        )
        self.ocr.start()

        # --- mqtt connect ---
        self.bus.connect()

        # 走到這裡代表五個 DLA 模型全部載入成功：清掉自動重開計數並記下檔案
        # 指紋，讓下一次偶發的 APU 問題有完整的重開額度，也讓「模型被換掉」
        # 的情況可以被分辨出來（見 utils/dla_guard.py 的說明）。
        dla_guard.mark_ok()

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

    # ---------- Preset ROI ----------

    def _normalize_roi_points(self, points) -> Optional[List[List[float]]]:
        """
        檢查並整理 App / 自動校正給的四個點。

        會做的事:確認真的是 4 個 [x, y]、數值可轉 float、夾回取像解析度範圍內、
        面積不為 0(避免三點共線或四點重疊的框直接讓 shapely 判斷全部失效)。
        座標系跟 App 約定為 runtime.camera.width/height 的原始取像解析度。
        """
        if not isinstance(points, (list, tuple)) or len(points) != 4:
            logger.error(f"[ROI] points 必須是 4 個點,收到: {points}")
            return None

        width = self.cfg.runtime.camera.width
        height = self.cfg.runtime.camera.height

        cleaned: List[List[float]] = []
        for pt in points:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                logger.error(f"[ROI] 每個點都必須是 [x, y],收到: {pt}")
                return None
            try:
                x, y = float(pt[0]), float(pt[1])
            except (TypeError, ValueError):
                logger.error(f"[ROI] 座標不是數值: {pt}")
                return None

            clamped_x = min(max(x, 0.0), float(width - 1))
            clamped_y = min(max(y, 0.0), float(height - 1))
            if (clamped_x, clamped_y) != (x, y):
                logger.warning(
                    f"[ROI] 座標 {[x, y]} 超出取像解析度 {width}x{height},"
                    f"夾到 {[clamped_x, clamped_y]}"
                )
            cleaned.append([clamped_x, clamped_y])

        from logic.geometry import polygon_area
        flat = tuple(v for pt in cleaned for v in pt)
        if polygon_area(flat) <= 0.0:
            logger.error(f"[ROI] 這四個點圍不出有效面積(重疊或共線),略過: {cleaned}")
            return None

        return cleaned

    @staticmethod
    def _normalize_roi_color(color) -> Optional[List[int]]:
        """色值跟 App 約定是 RGB [r, g, b] 0~255;格式不對就當作沒給(保留原本顏色)"""
        if color is None:
            return None
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            logger.warning(f"[ROI] color 必須是 [r, g, b],忽略: {color}")
            return None
        try:
            rgb = [int(c) for c in color]
        except (TypeError, ValueError):
            logger.warning(f"[ROI] color 不是數值,忽略: {color}")
            return None
        if any(c < 0 or c > 255 for c in rgb):
            logger.warning(f"[ROI] color 超出 0~255,忽略: {color}")
            return None
        return rgb

    def apply_preset_roi(
        self,
        name: str,
        points,
        color=None,
        persist: bool = True,
        publish: bool = True,
        source: str = "mqtt",
    ) -> Optional[RoiCfg]:
        """
        套用一個 preset_roi 區域:記憶體 -> tracker -> config.yaml -> 回報 App。

        App 端只送指令、不寫檔(跟前端溝通好的分工),所以現值一律由這裡落地;
        同時馬上推給 tracker,讓現場調 ROI 不需要重啟程式驗證。

        回傳套用後的 RoiCfg,失敗回 None(錯誤原因已經寫進 log)。
        """
        if self.cfg is None or self.cfg.preset_roi is None:
            logger.error("[ROI] 設定檔沒有 preset_roi 區塊,無法設定 ROI")
            return None

        if name not in PresetRoiCfg.ROI_FIELDS:
            logger.error(
                f"[ROI] 不支援的 ROI 名稱: '{name}',可用的有 {list(PresetRoiCfg.ROI_FIELDS)}"
            )
            return None

        cleaned_points = self._normalize_roi_points(points)
        if cleaned_points is None:
            return None
        cleaned_color = self._normalize_roi_color(color)

        roi: Optional[RoiCfg] = getattr(self.cfg.preset_roi, name, None)
        if roi is None:
            roi = RoiCfg(name=name, points=cleaned_points, color=cleaned_color)
            setattr(self.cfg.preset_roi, name, roi)
        else:
            old_points = roi.points
            roi.points = cleaned_points
            if cleaned_color is not None:
                roi.color = cleaned_color
            logger.info(f"[ROI] {name} 座標更新 ({source}): {old_points} -> {cleaned_points}")

        # tracker 不會重讀 cfg，必須主動同步才會生效
        if self.logic is not None:
            self.logic.refresh_preset_roi()

        if persist:
            try:
                save_preset_roi(self.cfg, roi)
            except Exception:
                # 存檔失敗不該讓這次設定整個失敗：記憶體/畫面已經是新值了，
                # 只是下次開機會退回舊值，log 留下來讓現場知道要手動補。
                logger.error(f"[ROI] {name} 寫回設定檔失敗:\n{traceback.format_exc()}")

        if publish and self.bus is not None:
            self.bus.publish_system({
                "ts": get_now_iso_str(time.time(), utc=False),
                "type": "PRESET_ROI_UPDATED",
                "msg": {
                    "name": name,
                    "points": [[int(round(x)), int(round(y))] for x, y in roi.points],
                    "color": roi.color,
                    "source": source,
                },
            })

        return roi

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
            new_logic = LogicEngine(
                cfg=self.cfg,
                bus=self.bus,
                mmr=self.mmr,
                rec_path=self.cfg.runtime.model.ocr_rec,
                dict_path=self.cfg.runtime.model.text,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                get_video_filename=self._get_event_video_filename,
                save_frame=self._save_frame_cb,
            )
        except Exception as e:
            logger.error(f"[MODE] 切換模式失敗，退回 '{old_mode}': {e}")
            self.cfg.mode = old_mode
            return False

        # 新的建成功才收掉舊的：舊的進行中訂單不寫 CSV，只停錄影（見 LogicEngine.shutdown）
        self.logic.shutdown(reason=f"switch_mode:{old_mode}->{new_mode}")
        self.logic = new_logic

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
