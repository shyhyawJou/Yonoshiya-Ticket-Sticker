from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from stream_manager import StreamManager

class CommandDispatcher:
    """
    處理透過 MQTT 收到的指令。

    用 dispatch table 取代原本的 if/elif 鏈：新增指令只要多加一個
    handler method + 註冊一行，不用去改一個越來越長的函式，每個
    handler 也能各自獨立測試。
    """

    def __init__(self, manager: "StreamManager"):
        self.manager = manager
        self._handlers = {
            "plot_setting": self._on_plot_setting,
            "mode_setting": self._on_mode_setting,
            "capture": self._on_capture,
            "reset": self._on_reset,
            "hardware_ctrl": self._on_hardware_ctrl,
            "no_tray_setting": self._on_no_tray_setting,
            "img_reverse_xy": self._on_img_reverse_xy,
        }

    def _flip_roi_points(self, points, width: int, height: int, flip_x: bool, flip_y: bool):
        flipped = []
        for x, y in points:
            new_x = (width - 1 - x) if flip_x else x
            new_y = (height - 1 - y) if flip_y else y
            flipped.append([new_x, new_y])
        return flipped

    def _flip_preset_roi(self, m: "StreamManager", flip_x: bool, flip_y: bool) -> None:
        """依 flip_x/flip_y 翻轉 cfg.preset_roi 內所有 ROI 的頂點座標。"""
        if not flip_x and not flip_y:
            return

        cfg = m.task.cfg
        width = cfg.runtime.camera.width
        height = cfg.runtime.camera.height

        preset_roi = getattr(cfg, "preset_roi", None)
        if not preset_roi:
            return

        # PresetRoiCfg 是 dataclass,固定欄位是 ticket_roi / object_roi / packing_roi,
        # 其中後兩個可能是 None(預留欄位，尚未設定)
        roi_field_names = ("ticket_roi", "object_roi", "packing_roi")

        with m.data_lock:
            for field_name in roi_field_names:
                roi_cfg: Optional[RoiCfg] = getattr(preset_roi, field_name, None)
                if roi_cfg is None:
                    continue

                old_points = roi_cfg.points
                new_points = self._flip_roi_points(old_points, width, height, flip_x, flip_y)
                roi_cfg.points = new_points

                logger.info(
                    f"[MODE] preset_roi '{field_name}' 座標已翻轉 "
                    f"(flip_x={flip_x}, flip_y={flip_y}): {old_points} -> {new_points}"
                )

    def handle(self, cmd: str, payload: dict) -> None:
        logger.info(f"收到 MQTT 指令: '{cmd}', 內容: {payload}")

        handler = self._handlers.get(cmd)
        if handler is None:
            logger.warning(f"不支援的指令: '{cmd}'")
            return

        try:
            handler(payload)
        except Exception:
            logger.error(traceback.format_exc())

    # ---------- handlers ----------

    def _on_plot_setting(self, payload: dict) -> None:
        m = self.manager
        if "box" in payload:
            val = payload["box"]
            m.show_box = str(val).lower() == 'true'
            logger.info(f"設定 show_box: {m.show_box} 成功")
        elif "fps" in payload:
            val = payload["fps"]
            m.show_fps = str(val).lower() == 'true'
            logger.info(f"設定 show_box: {m.show_fps} 成功")
        else:
            logger.warning(f"不支援 {payload} 畫圖設定")

    def _on_mode_setting(self, payload: dict) -> None:
        pass

    def _on_capture(self, payload: dict) -> None:
        self.manager._trigger_capture = True
        logger.info("CAPTURE done.")

    def _on_reset(self, payload: dict) -> None:
        m = self.manager
        reset_type = payload.get("type")
        tray_id = payload.get("tray_id")

        # 用原始畫面 (未經錄影壓縮) 存檔，檔名含日期時間避免覆蓋
        if getattr(m, "tmp_frame", None) is not None:
            m._save_frame(m.tmp_frame, "reset_capture")
        else:
            logger.warning("[RESET] tmp_frame 尚未就緒，略過截圖")

        logger.info(f"[RESET] 收到重置指令 (type: '{reset_type}')...")
        with m.data_lock:
            if reset_type == "a":
                if not tray_id:
                    logger.warning("[RESET] 缺少 tray_id，略過")
                    return
                m.task.logic.reset(tray_id=tray_id)
            elif reset_type == "b":
                if not tray_id:
                    logger.warning("[RESET] 缺少 tray_id，略過")
                    return
                m.task.logic.reset_tray_states(tray_id=tray_id)
            elif reset_type == "all":
                m.task.logic.reset_all()
            m.task.drain_ocr_results()

    def _on_hardware_ctrl(self, payload: dict) -> None:
        m = self.manager
        ctrl_type = payload.get("type")
        ctrl = payload.get('control')
        if ctrl_type == "camera":
            if ctrl == 'reset_parameter':
                m.camera.reset_camera_parameters()
            else:
                m.camera.set_camera_parameters(payload, True)
        else:
            logger.warning(f"不支援 {ctrl_type} 硬體設定")

    def _on_no_tray_setting(self, payload: dict) -> None:
        m = self.manager
        no_tray = payload.get("no_tray")
        new_mode = 'preset_roi' if no_tray else 'tray'

        changed = m.task.switch_mode(new_mode)
        if changed:
            # 舊模式殘留的畫面暫存資料 (ocr_data) 一併清掉，避免對到錯的 tray_id
            with m.data_lock:
                m.ocr_data = None

    def _on_img_reverse_xy(self, payload: dict) -> None:
        m = self.manager

        if "img_reverse_xy" not in payload:
            logger.warning(f"[MODE] img_reverse_xy 指令缺少 'img_reverse_xy' 欄位，略過: {payload}")
            return

        if m.camera.capture is None:
            logger.error("[MODE] 相機尚未就緒 (capture is None)，略過 img_reverse_xy")
            return

        reverse = str(payload.get("img_reverse_xy")).lower() == 'true'

        hik_params = m.task.cfg.camera_params.get('hik')
        if not isinstance(hik_params, dict):
            logger.error("[MODE] camera_params 內找不到 'hik' 設定區塊，無法切換 ReverseX/ReverseY")
            return

        reverse_x_param = hik_params.get('ReverseX')
        reverse_y_param = hik_params.get('ReverseY')
        if not isinstance(reverse_x_param, dict) or not isinstance(reverse_y_param, dict):
            logger.error(
                f"[MODE] camera_params['hik'] 內 ReverseX/ReverseY 格式不符預期: "
                f"ReverseX={reverse_x_param}, ReverseY={reverse_y_param}"
            )
            return

        old_x = reverse_x_param.get('value')
        old_y = reverse_y_param.get('value')

        logger.info(
            f"[MODE] 收到水平垂直翻轉影像指令, "
            f"目前 ReverseX={old_x}, ReverseY={old_y} -> 要求 img_reverse_xy={reverse}"
        )

        if old_x == reverse and old_y == reverse:
            logger.warning("[MODE] ReverseX/ReverseY 已經是要求的值，略過重啟")
            return

        # --- 新增：算出哪些軸真的要翻，並同步翻轉 preset_roi 座標 ---
        # flip_x = (old_x != reverse)
        # flip_y = (old_y != reverse)
        # self._flip_preset_roi(m, flip_x, flip_y)

        # 先更新記憶體內的值（camera reload 時會用 cfg.camera_params['hik']
        # 裡目前的值套用到新開的相機上）
        reverse_x_param['value'] = reverse
        reverse_y_param['value'] = reverse
        # 手動同步存檔，不等相機內部 debounce
        # （因為馬上要呼叫 reload() 把整個 capture 物件砍掉重建，
        # debounce 執行緒會被連帶結束，來不及寫入的話這次改動就白改了）
        try:
            m.camera.save_camera_config()
        except Exception:
            logger.error(f"[MODE] 同步存檔 ReverseX/ReverseY 失敗: {traceback.format_exc()}")

        logger.info(
            f"[MODE] 已更新 ReverseX={reverse}, ReverseY={reverse}，重啟相機套用..."
        )
        m.camera.reload()