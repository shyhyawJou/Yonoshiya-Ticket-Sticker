from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from utils.quad_validator import is_convex_ordered, find_angle_outliers


class OverlayRenderer:
    """
    負責把 ROI / OCR 辨識結果畫到畫面上。

    不持有任何可變狀態 —— 每次呼叫 draw() 都是「拿目前這一份 ocr_data
    畫一張圖」，畫完之後 ocr_data 要不要被清掉，由回傳值 keep_ocr_data
    告訴呼叫端，實際的清除/上鎖動作留給 StreamManager 處理（因為
    ocr_data 是主迴圈的共享狀態，不屬於這個類別）。
    """

    def __init__(self, task_context):
        # 只需要讀 task_context.cfg / task_context.logic，
        # 不持有自己的一份，永遠讀到最新狀態（包含模式切換後）
        self.task = task_context

    def draw_preset_roi(self, frame):
        """
        preset_roi 模式下，把設定檔裡定義的 ROI 區域畫到畫面上，
        方便現場除錯/調整 ROI 座標時直接對照。
        跟 OCR overlay 分開，因為 ROI 是「設定檔畫出來的靜態框」，
        不管當下有沒有 OCR 任務在跑都應該持續顯示。
        """
        cfg = self.task.cfg
        if cfg.preset_roi is None:
            return frame

        roi_entries = [
            cfg.preset_roi.ticket_roi,
            cfg.preset_roi.object_roi,
            cfg.preset_roi.packing_roi,
        ]

        for roi in roi_entries:
            if roi is None:
                continue

            pts = np.array(roi.points, dtype=np.int32).reshape(-1, 1, 2)
            color = tuple(roi.color) if roi.color else (0, 255, 255)  # 預設黃色(BGR)

            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=3)

            label_pos = tuple(pts[0][0] + np.array([10, -10]))
            cv2.putText(frame, roi.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, color, 3)

        return frame

    def draw(self, frame, ocr_data: Optional[dict]) -> Tuple[np.ndarray, bool]:
        """
        畫上 ROI + OCR 覆蓋層。

        回傳 (vis_frame, keep_ocr_data)：
        keep_ocr_data 為 False 時，代表這份 ocr_data 對應的 tray 已經
        不存在了，呼叫端應該把快取的 ocr_data 清掉。
        """
        vis_frame = self.draw_preset_roi(frame)

        if ocr_data is None:
            return vis_frame, True

        tray_id = ocr_data["metadata"]["tray_id"]
        try:
            if tray_id not in self.task.logic.trays:
                return vis_frame, False
        except RuntimeError:
            return vis_frame, True

        dt_boxes = ocr_data['dt_boxes']
        rec_res = ocr_data['rec_res']
        metadata = ocr_data["metadata"]
        # M_inv = metadata["M_inv"]
        # --- 改動處：用完整鏈條的反矩陣，而不是只還原 crop 那一步的 M_inv ---
        M_inv_full = metadata["M_inv_full"]          # 3x3 homogeneous
        M_inv_affine = M_inv_full[:2, :]              # 取前兩列 -> 2x3，給 cv2.transform 用

        if len(dt_boxes) > 0 and len(rec_res) > 0:
            # --- 第一階段：先收集這一批所有合法框，準備算角度離群值 ---
            valid_entries = []
            for box, (text, score) in zip(dt_boxes, rec_res):
                if score <= 0.5:
                    continue
                if box.size != 8:  # 4 個點 × (x,y) = 8
                    logger.warning(f"略過異常 box(點數不足), size={box.size}）: {box}")
                    continue
                pts = np.array(box, dtype="float32").reshape(-1, 1, 2)
                transformed_pts = cv2.transform(pts, M_inv_affine)
                transformed_pts = transformed_pts.reshape(4, 2).astype(int)
                valid_entries.append(transformed_pts)

            # --- 第二階段：對整批框算角度離群值（跟 OrderParser 用同一套邏輯）---
            angle_outlier_indices = set(find_angle_outliers(valid_entries, max_dev_deg=20.0))

            # --- 第三階段：迴圈畫圖，順序錯亂 or 角度離群 都畫紅色 ---
            for idx, transformed_pts in enumerate(valid_entries):
                is_bad = (not is_convex_ordered(transformed_pts)) or (idx in angle_outlier_indices)
                box_color = (0, 0, 255) if is_bad else (0, 255, 0)
                cv2.polylines(vis_frame, [transformed_pts], isClosed=True, color=box_color, thickness=2)

        # plot checked stickers
        for sticker in self.task.logic.trays[tray_id].stickers:
            if sticker.is_checked:
                bbox = np.array(sticker.bbox).astype(int)
                xy = np.array(sticker.xywhr[:2]).astype(int)
                cv2.polylines(vis_frame, bbox.reshape(-1, 4, 2), True, (0, 255, 0), 2)
                cv2.putText(vis_frame, 'OK', xy + [-50, 30], cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 255, 0), 7)

        return vis_frame, True

    @staticmethod
    def enhance(frame: np.ndarray) -> np.ndarray:
        """
        針對 crop 後的影像做強化處理。
        適用於 OCR 前的 warped_img。
        """
        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # 降噪（bilateral 保邊又快）
        frame = cv2.bilateralFilter(frame, d=5, sigmaColor=15, sigmaSpace=15)

        # 轉灰階（保留階層，不二值化）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE（保留梯度，不破壞階層）
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 轉回 BGR 送給 PaddleOCR（它預期 3 通道輸入）
        frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        return frame