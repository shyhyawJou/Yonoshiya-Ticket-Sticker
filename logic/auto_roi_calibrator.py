"""
auto_roi_calibrator.py
======================
AutoRoiCalibrator:開機時用現場某個實體標記的偵測框,自動算出一個 preset_roi。

為什麼需要:ROI 原本要靠人在設定檔裡填座標(或由 App 拉框設定),每換一個
站點、每動一次相機就得重新量一次。但現場本來就有固定貼在檯面上的實體標記
(目前是 ticket_mat 傳票墊),偵測器也已經認得它 —— 直接拿它的位置當 ROI
基準,現場就不用再手動對座標。

要盯哪一個類別由呼叫端指定(設定檔 preset_roi.<roi>.auto.class_name),
不是寫死 ticket_mat:判斷訂單結束用的實體標記以後可能會換成別的物件,
換的時候只要改設定檔。同理,每個 ROI 可以各自盯不同類別、各自決定要不要
自動校正。

刻意保留的兩個設計:

1. **輸出正矩形,不是旋轉框。** 物件偵測給的是旋轉框,但現場慣例是把 ROI
   拉得比物件大一點、方向水平,留一點操作彈性(傳票不會每次都跟墊子完全
   平行),所以這裡取旋轉框的外接正矩形再往外擴 padding_ratio。

2. **一次性。** 只在啟動後跑一次,定案(或逾時)就停,不會每幀都在動 ROI ——
   ROI 一直漂移的話,訂單生命週期判斷會跟著抖。要重跑就再 arm() 一次
   (MQTT: cmd/preset_roi {"action": "auto", "name": "<roi 名稱>"})。

這個類別只做幾何與計數,不碰設定檔、不碰 MQTT、不碰 tracker:定案後回傳
四個點,由呼叫端(StreamManager -> TaskContext)決定怎麼套用與落地,
這樣它可以完全不依賴硬體單獨測試。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from loguru import logger

from logic.geometry import PolygonXYXY, iou_poly_poly, polygon_area


class AutoRoiCalibrator:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        class_name: str = "ticket_mat",
        roi_name: str = "ticket_roi",
        padding_ratio: float = 0.12,
        confirm_frames: int = 15,
        stable_iou: float = 0.9,
        timeout_frames: int = 600,
    ):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.class_name = str(class_name)
        self.roi_name = str(roi_name)
        self.padding_ratio = float(padding_ratio)
        self.confirm_frames = max(1, int(confirm_frames))
        self.stable_iou = float(stable_iou)
        self.timeout_frames = int(timeout_frames)

        self._active = False
        self._fed_frames = 0
        self._streak = 0
        self._last_poly: Optional[PolygonXYXY] = None

    # ------------------------------------------------------------
    # 狀態
    # ------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        return self._active

    def arm(self) -> None:
        """(重新)啟動一次校正。已經在跑就只是把計數歸零重來。"""
        self._active = True
        self._fed_frames = 0
        self._streak = 0
        self._last_poly = None
        logger.info(
            f"[AUTO ROI] 開始自動校正 {self.roi_name}:等 {self.class_name} 連續 "
            f"{self.confirm_frames} 幀穩定 (IoU>={self.stable_iou}), "
            f"外擴 {self.padding_ratio:.0%}"
        )

    def cancel(self, reason: str = "") -> None:
        self._active = False
        self._streak = 0
        self._last_poly = None
        if reason:
            logger.info(f"[AUTO ROI] 停止自動校正 {self.roi_name}: {reason}")

    # ------------------------------------------------------------
    # 每幀餵資料
    # ------------------------------------------------------------
    def feed(self, detections: Sequence) -> Optional[List[List[int]]]:
        """
        餵入這一幀的所有偵測(只有目標類別的偵測也可以),
        回傳定案的四個點 [[x,y] x4] = 左上/右上/右下/左下;
        還沒定案就回 None。定案後自動停止,不需要呼叫端額外關閉。
        """
        if not self._active:
            return None

        self._fed_frames += 1
        if 0 < self.timeout_frames < self._fed_frames:
            logger.warning(
                f"[AUTO ROI] 等了 {self._fed_frames} 幀仍未取得穩定的 {self.class_name},"
                f"放棄自動校正,{self.roi_name} 沿用設定檔既有的座標"
            )
            self.cancel()
            return None

        target_poly = self._pick_target(detections)
        if target_poly is None:
            self._streak = 0
            self._last_poly = None
            return None

        if self._last_poly is not None and iou_poly_poly(target_poly, self._last_poly) >= self.stable_iou:
            self._streak += 1
        else:
            self._streak = 1
        self._last_poly = target_poly

        if self._streak < self.confirm_frames:
            return None

        points = self._expand_to_rect(target_poly)
        logger.success(
            f"[AUTO ROI] {self.class_name} 已穩定 {self._streak} 幀,"
            f"自動定出 {self.roi_name} = {points}"
        )
        self.cancel()
        return points

    # ------------------------------------------------------------
    # 內部
    # ------------------------------------------------------------
    def _pick_target(self, detections: Sequence) -> Optional[PolygonXYXY]:
        """
        取畫面上面積最大的那個目標物件。

        現場只會有一個,但偵測偶爾會多冒一個小碎框;取最大面積比取第一個
        穩定,不然 ROI 可能被一個只有幾十個像素的雜訊框決定。
        """
        targets = [d for d in detections if getattr(d, "cls_name", None) == self.class_name]
        if not targets:
            return None
        best = max(targets, key=lambda d: polygon_area(d.xyxy))
        return best.xyxy

    def _expand_to_rect(self, poly: PolygonXYXY) -> List[List[int]]:
        """旋轉框 -> 外接正矩形 -> 外擴 padding_ratio -> 夾回畫面範圍內"""
        xs = poly[0::2]
        ys = poly[1::2]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad_x = (x2 - x1) * self.padding_ratio
        pad_y = (y2 - y1) * self.padding_ratio

        x1 = int(round(max(0.0, x1 - pad_x)))
        y1 = int(round(max(0.0, y1 - pad_y)))
        x2 = int(round(min(float(self.frame_width - 1), x2 + pad_x)))
        y2 = int(round(min(float(self.frame_height - 1), y2 + pad_y)))

        # 左上 / 右上 / 右下 / 左下：跟 App 溝通好的點序
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
