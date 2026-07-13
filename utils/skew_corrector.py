"""
SkewCorrector
=============
用途：
    當一張裁切影像（例如由 mmrotate 依 OBB 框裁出的訂單/貼紙影像）
    因為偵測框角度不完全貼合物件，導致整體內容仍帶有一個小角度的
    傾斜時，PaddleOCR 產生的所有文字框（dt_boxes）通常會「一致地」
    帶有同一個傾斜角度 theta。

    這個 class 會：
        1. 從一批 dt_boxes 自動估計出整體傾斜角度 theta（取中位數，
           對少數雜訊框有抵抗力）。
        2. 提供 correct_point / correct_box，把座標「反向旋轉 -theta」，
           摆正成真正水平垂直的座標系，讓後續依賴水平垂直假設的邏輯
           （例如同一行判斷、外帶框過濾、找最近的數字框）重新有效。

    注意：
        - 摆正後的座標只應該用在「邏輯判斷」上（比較 cy / cx / 行高等）。
        - 若需要對原始影像做實際裁切（例如丟進 rec_pred.predict_quantity），
          仍必須使用「原始未旋轉」的真實像素座標，因為影像本身沒有被
          真的旋轉，只是座標的參考系被摆正了。

用法範例：
    skew = SkewCorrector.from_dt_boxes(dt_boxes)

    for dt_box, (text, score) in zip(dt_boxes, rec_res):
        corrected_box = skew.correct_box(dt_box)
        cy_logic = np.mean([pt[1] for pt in corrected_box])   # 邏輯判斷用
        cx_real  = np.mean([pt[0] for pt in dt_box])           # 真實裁圖用
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]
Box = Sequence[Sequence[float]]  # 4 個點的多邊形框，例如 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


@dataclass
class SkewCorrector:
    """
    估計並校正一批文字框的整體傾斜角度。

    Attributes:
        theta: 反向旋轉角度（弧度）。0.0 代表不需要 / 不應該校正。
        pivot: 旋轉校正時使用的中心點 (x, y)。
    """

    theta: float
    pivot: Point

    # ------------------------------------------------------------
    # 建構子
    # ------------------------------------------------------------
    @classmethod
    def from_dt_boxes(
        cls,
        dt_boxes: List[Box],
        max_std_deg: float = 15.0,
        max_theta_deg: float = 30.0,
    ) -> "SkewCorrector":
        """
        從一批 dt_boxes 估計整體傾斜角度。

        做法：
            每個框的「上邊」（第0點 -> 第1點，一般為左上->右上）理論上
            應該接近水平（角度 ~ 0）。若整張影像整體歪了 theta，則所有
            框的上邊角度都會偏移 theta。取這些角度的中位數作為估計值，
            比平均值更能抵抗少數方向錯誤或雜訊框的干擾。

        Args:
            dt_boxes: PaddleOCR 偵測到的文字框列表，每個框為 4 個點。
            max_std_deg: 若所有框角度的標準差超過此值（度），代表框的方向
                不一致（可能不是整體傾斜，而是多方向文字混雜），此時不校正，
                theta 設為 0，避免誤傷正常資料。
            max_theta_deg: 若估計出的角度超過此值（度），視為異常估計
                （例如整批框幾乎垂直排列被誤判），保守起見不校正。

        Returns:
            SkewCorrector 實例。若無法估計（框數不足或角度不合理），
            回傳 theta=0.0（等同不做任何校正，行為與原本一致）。
        """
        angles: List[float] = []

        for box in dt_boxes:
            box = np.asarray(box, dtype=np.float64)
            if box.shape[0] < 2:
                continue
            dx = box[1][0] - box[0][0]
            dy = box[1][1] - box[0][1]
            if dx == 0.0 and dy == 0.0:
                continue
            angles.append(float(np.arctan2(dy, dx)))

        if not angles:
            return cls(theta=0.0, pivot=(0.0, 0.0))

        angles_arr = np.array(angles)
        theta = float(np.median(angles_arr))

        # 角度離散程度過大 -> 不是穩定的整體傾斜，放棄校正
        if np.degrees(np.std(angles_arr)) > max_std_deg:
            theta = 0.0

        # 角度本身過大 -> 可能估計異常，保守起見不校正
        if abs(np.degrees(theta)) > max_theta_deg:
            theta = 0.0

        all_pts = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 2)
        pivot = (float(all_pts[:, 0].mean()), float(all_pts[:, 1].mean()))

        return cls(theta=theta, pivot=pivot)

    # ------------------------------------------------------------
    # 校正函式
    # ------------------------------------------------------------
    def correct_point(self, x: float, y: float) -> Point:
        """將單一點座標反向旋轉 -theta，回傳摆正後座標。"""
        if self.theta == 0.0:
            return float(x), float(y)

        px, py = self.pivot
        c = np.cos(-self.theta)
        s = np.sin(-self.theta)
        dx = x - px
        dy = y - py
        new_x = px + dx * c - dy * s
        new_y = py + dx * s + dy * c
        return float(new_x), float(new_y)

    def correct_box(self, box: Box) -> List[Point]:
        """將一個 4 點文字框整體摆正，回傳新的點列表。"""
        return [self.correct_point(pt[0], pt[1]) for pt in box]

    def correct_boxes(self, boxes: List[Box]) -> List[List[Point]]:
        """批次校正多個框。"""
        return [self.correct_box(b) for b in boxes]
        
    def inverse_correct_point(self, x: float, y: float) -> Point:
        """將摆正後的邏輯座標反向旋轉 (+theta) 還原回原始傾斜影像的真實座標。"""
        if self.theta == 0.0:
            return float(x), float(y)

        px, py = self.pivot
        # 注意這裡是用 +self.theta，與 correct_point 的 -self.theta 相反
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        dx = x - px
        dy = y - py
        real_x = px + dx * c - dy * s
        real_y = py + dx * s + dy * c
        return float(real_x), float(real_y)

    # ------------------------------------------------------------
    # 便利屬性
    # ------------------------------------------------------------
    @property
    def is_active(self) -> bool:
        """是否實際套用了非零校正角度。"""
        return self.theta != 0.0

    @property
    def theta_deg(self) -> float:
        """回傳角度（度），方便 log 除錯用。"""
        return float(np.degrees(self.theta))


# ------------------------------------------------------------
# 簡單自我測試（可直接執行此檔驗證邏輯）
# ------------------------------------------------------------
if __name__ == "__main__":
    import math

    def rotate_pt(x, y, cx, cy, deg):
        rad = math.radians(deg)
        dx, dy = x - cx, y - cy
        nx = cx + dx * math.cos(rad) - dy * math.sin(rad)
        ny = cy + dx * math.sin(rad) + dy * math.cos(rad)
        return [nx, ny]

    # 模擬一個整體傾斜 8 度的訂單畫面：兩列文字，每列本應同高
    cx, cy = 300, 300
    deg = 8.0

    raw_boxes = [
        # row 1: y=100, x=100~180 與 x=250~330（同一列，理論上 cy 應相同）
        [[100, 100], [180, 100], [180, 130], [100, 130]],
        [[250, 100], [330, 100], [330, 130], [250, 130]],
        # row 2: y=200
        [[100, 200], [180, 200], [180, 230], [100, 230]],
        [[250, 200], [330, 200], [330, 230], [250, 230]],
    ]

    skewed_boxes = [
        [rotate_pt(x, y, cx, cy, deg) for x, y in box]
        for box in raw_boxes
    ]

    skew = SkewCorrector.from_dt_boxes(skewed_boxes)
    print(f"估計角度: {skew.theta_deg:.2f} 度 (預期接近 {deg} 度)")

    corrected = skew.correct_boxes(skewed_boxes)
    for i in range(0, len(corrected), 2):
        cy_a = np.mean([p[1] for p in corrected[i]])
        cy_b = np.mean([p[1] for p in corrected[i + 1]])
        print(f"row {i//2}: cy_a={cy_a:.2f}, cy_b={cy_b:.2f}, diff={abs(cy_a - cy_b):.2f}")