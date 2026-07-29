"""
image_utils.py
===============
影像編碼小工具，跟 tray / OCR 業務邏輯無關，純粹是
「numpy 影像 -> base64 jpg 字串」的轉換，供 log payload 使用。
"""

from __future__ import annotations

from typing import Optional

import base64
import cv2
import numpy as np
from loguru import logger


def encode_image_base64(img: Optional[np.ndarray], quality: int = 80) -> str:
    """把一張影像編碼成 base64 jpg 字串；失敗或輸入為 None 時回傳空字串。"""
    if img is None:
        return ""
    try:
        success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if success:
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encode error: {e}")
    return ""

"""
輕量級影像內容比對工具:用感知雜湊(dHash)取代逐像素比對或 SSIM,
避免在效能有限的裝置上持續做重運算的影像比對。
 
用途:偵測「同一個追蹤框內的實際內容是否被置換」(例如貼紙被疊上
新的餐點),而不是用來做精確的影像品質分析,因此刻意選用計算量
最低、對局部亮度飄動容忍度較好的方法——比較的是相鄰像素的
相對亮暗關係,而非絕對數值,對局部反光/些微曝光飄動較不敏感。
"""
import cv2

def compute_dhash(image: np.ndarray, hash_size: int = 16) -> int:
    """
    計算 difference hash:縮小到 (hash_size+1) x hash_size 灰階,
    比較每一列相鄰像素的亮暗關係,組成一組 bit 序列,
    回傳整數表示(方便用 XOR + bit count 算漢明距離)。
    """
    if image is None or image.size == 0:
        return 0
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
 
    diff = resized[:, 1:] > resized[:, :-1]
    bits = diff.flatten()
 
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value
 
 
def hamming_distance(hash_a: int, hash_b: int) -> int:
    """兩組雜湊值的漢明距離,數值越大代表內容差異越大"""
    return bin(hash_a ^ hash_b).count("1")



def preprocess_for_diff(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    
    # 1. 轉灰階 (比對文字內容不用 BGR，灰階更穩定且效能好)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. 高斯模糊 (消除 1~2px 的微小抖動與高頻雜訊)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return blurred.astype(np.float32)


def compute_change_ratio(
    ref: Optional[np.ndarray],
    cur: Optional[np.ndarray],
    pixel_diff_thresh: float = 30.0, # 轉灰階後可稍微調高門檻 (例如 25.0 ~ 35.0)
    content_thresh: float = 200.0,
) -> float:
    if ref is None or cur is None:
        return 1.0

    # 1. 強制對齊尺寸 (避免形狀微小不一致)
    if ref.shape != cur.shape:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))

    # 2. 計算像素絕對差值
    diff = np.abs(ref - cur)

    # 3. 定義內容區域 (加上形態學膨脹，避免邊緣抖動擴張)
    content_mask = (ref < content_thresh) | (cur < content_thresh)
    
    # 使用膨脹將 Mask 邊緣稍微擴展 1~2 px，允許邊緣容錯
    kernel = np.ones((3, 3), np.uint8)
    content_mask_dilated = cv2.dilate(content_mask.astype(np.uint8), kernel).astype(bool)

    mask_pixel_count = int(np.count_nonzero(content_mask_dilated))
    if mask_pixel_count == 0:
        return 0.0

    # 4. 統計超過差異門檻的像素
    exceeds = diff > pixel_diff_thresh
    changed_in_mask = int(np.count_nonzero(exceeds & content_mask_dilated))

    return changed_in_mask / mask_pixel_count