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