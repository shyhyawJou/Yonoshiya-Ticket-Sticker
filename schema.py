# schema.py
from dataclasses import dataclass
from typing import Tuple, Optional

XYXYXYXY = Tuple[float, float, float, float, float, float, float, float]
XYWHR = Tuple[float, float, float, float, float]

@dataclass(frozen=True)
class Detection:
    xyxy: XYXYXYXY     
    xywhr: XYWHR
    cls_id: int
    cls_name: str
    conf: float
    frame_id: Optional[int] = None
