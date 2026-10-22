"""
Step 3 整合測試用的假依賴：Detection / MqttBus / mmr / CsvWriter / Config
讓我們可以在沒有真實硬體/模型的情況下，跑完整的 update() + apply_ocr_result() 流程。
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Detection:
    cls_name: str
    xyxy: tuple
    xywhr: tuple


class FakeScalar(float):
    """模擬 numpy scalar：本身是 float（可參與 Shapely 幾何運算），
    同時支援 .item()（NEW_TRAY_DETECTED payload 組裝時會呼叫）。"""
    def item(self):
        return float(self)


def make_xyxy(x1, y1, x2, y2):
    """建立一個水平矩形的 8 點座標，每個元素都是 FakeScalar。"""
    return tuple(FakeScalar(v) for v in (x1, y1, x2, y1, x2, y2, x1, y2))


class FakeMqttBus:
    def __init__(self):
        self.system_events = []
        self.det_status_events = []

    def publish_system(self, payload):
        self.system_events.append(payload)

    def publish_det_status(self, payload):
        self.det_status_events.append(payload)


class FakeMmr:
    def crop_by_angle(self, frame, cx, cy, w, h, r):
        return None, None


class FakeCsvWriter:
    def __init__(self, log_dir=None):
        self.logs = []

    def log(self, payload):
        self.logs.append(payload)


@dataclass
class FakeStability:
    n_settle_frame: int = 3
    k_container_fail: int = 2
    k_container_new: int = 2
    tray_missing_frame: int = 5


@dataclass
class FakePlacement:
    roi_strink: float = 0.1
    iou_assign: float = 0.3
    iou_candidate: float = 0.3
    drift_iou_thresh: float = 0.5


@dataclass
class FakeMenuMapping:
    name: str
    sub_items: List[str]


@dataclass
class FakeConfig:
    menus_ticket: List[str] = field(default_factory=lambda: ["味噌ラーメン", "醤油ラーメン"])
    menus_sticker: List[str] = field(default_factory=lambda: ["味噌ラーメン", "醤油ラーメン"])
    menus_mapping: List[FakeMenuMapping] = field(default_factory=list)
    stability: FakeStability = field(default_factory=FakeStability)
    placement: FakePlacement = field(default_factory=FakePlacement)
