from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from loguru import logger
import os
import yaml
from pathlib import Path

# ---------- Data Classes ----------

@dataclass
class MqttCfg:
    """MQTT Broker Configuration"""
    host: str
    port: int
    base_topic: str
    session: str

@dataclass
class ModelCfg:
    """AI Model and Text Configuration"""
    object_det: str
    ocr_det: str
    ocr_cls: str
    ocr_rec: str
    text: str

@dataclass
class CameraCfg:
    """Camera Source Configuration"""
    source: Any
    width: int
    height: int
    device: str  # 'aravis' or 'hik'


@dataclass
class DepthCamCfg:
    """Detph Camera Source Configuration"""
    cfg: dict


@dataclass
class StreamCfg:
    """Output Stream Configuration"""
    port: int
    stream_size: list

@dataclass
class RuntimeCfg:
    """Runtime configurations including model, camera, and stream"""
    model: ModelCfg
    camera: CameraCfg
    depth_cam: DepthCamCfg
    stream: StreamCfg

@dataclass
class ThresholdsCfg:
    """Thresholds for AI inference and object assignment"""
    ai_conf: float
    ai_iou: float

@dataclass
class PlacementCfg:
    """Geometric validation rules"""
    roi_strink: float
    iou_assign: float
    iou_candidate: float
    drift_iou_thresh: float

@dataclass
class StabilityCfg:
    """Configuration for temporal stability checks"""
    n_settle_frame: int    
    k_container_fail: int
    k_container_new: int
    tray_missing_frame: int
    sticker_missing_frame: int
    ticket_leave_frame: int

@dataclass
class ClassInfo:
    """Holds the ID and name for a single class from classes.yaml"""
    id: int
    name: str  

@dataclass
class MappingCasesCfg:
    """組合品項拆分規則：當 OCR 辨識到此品項名稱時，展開為子品項清單"""
    name: str
    sub_items: List[str]

@dataclass
class RoiCfg:
    """
    單一 ROI 區域的定義:四個角點的多邊形(可以是旋轉矩形),
    座標格式跟專案既有的 PolygonXYXY 一致(4 點攤平成 8 個數字),
    這樣可以直接餵給 geometry.py 裡的 iou_poly_poly / is_center_in_polygon 等工具,
    不需要為 ROI 另外寫一套幾何運算。
    """
    name: str
    points: List[List[float]]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    color: Optional[List[int]] = None  # 前端/overlay 繪圖用的 BGR 顏色,不填則由繪圖端指定預設色

    def to_polygon_xyxy(self) -> Tuple[float, ...]:
        """攤平成 (x1,y1,x2,y2,x3,y3,x4,y4) 供 geometry 工具直接使用"""
        flat: List[float] = []
        for pt in self.points:
            flat.extend([float(pt[0]), float(pt[1])])
        return tuple(flat)


@dataclass
class PresetRoiCfg:
    """
    preset_roi 模式專用的 ROI 集合。

    目前只有 ticket_roi 會影響訂單生命週期判斷(訂單是否進入/離開這個
    區域,決定要不要開始處理、要不要視為完成/結束)。其餘 ROI(辨識物件
    區域、餐品打包區域...)先預留欄位,之後要串接對應行為時,直接在
    PresetRoiTracker / TrayStateMachine 裡讀取對應欄位即可,不需要再
    回頭改這裡的結構或 load_config 的解析邏輯。
    """
    ticket_roi: RoiCfg
    object_roi: Optional[RoiCfg] = None    # 預留:辨識物件的偵測/篩選區域
    packing_roi: Optional[RoiCfg] = None   # 預留:餐品打包完成區域

@dataclass
class CameraParmCfg:
    """Camera parameters"""
    GainAuto: str
    Gain: float
    ExposureAuto: str
    ExposureTime: float
    BalanceWhiteAuto: str

@dataclass
class Config:
    """Main configuration object aggregating all settings"""
    mqtt: MqttCfg
    runtime: RuntimeCfg
    thresholds: ThresholdsCfg
    placement: PlacementCfg
    stability: StabilityCfg
    camera_params: CameraParmCfg
    bright_ctrl: dict
    classes: List[ClassInfo]
    menus_ticket: List[str]
    menus_sticker: List[str]
    menus_mapping: List[MappingCasesCfg]
    # "tray": 原本綁 tray 盤的狀態機; "single": 單一訂單、不需要 tray 盤的狀態機
    mode: str = "tray"
    preset_roi: Optional[PresetRoiCfg] = None   # 新增:僅 mode == "preset_roi" 時才會有值

# ---------- Configuration Loader ----------

def _resolve(base: Path, p: str | None) -> str | None:
    """Resolves a path relative to the base directory if it's not absolute."""
    return str((base / p).resolve())

def _load_yaml(path: str) -> dict:
    """Loads a YAML file and returns its content."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data

def load_config(config_path: str) -> Config:
    """
    Loads and parses all YAML configuration files into a structured Config object.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_dir = Path(config_path).resolve().parent
    base = _load_yaml(config_path)

    try:
        # Load Classes
        inc = base["includes"]
        classes: List[ClassInfo] = []
        if inc["classes"]:
            classes_path = os.path.join(base_dir, inc["classes"])
            classes_yaml = _load_yaml(classes_path)
            classes_data = classes_yaml["classes"]
            for c in classes_data:
                classes.append(ClassInfo(
                    id=int(c["id"]),
                    name=str(c["name"])
                ))

        # Load Mapping Case Menus
        mapping_cases: List[MappingCasesCfg] = []
        if inc["menus_mapping"]:
            mc_path = os.path.join(base_dir, inc["menus_mapping"])
            mc_yaml = _load_yaml(mc_path)
            for mc in mc_yaml["mapping"]:
                mapping_cases.append(MappingCasesCfg(
                    name=str(mc["name"]),
                    sub_items=[str(s) for s in mc["sub_items"]]
                ))

        menus_ticket: List[str] = []
        menus_sticker_set = set() 

        for mc in mapping_cases:
            menus_ticket.append(mc.name) 
            for sub in mc.sub_items:
                menus_sticker_set.add(sub) 

        menus_sticker: List[str] = list(menus_sticker_set)

        # Parse MQTT settings
        m = base["mqtt"]
        mqtt = MqttCfg(
            host=m["host"],
            port=int(m["port"]),
            base_topic=m["base_topic"],
            session=m["session"]
        )

        # Parse Runtime settings
        rt = base["runtime"]
        model = rt["model"]
        camera = rt["camera"]
        stream = rt["stream"]

        object_det_path = model["object_det"]
        ocr_det_path = model["ocr_det"]
        ocr_cls_path = model["ocr_cls"]
        ocr_rec_path = model["ocr_rec"]
        text_path = model["text"]

        # depth camera setting
        depth_cam = base["depth_camera"]

        runtime = RuntimeCfg(
            model=ModelCfg(
                object_det=_resolve(base_dir, object_det_path),
                ocr_det=_resolve(base_dir, ocr_det_path),
                ocr_cls=_resolve(base_dir, ocr_cls_path),
                ocr_rec=_resolve(base_dir, ocr_rec_path),
                text=_resolve(base_dir, text_path),
            ),
            camera=CameraCfg(
                source=camera["source"],
                width=camera["width"],
                height=camera["height"],
                device=camera["device"]
            ),
            depth_cam=DepthCamCfg(depth_cam),
            stream=StreamCfg(port=int(stream["port"]), stream_size=stream["stream_size"])
        )

        # Thresholds
        thr = base["thresholds"]
        thresholds = ThresholdsCfg(
            ai_conf=float(thr["ai_conf"]),
            ai_iou=float(thr["ai_iou"])
        )

        # Placement
        pl = base["placement"]
        placement = PlacementCfg(
            roi_strink=float(pl["roi_strink"]),
            iou_assign=float(pl["iou_assign"]),
            iou_candidate = float(pl["iou_candidate"]),
            drift_iou_thresh=float(pl["drift_iou_thresh"]),
        )

        # Stability
        st = base["stability"]
        stability = StabilityCfg(
            n_settle_frame=int(st["n_settle_frame"]),
            k_container_fail=int(st["k_container_fail"]),
            k_container_new=int(st["k_container_new"]),
            tray_missing_frame=int(st["tray_missing_frame"]),
            sticker_missing_frame=int(st["sticker_missing_frame"]),
            ticket_leave_frame=int(st["ticket_leave_frame"]),
        )

        # Mode: 決定要啟用哪一種狀態機 ("tray" | "single" | "preset_roi")
        # 舊的 config.yaml 沒有這個欄位也沒關係，預設當作 "tray"，行為完全不變。
        mode = str(base.get("mode", "tray")).strip().lower()
        if mode not in ("tray", "single", "preset_roi"):
            logger.warning(f"設定檔 mode 值不合法: '{mode}'，退回預設值 'tray'")
            mode = "tray"

        # Preset ROI(僅 preset_roi 模式需要;其餘模式底下若 yaml 有寫,仍會解析但不會被使用)
        preset_roi_cfg: Optional[PresetRoiCfg] = None
        if "preset_roi" in base:
            pr = base["preset_roi"]

            def _parse_roi(name: str, roi_data: dict) -> RoiCfg:
                return RoiCfg(
                    name=name,
                    points=[[float(x), float(y)] for x, y in roi_data["points"]],
                    color=roi_data.get("color"),
                )

            if "ticket_roi" not in pr:
                raise ValueError("preset_roi 設定缺少必要欄位 'ticket_roi'")

            preset_roi_cfg = PresetRoiCfg(
                ticket_roi=_parse_roi("ticket_roi", pr["ticket_roi"]),
                object_roi=_parse_roi("object_roi", pr["object_roi"]) if "object_roi" in pr else None,
                packing_roi=_parse_roi("packing_roi", pr["packing_roi"]) if "packing_roi" in pr else None,
            )

        if mode == "preset_roi" and preset_roi_cfg is None:
            raise ValueError("mode 設為 'preset_roi' 時，設定檔必須提供 'preset_roi.ticket_roi'")
        # Camera
        ca = base["camera_params"]
        bright_ctrl = base["birghtness_controller"]
        #camera_params = CameraParmCfg(
        #    GainAuto = str(ca["GainAuto"]),
        #    Gain = float(ca["Gain"]),
        #    ExposureAuto = str(ca["ExposureAuto"]),
        #    ExposureTime = float(ca["ExposureTime"]),
        #    BalanceWhiteAuto = str(ca["BalanceWhiteAuto"])
        #)

        return Config(
            mqtt=mqtt,
            runtime=runtime,
            thresholds=thresholds,
            placement=placement,
            stability=stability,
            camera_params=ca,
            classes=classes,
            menus_ticket=menus_ticket,
            menus_sticker=menus_sticker,
            menus_mapping=mapping_cases,
            mode=mode,
            bright_ctrl=bright_ctrl,
            preset_roi=preset_roi_cfg,
        )

    except KeyError as e:
        error_msg = f"設定檔缺失必要參數: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        logger.error(f"解析設定檔時發生非預期錯誤: {e}")
        raise