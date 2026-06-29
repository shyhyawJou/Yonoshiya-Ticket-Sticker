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

@dataclass
class StreamCfg:
    """Output Stream Configuration"""
    port: int

@dataclass
class RuntimeCfg:
    """Runtime configurations including model, camera, and stream"""
    model: ModelCfg
    camera: CameraCfg
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
    classes: List[ClassInfo]
    menus_ticket: List[str]
    menus_sticker: List[str]
    menus_mapping: List[MappingCasesCfg]


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
                height=camera["height"]
            ),
            stream=StreamCfg(port=int(stream["port"]))
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
            drift_iou_thresh=float(pl["drift_iou_thresh"])
        )

        # Stability
        st = base["stability"]
        stability = StabilityCfg(
            n_settle_frame=int(st["n_settle_frame"]),
            k_container_fail=int(st["k_container_fail"]),
            k_container_new=int(st["k_container_new"]),
            tray_missing_frame=int(st["tray_missing_frame"])
        )

        # Camera
        ca = base["camera_params"]
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
            menus_mapping=mapping_cases
        )

    except KeyError as e:
        error_msg = f"設定檔缺失必要參數: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    except Exception as e:
        logger.error(f"解析設定檔時發生非預期錯誤: {e}")
        raise
