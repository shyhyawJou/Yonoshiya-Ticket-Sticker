# -*- coding: utf-8 -*-
"""
ocr_backend.py
==============
把 det/cls/rec 推論包成一個共用介面，讓同一份後處理程式可以在兩個環境跑：

    edge (aarch64 + NPU)  : DLA_on_720/ocr_engine.py 的 TextSystemDLA
                            （需要 AI.cpython-312-aarch64-linux-gnu.so）
    開發機 (x86, 無 NPU)  : ONNX_on_nuc/ocr_engine.py 的 TextSystemONNX
                            （OpenVINO CPU）

重要限制
--------
AI.*.so 是 ARM aarch64 ELF，開發機（win32/x86）**無法載入**，所以本機
一律走 ONNX 後端。ONNX 與 DLA 是**不同權重檔**：

模型與字典讀取所選品牌的 config.yaml。DLA／ONNX 權重不同，
本機結果不能宣稱為 NPU 實測。本工具固定測試 DET，不套用 app 的 cv_table 路由。

TextSystemONNX.predict 沒有 DLA 版的 90 度校正，這裡補上同樣的流程
（先 det 判斷是否直式 -> 轉 90 CW 重新 det -> cls 判斷 180 -> 轉 180 重新 det），
讓兩邊的輸入條件盡量一致。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DLA_DIR = os.path.join(PROJECT_ROOT, "DLA_on_720")
ONNX_DIR = os.path.join(PROJECT_ROOT, "ONNX_on_nuc")
DEPS_DIR = os.path.join(PROJECT_ROOT, "annotation_deps")

if sys.prefix == sys.base_prefix and os.path.isdir(DEPS_DIR) and DEPS_DIR not in sys.path:
    sys.path.insert(0, DEPS_DIR)

import cv2  # noqa: E402  (需要先設好 annotation_deps 路徑)


@dataclass
class OcrOutput:
    """一次 OCR 的結果。frame 已轉正，且與 boxes 同一座標系。

    這刻意對齊 AsyncOCR._worker_loop 交給 OrderParser 的狀態：
    worker 會先把 frame_crop 依 rot_code 轉正，再把 is_flip 設為 False。
    """

    frame: np.ndarray
    boxes: List[np.ndarray]
    rec: List[Tuple[str, float]]
    rot_deg: int          # 為了轉正而套用的角度 (0/90/180/270, 影像順時針)
    elapsed_ms: float
    backend: str


class OcrBackend:
    """det/cls/rec 的統一入口。"""

    def __init__(self, prefer: str = "auto", task_dir: Optional[str] = None, profile: Optional[str] = None):
        if DLA_DIR not in sys.path:
            sys.path.insert(0, DLA_DIR)
        from profile_config import load_profile_config
        self.cfg = load_profile_config(DLA_DIR, profile)
        self.profile = self.cfg.profile_name
        self.kind = None
        self._sys = None
        self._load(prefer, task_dir)

    # ------------------------------------------------------------------
    def _load(self, prefer: str, task_dir: Optional[str]):
        errors = []
        if prefer in ("auto", "dla"):
            try:
                self._load_dla(task_dir)
                return
            except Exception as exc:
                errors.append(f"dla: {exc}")
                if prefer == "dla":
                    raise RuntimeError(f"無法載入 DLA 後端: {exc}") from exc
        if prefer in ("auto", "onnx"):
            try:
                self._load_onnx(task_dir)
                return
            except Exception as exc:
                errors.append(f"onnx: {exc}")
        raise RuntimeError("沒有可用的 OCR 後端: " + "; ".join(errors))

    def _engine_class(self, directory, name):
        # Distinct module names prevent a failed DLA import poisoning ONNX fallback.
        import importlib.util
        module_name = '_probe_' + name
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, os.path.join(directory, 'ocr_engine.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        return getattr(sys.modules[module_name], name)

    def _load_dla(self, task_dir):
        from config import load_config
        cfg = load_config(os.path.join(task_dir, 'config.yaml')) if task_dir else self.cfg
        model = cfg.runtime.model
        engine = self._engine_class(DLA_DIR, 'TextSystemDLA')
        self._sys = engine(model.ocr_det, model.ocr_cls, model.ocr_rec, model.text)
        self.kind = 'dla'

    def _load_onnx(self, task_dir):
        from pathlib import Path
        import yaml
        td = Path(task_dir) if task_dir else Path(ONNX_DIR)/'tasks'/self.profile
        data = yaml.safe_load((td/'config.yaml').read_text(encoding='utf-8-sig'))
        model = data['runtime']['model']
        paths = [str((td/model[key]).resolve()) for key in ('ocr_det', 'ocr_cls', 'ocr_rec', 'text')]
        engine = self._engine_class(ONNX_DIR, 'TextSystemONNX')
        self._sys = engine(*paths)
        self.kind = 'onnx'

    # ------------------------------------------------------------------
    def _det(self, img):
        boxes, _ = self._sys.run_det(img)
        return boxes

    def _crop(self, frame, boxes, obj_type):
        """排序 + 裁切。DLA 版有 _postprocess_and_crop / _filter_sticker_boxes，
        ONNX 版沒有，所以這裡用兩邊都有的 sorted_boxes / get_rotate_crop_image
        自行組出同樣的流程，sticker 過濾在可用時才套用。"""
        helper = getattr(self._sys, "_postprocess_and_crop", None)
        if helper is not None:
            return helper(frame, boxes, obj_type)

        if obj_type == "sticker":
            filt = getattr(self._sys, "_filter_sticker_boxes", None)
            if filt is not None:
                boxes = filt(boxes)
        boxes = self._sys.sorted_boxes(boxes)
        crops = [self._sys.get_rotate_crop_image(frame, np.asarray(b, dtype=np.float32))
                 for b in boxes]
        return boxes, crops

    def _needs_90(self, boxes) -> bool:
        """與 TextSystemDLA._check_boxes_need_rotate_90 共用同一個判準函式，
        避免 edge 與開發機兩邊各自漂移。"""
        from logic.orientation import needs_quarter_turn
        return needs_quarter_turn(boxes)

    def predict(self, image: np.ndarray, obj_type: str = "ticket") -> Optional[OcrOutput]:
        t0 = time.perf_counter()
        rot = 0
        frame = image.copy()

        boxes = self._det(frame)
        if boxes is None or len(boxes) == 0:
            return None

        if self._needs_90(boxes):
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rot = 90
            boxes = self._det(frame)
            if boxes is None or len(boxes) == 0:
                return None

        boxes, crops = self._crop(frame, boxes, obj_type)
        if not crops:
            return None
        crops, _cls, flip_count, _ = self._sys.run_cls(crops)
        if flip_count / max(1, len(crops)) > 0.5:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            rot = (rot + 180) % 360
            boxes = self._det(frame)
            if boxes is None or len(boxes) == 0:
                return None
            boxes, crops = self._crop(frame, boxes, obj_type)
            if not crops:
                return None
            crops, _cls, _, _ = self._sys.run_cls(crops)

        # DLA 版是 run_rec，ONNX 版是 run_rec_pool（多執行緒池），名稱不同
        run_rec = getattr(self._sys, "run_rec", None) or self._sys.run_rec_pool
        rec, _ = run_rec(crops)
        return OcrOutput(
            frame=frame,
            boxes=[np.asarray(b, dtype=float) for b in boxes],
            rec=[(t, float(s)) for t, s in rec],
            rot_deg=rot,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            backend=self.kind,
        )


def load_vocab(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            w = line.strip()
            if w and not w.startswith("#"):
                out.append(w)
    return out
