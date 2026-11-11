"""
test_stub_integration.py
========================
針對客戶聯（stub / small_order_ticket）整合的單元與流程測試：
1. stub_cv_det  : OpenCV 水平線偵測、右側墨跡團塊偵測、垂直對齊數量欄補框、MMR 白紙邊界緊貼與候選推算
2. stub_parser  : 表頭時間/單號解析、POS 反查與易混淆修正、21->1 校正、垂直對齊重讀、fallback_rects、to_order_json
3. stub_reader & TrayStateMachine : 無主票時獨立觸發 stub OCR、主票確認後自動交叉驗證 verify_stub
"""

from __future__ import annotations

import math
import os
import sys
import types
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 若執行環境尚未安裝完整第三方 C-extension 套件（例如無 numpy/cv2 的輕量環境），提供最小測試替身以驗證業務與幾何邏輯
try:
    import numpy as np
except ImportError:
    np_stub = types.ModuleType("numpy")
    np_stub.pi = math.pi
    np_stub.float32 = float
    np_stub.float64 = float
    np_stub.int32 = int
    np_stub.uint8 = int

    class _NDArray:
        def __init__(self, data, shape=None, dtype=float):
            self.dtype = dtype
            if isinstance(data, _NDArray):
                self._flat = list(data._flat)
                self.shape = shape or data.shape
                return
            if shape is not None:
                self.shape = tuple(shape)
                if isinstance(data, (int, float)):
                    total = 1
                    for d in self.shape:
                        total *= d
                    self._flat = [dtype(data)] * total
                else:
                    self._flat = [dtype(x) for x in data]
                return
            self.shape, self._flat = self._infer(data, dtype)

        @staticmethod
        def _infer(obj, dtype):
            if not isinstance(obj, (list, tuple)):
                return (), [dtype(obj)]
            if len(obj) == 0:
                return (0,), []
            sub_shape, flat = _NDArray._infer(obj[0], dtype)
            all_flat = list(flat)
            for item in obj[1:]:
                _, f = _NDArray._infer(item, dtype)
                all_flat.extend(f)
            return (len(obj),) + sub_shape, all_flat

        @property
        def ndim(self):
            return len(self.shape)

        @property
        def size(self):
            return len(self._flat)

        def copy(self):
            return _NDArray(list(self._flat), shape=self.shape, dtype=self.dtype)

        def astype(self, dtype):
            return _NDArray([dtype(x) for x in self._flat], shape=self.shape, dtype=dtype)

        def reshape(self, *dims):
            if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
                dims = tuple(dims[0])
            known = 1
            neg_idx = -1
            new_dims = list(dims)
            for i, d in enumerate(new_dims):
                if d == -1:
                    neg_idx = i
                else:
                    known *= d
            if neg_idx >= 0:
                new_dims[neg_idx] = max(0, self.size // max(1, known))
            return _NDArray(list(self._flat), shape=tuple(new_dims), dtype=self.dtype)

        def min(self):
            return min(self._flat) if self._flat else 0

        def max(self):
            return max(self._flat) if self._flat else 0

        def mean(self, axis=None):
            if axis == 2 and self.ndim == 3:
                h, w, c = self.shape
                out = []
                for idx in range(0, len(self._flat), c):
                    out.append(sum(self._flat[idx:idx + c]) / c)
                return _NDArray(out, shape=(h, w), dtype=float)
            if axis == 0 and self.ndim == 2:
                h, w = self.shape
                out = [sum(self._flat[r * w + c] for r in range(h)) / max(1, h) for c in range(w)]
                return _NDArray(out, shape=(w,), dtype=float)
            if axis == 1 and self.ndim == 2:
                h, w = self.shape
                out = [sum(self._flat[r * w:(r + 1) * w]) / max(1, w) for r in range(h)]
                return _NDArray(out, shape=(h,), dtype=float)
            return (sum(self._flat) / len(self._flat)) if self._flat else 0.0

        def sum(self):
            return sum(self._flat)

        def __len__(self):
            return self.shape[0] if self.shape else 0

        def __iter__(self):
            if self.ndim == 1:
                for x in self._flat:
                    yield x
            elif self.ndim >= 2:
                for i in range(self.shape[0]):
                    yield self[i]

        def __getitem__(self, key):
            if isinstance(key, _NDArray):
                out = [v for v, m in zip(self._flat, key._flat) if m]
                return _NDArray(out, shape=(len(out),), dtype=self.dtype)
            if not isinstance(key, tuple):
                key = (key,)
            while len(key) < self.ndim:
                key = key + (slice(None),)
            ranges = []
            out_shape = []
            for k, dim in zip(key, self.shape):
                if isinstance(k, slice):
                    idxs = list(range(*k.indices(dim)))
                    ranges.append(idxs)
                    out_shape.append(len(idxs))
                else:
                    idx = k if k >= 0 else dim + k
                    ranges.append([idx])
            strides = [1] * self.ndim
            for i in range(self.ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * self.shape[i + 1]

            out = []
            def _collect(depth, offset):
                if depth == self.ndim:
                    out.append(self._flat[offset])
                    return
                st = strides[depth]
                for idx in ranges[depth]:
                    _collect(depth + 1, offset + idx * st)

            _collect(0, 0)
            if not out_shape:
                return out[0]
            return _NDArray(out, shape=tuple(out_shape), dtype=self.dtype)

        def __setitem__(self, key, val):
            if not isinstance(key, tuple):
                key = (key,)
            while len(key) < self.ndim:
                key = key + (slice(None),)
            ranges = []
            for k, dim in zip(key, self.shape):
                if isinstance(k, slice):
                    ranges.append(list(range(*k.indices(dim))))
                else:
                    ranges.append([k if k >= 0 else dim + k])
            strides = [1] * self.ndim
            for i in range(self.ndim - 2, -1, -1):
                strides[i] = strides[i + 1] * self.shape[i + 1]

            if isinstance(val, _NDArray):
                v_flat = val._flat
            elif isinstance(val, (list, tuple)):
                v_flat = list(val)
            else:
                v_flat = [val]

            v_idx = [0]
            def _assign(depth, offset):
                if depth == self.ndim:
                    self._flat[offset] = self.dtype(v_flat[v_idx[0] % len(v_flat)])
                    v_idx[0] += 1
                    return
                st = strides[depth]
                for idx in ranges[depth]:
                    _assign(depth + 1, offset + idx * st)

            _assign(0, 0)

        def _cmp(self, other, op):
            ov = other._flat if isinstance(other, _NDArray) else None
            if ov is not None:
                res = [1 if op(a, b) else 0 for a, b in zip(self._flat, ov)]
            else:
                res = [1 if op(a, other) else 0 for a in self._flat]
            return _NDArray(res, shape=self.shape, dtype=int)

        def __gt__(self, other): return self._cmp(other, lambda a, b: a > b)
        def __lt__(self, other): return self._cmp(other, lambda a, b: a < b)
        def __ge__(self, other): return self._cmp(other, lambda a, b: a >= b)
        def __le__(self, other): return self._cmp(other, lambda a, b: a <= b)
        def __sub__(self, other):
            ov = other._flat if isinstance(other, _NDArray) else None
            res = [a - b for a, b in zip(self._flat, ov)] if ov is not None else [a - other for a in self._flat]
            return _NDArray(res, shape=self.shape, dtype=self.dtype)
        def __rsub__(self, other):
            res = [other - a for a in self._flat]
            return _NDArray(res, shape=self.shape, dtype=self.dtype)
        def __matmul__(self, other):
            if self.ndim == 2 and other.ndim == 1:
                r, c = self.shape
                out = [sum(self._flat[i * c + j] * other._flat[j] for j in range(c)) for i in range(r)]
                return _NDArray(out, shape=(r,), dtype=float)
            return other

    np_stub.ndarray = _NDArray
    np_stub.array = lambda obj, dtype=float: _NDArray(obj, dtype=dtype)
    np_stub.asarray = lambda obj, dtype=float: obj.astype(dtype) if isinstance(obj, _NDArray) else _NDArray(obj, dtype=dtype)
    np_stub.zeros = lambda shape, dtype=float: _NDArray(0, shape=(shape,) if isinstance(shape, int) else shape, dtype=dtype)
    np_stub.ones = lambda shape, dtype=float: _NDArray(1, shape=(shape,) if isinstance(shape, int) else shape, dtype=dtype)
    np_stub.full = lambda shape, val, dtype=float: _NDArray(
        (val[0] if isinstance(val, (tuple, list)) else val),
        shape=(shape,) if isinstance(shape, int) else shape,
        dtype=dtype,
    )
    np_stub.zeros_like = lambda arr: _NDArray(0, shape=arr.shape, dtype=arr.dtype)
    np_stub.mean = lambda vals, axis=None: vals.mean(axis=axis) if isinstance(vals, _NDArray) else ((sum(vals) / len(vals)) if len(vals) else 0.0)
    def _median(vals):
        s = sorted(float(x) for x in vals)
        n = len(s)
        if n == 0:
            return 0.0
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    np_stub.median = _median
    np_stub.deg2rad = math.radians
    np_stub.rad2deg = math.degrees
    np_stub.cos = math.cos
    np_stub.sin = math.sin
    np_stub.hypot = math.hypot
    np_stub.count_nonzero = lambda arr: sum(1 for x in (arr._flat if isinstance(arr, _NDArray) else arr) if x)
    def _where(cond, x=None, y=None):
        if x is not None and y is not None:
            out = [x if c else y for c in cond._flat]
            return _NDArray(out, shape=cond.shape, dtype=int)
        if cond.ndim == 2:
            h, w = cond.shape
            ys, xs = [], []
            for idx, v in enumerate(cond._flat):
                if v:
                    ys.append(idx // w)
                    xs.append(idx % w)
            return (_NDArray(ys, shape=(len(ys),), dtype=int), _NDArray(xs, shape=(len(xs),), dtype=int))
        idxs = [i for i, v in enumerate(cond._flat) if v]
        return (_NDArray(idxs, shape=(len(idxs),), dtype=int),)
    np_stub.where = _where
    np_stub.unique = lambda arr: _NDArray(sorted(set(arr._flat)), dtype=int)
    np_stub.diff = lambda arr: _NDArray([arr._flat[i + 1] - arr._flat[i] for i in range(len(arr._flat) - 1)], dtype=int)
    def _isin(arr, test_elements):
        s = set(test_elements._flat if isinstance(test_elements, _NDArray) else test_elements)
        return _NDArray([1 if v in s else 0 for v in arr._flat], shape=arr.shape, dtype=int)
    np_stub.isin = _isin
    sys.modules["numpy"] = np_stub
    import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.COLOR_BGR2GRAY = 6
    cv2_stub.ADAPTIVE_THRESH_MEAN_C = 0
    cv2_stub.ADAPTIVE_THRESH_GAUSSIAN_C = 1
    cv2_stub.THRESH_BINARY = 0
    cv2_stub.THRESH_BINARY_INV = 1
    cv2_stub.MORPH_RECT = 0
    cv2_stub.MORPH_OPEN = 2
    cv2_stub.MORPH_CLOSE = 3
    cv2_stub.RETR_EXTERNAL = 0
    cv2_stub.CHAIN_APPROX_SIMPLE = 2
    cv2_stub.INTER_LINEAR = 1
    cv2_stub.BORDER_REPLICATE = 1
    cv2_stub.CC_STAT_LEFT = 0
    cv2_stub.CC_STAT_TOP = 1
    cv2_stub.CC_STAT_WIDTH = 2
    cv2_stub.CC_STAT_HEIGHT = 3
    cv2_stub.CC_STAT_AREA = 4

    def _cvtColor(img, code):
        return img.mean(axis=2).astype(np.uint8) if img.ndim == 3 else img

    def _gaussianBlur(img, ksize, sigma):
        return img

    def _threshold(img, thresh, maxval, type_):
        if type_ == cv2_stub.THRESH_BINARY_INV:
            return thresh, np.where(img <= thresh, maxval, 0).astype(np.uint8)
        return thresh, np.where(img > thresh, maxval, 0).astype(np.uint8)

    def _adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C):
        mean_val = float(src.mean()) - C
        if thresholdType == cv2_stub.THRESH_BINARY_INV:
            return np.where(src < mean_val, maxValue, 0).astype(np.uint8)
        return np.where(src > mean_val, maxValue, 0).astype(np.uint8)

    def _getStructuringElement(shape, ksize):
        return np.ones((ksize[1], ksize[0]), dtype=np.uint8)

    def _morphologyEx(src, op, kernel):
        kh, kw = kernel.shape[:2]
        if op == cv2_stub.MORPH_OPEN and kw >= 20 and kh == 1:
            out = np.zeros_like(src)
            for y in range(src.shape[0]):
                if np.count_nonzero(src[y, :] > 0) >= kw:
                    out[y, :] = src[y, :]
            return out
        return src

    def _dilate(src, kernel):
        return src

    def _connectedComponentsWithStats(bw):
        ys, xs = np.where(bw > 0)
        if len(xs) == 0:
            return 1, np.zeros_like(bw), np.zeros((1, 5), dtype=np.int32), np.zeros((1, 2))
        unique_ys = sorted(set(ys))
        y_groups = []
        cur = []
        for y in unique_ys:
            if not cur or y - cur[-1] <= 2:
                cur.append(y)
            else:
                y_groups.append(cur)
                cur = [y]
        if cur:
            y_groups.append(cur)
        stats = [[0, 0, bw.shape[1], bw.shape[0], bw.size]]
        centroids = [[bw.shape[1] / 2.0, bw.shape[0] / 2.0]]
        for grp in y_groups:
            grp_set = set(grp)
            gx = [x for x, y in zip(xs, ys) if y in grp_set]
            gy = [y for y in ys if y in grp_set]
            x0, x1 = int(min(gx)), int(max(gx))
            y0, y1 = int(min(gy)), int(max(gy))
            stats.append([x0, y0, x1 - x0 + 1, y1 - y0 + 1, len(gx)])
            centroids.append([float(sum(gx) / len(gx)), float(sum(gy) / len(gy))])
        return len(stats), np.zeros_like(bw), np.array(stats, dtype=np.int32), np.array(centroids, dtype=np.float64)

    def _findContours(bw, mode, method):
        ys, xs = np.where(bw > 0)
        if len(xs) == 0:
            return [], None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        cnt = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)
        return [cnt], None

    def _contourArea(cnt):
        pts = cnt.reshape(-1, 2)
        return float((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()))

    def _minAreaRect(cnt):
        pts = cnt.reshape(-1, 2)
        x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
        y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
        return ((0.5 * (x0 + x1), 0.5 * (y0 + y1)), (x1 - x0, y1 - y0), 0.0)

    def _Canny(img, t1, t2):
        return np.zeros_like(img)

    def _HoughLinesP(edges, rho, theta, threshold, minLineLength=0, maxLineGap=0):
        return None

    def _getRotationMatrix2D(center, angle, scale):
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

    def _warpAffine(src, M, dsize, flags=0, borderMode=0):
        return src

    def _line(img, pt1, pt2, color, thickness=1):
        x0, y0 = pt1
        x1, y1 = pt2
        y_min, y_max = max(0, min(y0, y1) - thickness // 2), min(img.shape[0], max(y0, y1) + thickness // 2 + 1)
        x_min, x_max = max(0, min(x0, x1)), min(img.shape[1], max(x0, x1) + 1)
        c_val = color[0] if isinstance(color, (tuple, list)) else color
        img[y_min:y_max, x_min:x_max] = c_val
        return img

    def _rectangle(img, pt1, pt2, color, thickness=1):
        x0, y0 = max(0, int(pt1[0])), max(0, int(pt1[1]))
        x1, y1 = min(img.shape[1], int(pt2[0])), min(img.shape[0], int(pt2[1]))
        c_val = color[0] if isinstance(color, (tuple, list)) else color
        img[y0:y1, x0:x1] = c_val
        return img

    cv2_stub.cvtColor = _cvtColor
    cv2_stub.GaussianBlur = _gaussianBlur
    cv2_stub.threshold = _threshold
    cv2_stub.adaptiveThreshold = _adaptiveThreshold
    cv2_stub.getStructuringElement = _getStructuringElement
    cv2_stub.morphologyEx = _morphologyEx
    cv2_stub.dilate = _dilate
    cv2_stub.connectedComponentsWithStats = _connectedComponentsWithStats
    cv2_stub.findContours = _findContours
    cv2_stub.contourArea = _contourArea
    cv2_stub.minAreaRect = _minAreaRect
    cv2_stub.Canny = _Canny
    cv2_stub.HoughLinesP = _HoughLinesP
    cv2_stub.getRotationMatrix2D = _getRotationMatrix2D
    cv2_stub.warpAffine = _warpAffine
    cv2_stub.line = _line
    cv2_stub.rectangle = _rectangle
    sys.modules["cv2"] = cv2_stub
    import cv2

try:
    from loguru import logger  # noqa: F401
except ImportError:
    loguru_stub = types.ModuleType("loguru")
    class _Logger:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass
    loguru_stub.logger = _Logger()
    sys.modules["loguru"] = loguru_stub

try:
    import rapidfuzz  # noqa: F401
except ImportError:
    import difflib
    rf_stub = types.ModuleType("rapidfuzz")
    fuzz_stub = types.ModuleType("rapidfuzz.fuzz")
    proc_stub = types.ModuleType("rapidfuzz.process")

    def _ratio(a, b, **kw):
        return difflib.SequenceMatcher(None, str(a), str(b)).ratio() * 100.0

    def _extractOne(query, choices, scorer=_ratio, score_cutoff=0.0, **kw):
        best_c, best_s, best_i = None, -1.0, -1
        for idx, c in enumerate(choices):
            s = (scorer or _ratio)(query, c)
            if s > best_s:
                best_c, best_s, best_i = c, s, idx
        if best_c is not None and best_s >= score_cutoff:
            return (best_c, best_s, best_i)
        return None

    fuzz_stub.WRatio = _ratio
    fuzz_stub.ratio = _ratio
    fuzz_stub.partial_ratio = _ratio
    fuzz_stub.token_set_ratio = _ratio
    proc_stub.extractOne = _extractOne
    rf_stub.fuzz = fuzz_stub
    rf_stub.process = proc_stub
    sys.modules["rapidfuzz"] = rf_stub
    sys.modules["rapidfuzz.fuzz"] = fuzz_stub
    sys.modules["rapidfuzz.process"] = proc_stub

try:
    import shapely.geometry  # noqa: F401
except ImportError:
    shapely_mod = types.ModuleType("shapely")
    geom_mod = types.ModuleType("shapely.geometry")
    class _Poly:
        def __init__(self, pts):
            arr = np.asarray(pts, dtype=float).reshape(-1, 2)
            self.x0, self.y0 = float(arr[:, 0].min()), float(arr[:, 1].min())
            self.x1, self.y1 = float(arr[:, 0].max()), float(arr[:, 1].max())
            self.is_valid = True
            self.area = max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)
        def intersection(self, other):
            ix0, iy0 = max(self.x0, other.x0), max(self.y0, other.y0)
            ix1, iy1 = min(self.x1, other.x1), min(self.y1, other.y1)
            p = _Poly([[0, 0], [0, 0], [0, 0], [0, 0]])
            p.area = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            return p
        def union(self, other):
            p = _Poly([[0, 0], [0, 0], [0, 0], [0, 0]])
            p.area = self.area + other.area - self.intersection(other).area
            return p
        def contains(self, pt):
            return self.x0 <= pt.x <= self.x1 and self.y0 <= pt.y <= self.y1
    class _Point:
        def __init__(self, x, y=None):
            if y is None:
                x, y = x[0], x[1]
            self.x, self.y = float(x), float(y)
    geom_mod.Polygon = _Poly
    geom_mod.Point = _Point
    shapely_mod.geometry = geom_mod
    sys.modules["shapely"] = shapely_mod
    sys.modules["shapely.geometry"] = geom_mod

# 若有相鄰的純 Python site-packages（如 yaml 等），加入低優先級搜尋路徑
for _cand in (
    PROJECT_DIR.parent / "ONNX_on_nuc" / ".venv" / "Lib" / "site-packages",
    PROJECT_DIR.parent.parent.parent / "C-SZM_YOSHI_DISHUP-AI-main" / "ONNX_on_nuc" / ".venv" / "Lib" / "site-packages",
):
    if _cand.is_dir() and str(_cand) not in sys.path:
        sys.path.append(str(_cand))

from utils.schema import Detection
from logic.models import Stub, TrackedItem, Tray, TrayState
from logic.ocr_routing import backend_for, validate_routes
from logic.stub_cv_det import (
    build_aligned_qty_rect,
    compute_qty_column_x,
    enhance_stub_detections,
    estimate_document_roi,
    find_horizontal_lines,
    find_rightmost_ink,
    generate_stub_candidates,
)
from logic.stub_parser import StubParser, StubTicketConfig
from logic.stub_verifier import verify_stub
from logic.tray_state_machine import TrayStateMachine
from logic.single_order_tracker import SingleOrderTracker
from logic.readers.sticker_matcher import StickerMatcher
from logic.text_utils import FuzzyMatcher


class _DummyBus:
    def __init__(self):
        self.det_events = []
        self.sys_events = []

    def publish_det_status(self, payload):
        self.det_events.append(payload)

    def publish_system(self, payload):
        self.sys_events.append(payload)


class _DummyRecPred:
    def __init__(self, reply=("1", 0.95)):
        self.reply = reply
        self.calls = 0
        self.last_args = None

    def predict_quantity(self, frame_crop, cx, cy, h, w, is_flip):
        self.calls += 1
        self.last_args = (cx, cy, h, w, is_flip)
        return self.reply


class _DummyOrderParser:
    def __init__(self, rec_pred=None, result=None):
        self.rec_pred = rec_pred
        self.result = result or ([("牛丼並盛", 1), ("みそ汁", 1)], "325")

    def parse(self, frame_crop, dt_boxes, rec_res, is_flip):
        return self.result


def _make_box(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _make_xyxy8(x0, y0, x1, y1):
    return (float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1))


class TestStubIntegration(unittest.TestCase):
    def test_ocr_routing_supports_stub(self):
        routes = validate_routes({"ticket": "det", "stub": "det", "sticker": "det"})
        self.assertEqual(routes["small_order_ticket"], "det")
        self.assertEqual(backend_for(routes, "stub"), "det")

    def test_stub_cv_det_functions(self):
        # 1. 測試水平分隔線偵測
        img = np.full((300, 200, 3), 240, dtype=np.uint8)
        cv2.line(img, (10, 75), (190, 75), (10, 10, 10), 2)
        lines = find_horizontal_lines(img)
        self.assertTrue(any(abs(y - 75.0) <= 4.0 for y in lines))

        # 2. 測試右側漏檢墨跡偵測
        band = np.full((40, 200, 3), 240, dtype=np.uint8)
        cv2.rectangle(band, (172, 8), (176, 30), (15, 15, 15), -1)  # 模擬細長數字 '1'
        ink_rect = find_rightmost_ink(band, min_x_ratio=0.70)
        self.assertIsNotNone(ink_rect)
        ix0, iy0, ix1, iy1 = ink_rect
        self.assertTrue(165 <= ix0 <= 180 and ix1 > ix0)

        # 3. 測試垂直對齊補框座標
        rect = build_aligned_qty_rect(100.0, 128.0, 174.0, 200, 300)
        self.assertIsNotNone(rect)
        rx0, ry0, rx1, ry1 = rect
        self.assertTrue(rx0 < 174 < rx1 and ry0 < 114 < ry1)

        # 4. 測試 MMR 白紙邊界緊貼與候選推算
        canvas = np.full((400, 400, 3), 40, dtype=np.uint8)
        cv2.rectangle(canvas, (100, 60), (300, 260), (235, 235, 235), -1)
        cx, cy, rw, rh, _ = estimate_document_roi(canvas)
        self.assertTrue(150 <= cx <= 250 and 120 <= cy <= 200)
        self.assertTrue(160 <= rw <= 240)

        ticket_det = Detection(
            xyxy=_make_xyxy8(100, 100, 260, 380),
            xywhr=(180.0, 240.0, 160.0, 280.0, 0.0),
            cls_id=4,
            cls_name="order_ticket",
            conf=0.9,
        )
        enhanced = enhance_stub_detections(
            canvas, [ticket_det], enable_candidate_fallback=True
        )
        self.assertEqual(len(enhanced), 2)
        self.assertEqual(enhanced[1].cls_name, "small_order_ticket")

    def test_stub_parser_and_yaml_config(self):
        yaml_path = PROJECT_DIR / "tasks" / "ocr" / "yoshinoya" / "stub_ticket.yaml"
        cfg = StubTicketConfig.from_yaml(str(yaml_path))
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.pos_confusion.get("9996"), "5996")
        self.assertEqual(cfg.qty_corrections.get("21"), 1)

        menus = ["牛丼並盛", "みそ汁", "半熟玉子"]
        parser = StubParser(menus=menus, cfg=cfg)
        rec_pred = _DummyRecPred(reply=("21", 0.92))  # 模擬細長 '1' 被辨識成 '21'

        img = np.full((400, 240, 3), 245, dtype=np.uint8)
        cv2.line(img, (10, 80), (230, 80), (10, 10, 10), 2)

        dt_boxes = [
            _make_box(15, 30, 95, 58),     # 表頭時間 12.34
            _make_box(140, 28, 215, 60),   # 表頭單號 325
            _make_box(15, 110, 145, 138),  # 第 1 列：無 POS 碼但有菜名「牛丼並盛」（應反查出 1102），且右側漏框數量欄（應觸發垂直對齊補框橘框）
            _make_box(15, 160, 145, 188),  # 第 2 列：易混淆 POS 9996 + みそ汁（應修正為 5996）
            _make_box(195, 160, 222, 188), # 第 2 列右側數量框：文字讀成 "21"（應自動校正為 1）
        ]
        rec_res = [
            ("12.34", 0.95),
            ("325", 0.98),
            ("牛丼並盛", 0.96),
            ("9996 みそ汁", 0.94),
            ("21", 0.88),
        ]

        parsed = parser.parse(dt_boxes, rec_res, img=img, rec_pred=rec_pred)
        self.assertTrue(parsed.ok)
        self.assertEqual(parsed.time_str, "12:34")
        self.assertEqual(parsed.ticket_no, "325")
        self.assertEqual(len(parsed.items), 2)

        # 第 1 列：牛丼並盛 -> 反查 POS 1102，垂直對齊補框重讀 "21" -> 校正為 1，且產生 fallback_rects
        self.assertEqual(parsed.items[0].name, "牛丼並盛")
        self.assertEqual(parsed.items[0].pos_code, "1102")
        self.assertEqual(parsed.items[0].quantity, 1)
        self.assertEqual(parsed.items[0].qty_source, "aligned_col")
        self.assertEqual(len(parsed.fallback_rects), 1)

        # 第 2 列：9996 -> 修正為 5996，"21" -> 校正為 1
        self.assertEqual(parsed.items[1].name, "みそ汁")
        self.assertEqual(parsed.items[1].pos_code, "5996")
        self.assertEqual(parsed.items[1].quantity, 1)

        order_json = parsed.to_order_json()
        self.assertEqual(order_json["order_no"], 325)
        self.assertEqual(order_json["quantity"], 2)
        self.assertEqual(order_json["total"], 2)
        self.assertEqual(order_json["item"]["1"]["pos_no"], 1102)
        self.assertEqual(order_json["item"]["2"]["pos_no"], 5996)

    def test_state_machine_stub_dispatch_and_verify(self):
        bus = _DummyBus()
        yaml_path = PROJECT_DIR / "tasks" / "ocr" / "yoshinoya" / "stub_ticket.yaml"
        cfg = StubTicketConfig.from_yaml(str(yaml_path))
        menus = ["牛丼並盛", "みそ汁"]
        stub_parser = StubParser(menus=menus, cfg=cfg)
        rec_pred = _DummyRecPred(reply=("1", 0.95))
        order_parser = _DummyOrderParser(
            rec_pred=rec_pred,
            result=([("牛丼並盛", 1), ("みそ汁", 1)], "325"),
        )
        matcher = StickerMatcher(
            menus_sticker=menus,
            fuzzy_matcher=FuzzyMatcher(["ねぎ"], 2, 35, 40),
        )
        sm = TrayStateMachine(
            bus=bus,
            order_parser=order_parser,
            sticker_matcher=matcher,
            menu_variants={"牛丼並盛": [["牛丼並盛"]], "みそ汁": [["みそ汁"]]},
            n_settle_frame=2,
            stub_parser=stub_parser,
        )

        tracker = SingleOrderTracker(
            bus=bus,
            sticker_missing_frame=15,
            tray_missing_frame=15,
            frame_width=1280,
            frame_height=960,
            ticket_leave_frame=15,
        )

        stub_det = Detection(
            xyxy=_make_xyxy8(100, 100, 300, 400),
            xywhr=(200.0, 250.0, 200.0, 300.0, 0.0),
            cls_id=9,
            cls_name="small_order_ticket",
            conf=0.92,
        )

        # 在 WAITING_TICKET（尚未放主票）時，連續更新讓 stub 穩定滿 2 幀
        for _ in range(3):
            tracker.update_ticket_and_stickers([], [], [], stub_dets=[stub_det])

        tasks = sm.generate_tasks(tracker.trays)
        stub_tasks = [t for t in tasks if t["type"] == "stub"]
        self.assertEqual(len(stub_tasks), 1)

        task = stub_tasks[0]
        sm.set_ocr_busy(tracker.trays, task["tray_id"], "stub", task["bbox"], target=task["target"])
        tray = tracker.trays[task["tray_id"]]
        self.assertTrue(tray.stub.item.is_ocr_busy)
        self.assertEqual(tray.state, TrayState.WAITING_TICKET)

        # 套用 stub OCR 結果
        dt_boxes = [
            _make_box(140, 28, 215, 60),
            _make_box(15, 110, 145, 138),
            _make_box(195, 110, 222, 138),
            _make_box(15, 160, 145, 188),
            _make_box(195, 160, 222, 188),
        ]
        rec_res = [
            ("325", 0.98),
            ("1102 牛丼並盛", 0.96),
            ("1", 0.95),
            ("5996 みそ汁", 0.94),
            ("1", 0.95),
        ]
        img = np.full((300, 240, 3), 245, dtype=np.uint8)
        sm.apply_ocr_result(
            tracker.trays, task["tray_id"], "stub", img, dt_boxes, rec_res, False, task["bbox"], target=task["target"]
        )
        self.assertIsNotNone(tray.stub.result)
        self.assertIsNotNone(tray.stub.order_json)
        self.assertIsNone(tray.stub.verification)  # 主票尚未放入

        # 放入主票並完成主票 OCR，應自動觸發與已解析 stub 的交叉驗證
        ticket_det = Detection(
            xyxy=_make_xyxy8(400, 100, 700, 600),
            xywhr=(550.0, 350.0, 300.0, 500.0, 0.0),
            cls_id=4,
            cls_name="order_ticket",
            conf=0.95,
        )
        for _ in range(3):
            tracker.update_ticket_and_stickers([ticket_det], [], [], stub_dets=[stub_det])

        t_tasks = [t for t in sm.generate_tasks(tracker.trays) if t["type"] == "ticket"]
        self.assertEqual(len(t_tasks), 1)
        sm.set_ocr_busy(tracker.trays, t_tasks[0]["tray_id"], "ticket", t_tasks[0]["bbox"])
        sm.apply_ocr_result(
            tracker.trays, t_tasks[0]["tray_id"], "ticket", img, [], [], False, t_tasks[0]["bbox"]
        )

        self.assertEqual(tray.state, TrayState.WAITING_STICKERS)
        self.assertIsNotNone(tray.stub.verification)
        self.assertTrue(tray.stub.verification.matched)

    def test_real_board_fig1_and_fig2_cases(self):
        yaml_path = PROJECT_DIR / "tasks" / "ocr" / "yoshinoya" / "stub_ticket.yaml"
        cfg = StubTicketConfig.from_yaml(str(yaml_path))
        menus = ["牛丼並盛", "みそ汁", "黒カレー", "チーズ"]
        parser = StubParser(menus=menus, cfg=cfg)

        # --- 圖一：無菜單品項預約聯（778 為上列桌號，397 與 16:50 位於同列，80 為中間黑底標籤誤讀）---
        img1 = np.full((347, 314, 3), 245, dtype=np.uint8)
        cv2.line(img1, (20, 120), (285, 120), (10, 10, 10), 2)
        dt_boxes_1 = [
            _make_box(35, 38, 105, 62),    # 純番号 (上列左)
            _make_box(170, 44, 215, 65),   # キー品 (上列中)
            _make_box(235, 46, 282, 68),   # 778 (上列右，微傾因此 cx 甚至比 397 稍右)
            _make_box(32, 68, 112, 92),    # 16:50 (下列左)
            _make_box(172, 72, 210, 95),   # 80 (下列中，●伝票No● 黑底誤讀)
            _make_box(230, 75, 276, 98),   # 397 (下列右，與 16:50 同列)
            _make_box(22, 155, 215, 185),  # 麦皿特肉:102217:00
            _make_box(18, 190, 208, 220),  # 菜き号:09091333646
        ]
        rec_res_1 = [
            ("純番号", 0.7006),
            ("キー品", 0.6499),
            ("778", 1.0),
            ("16:50", 0.9990),
            ("80", 0.7854),
            ("397", 1.0),
            ("麦皿特肉:102217:00", 0.8024),
            ("菜き号:09091333646", 0.8492),
        ]
        parsed1 = parser.parse(dt_boxes_1, rec_res_1, img=img1)
        self.assertTrue(parsed1.ok)
        self.assertEqual(parsed1.time_str, "16:50")
        self.assertEqual(parsed1.table_no, "778")
        self.assertEqual(parsed1.ticket_no, "397")
        self.assertEqual(len(parsed1.items), 0)
        json1 = parsed1.to_order_json()
        self.assertEqual(json1["order_no"], 397)
        self.assertEqual(json1["quantity"], 0)
        self.assertEqual(json1["total"], 0)
        self.assertEqual(json1["item"], {})

        # --- 圖二：水平線右尾端對齊 + StandaloneRecDLA.predict_quantity 6 參數呼叫 + '961TO 黒カレー' / '9948 TO チーズ' ---
        # 特意將圖片寬度設很大 (400)，但第一條水平線右尾端在 x=210，驗證對齊線是依水平線尾端而非圖片寬度 400*0.87=348
        img2 = np.full((368, 400, 3), 245, dtype=np.uint8)
        cv2.line(img2, (25, 140), (210, 140), (10, 10, 10), 2)
        dt_boxes_2 = [
            _make_box(35, 38, 98, 62),     # 枚7番号 (上列左，內含誤讀數字 7 應被忽略)
            _make_box(170, 44, 202, 66),   # 06 (上列右)
            _make_box(30, 68, 105, 92),    # 12:12 (下列左)
            _make_box(165, 72, 206, 96),   # 381 (下列右，與 12:12 同列)
            _make_box(28, 170, 145, 198),  # 961TO 黒カレー
            _make_box(24, 215, 138, 243),  # 9948 TO チーズ
        ]
        rec_res_2 = [
            ("枚7番号", 0.6156),
            ("06", 1.0),
            ("12:12", 0.9996),
            ("381", 1.0),
            ("961TO 黒カレー", 0.9876),
            ("9948 TO チーズ", 0.9647),
        ]
        rec_pred_2 = _DummyRecPred(reply=("1", 0.96))
        parsed2 = parser.parse(dt_boxes_2, rec_res_2, img=img2, rec_pred=rec_pred_2, is_flip=False)
        self.assertTrue(parsed2.ok)
        self.assertEqual(parsed2.time_str, "12:12")
        self.assertEqual(parsed2.table_no, "06")
        self.assertEqual(parsed2.ticket_no, "381")
        self.assertEqual(len(parsed2.items), 2)
        self.assertEqual(parsed2.items[0].name, "黒カレー")
        self.assertEqual(parsed2.items[0].pos_code, "9671")
        self.assertEqual(parsed2.items[0].quantity, 1)
        self.assertEqual(parsed2.items[1].name, "チーズ")
        self.assertEqual(parsed2.items[1].pos_code, "9748")
        self.assertEqual(parsed2.items[1].quantity, 1)
        # 驗證每列只呼叫 1 次 predict_quantity（共 2 次），且對齊點是依水平線尾端 210 推算（非 400*0.87=348）
        self.assertEqual(rec_pred_2.calls, 2)
        self.assertEqual(len(parsed2.fallback_rects), 2)
        rect_x1 = parsed2.fallback_rects[0][2]
        self.assertTrue(205 <= rect_x1 <= 215)


if __name__ == "__main__":
    unittest.main()
