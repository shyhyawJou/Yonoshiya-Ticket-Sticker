"""
readers/label_cv.py
===================
固定版面標籤的參考圖輔助（CV）：在貼紙 crop 上找參考圖的「模板區」，決定方向與尺度，
再依版面切出要讀的那幾列交給 rec。給 readers/date_label.py（沙拉日期標籤）用，
寫成不綁沙拉的形式，之後別的固定版面標籤可以共用（一類一個 reference_img/<類別>/ 資料夾）。

參考圖放在 label.yaml 同一個資料夾，版面不用手動標：載入時每張參考圖自己用水平投影切列，
fields 指定「第幾列」（-2 = 倒數第二列），欄位上方的所有列就是模板區（沙拉：サラダ +
保存方法 / 消費期限，每張都一樣）。新增參考圖只要把轉正的 crop 丟進資料夾。

CV 只當輔助（det 優先）：模板分數不到 min_score 就完全不參與。
2026-11-06 實驗機實測（7 張沙拉日期標籤、502 張一般貼紙）：沙拉 0.93~0.99、一般貼紙最高 0.46；
四個方向都判對；歪 10 度時開始讀錯（det 不受影響），所以不能讓 CV 取代 det。
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
_MIN_ROW_H = 6   # 參考圖上比這矮的列（雜點、框線殘段）不算一列
# 粗找（半解析度）的分數比全解析度略低，粗找門檻放寬這麼多，不讓真的標籤在粗找就被擋掉
_COARSE_SLACK = 0.15


@dataclass
class LabelCvConfig:
    enabled: bool = False
    refs: str = "*.jpg"                    # 參考圖（相對 label.yaml 的資料夾）
    min_score: float = 0.75                # 模板分數門檻（TM_CCOEFF_NORMED）
    scales: Tuple[float, float] = (0.80, 1.25)
    fields: Dict[str, int] = field(default_factory=dict)   # 欄位名 -> 第幾列（負數從下面數）

    @classmethod
    def from_dict(cls, raw: Optional[dict]) -> "LabelCvConfig":
        raw = raw or {}
        sc = raw.get("scales") or cls.scales
        return cls(enabled=bool(raw.get("enabled", False)), refs=str(raw.get("refs") or cls.refs),
                   min_score=float(raw.get("min_score", cls.min_score)),
                   scales=(float(sc[0]), float(sc[1])),
                   fields={str(k): int(v) for k, v in (raw.get("fields") or {}).items()})


@dataclass
class _Ref:
    name: str
    gray: np.ndarray                       # 拉平背景後的灰階
    anchor: Tuple[int, int, int, int]      # 模板區 (x0, y0, x1, y1)
    fields: Dict[str, Tuple[int, int, int, int]]   # 欄位列 (x0, y0, x1, y1)


@dataclass
class LabelCvResult:
    score: float
    rotate: int                            # crop 順時針轉幾度是正向
    ref: str
    texts: Dict[str, Tuple[str, float]]    # 欄位 -> (rec 原文, 信心)
    ms: float


def _flatten(gray: np.ndarray) -> np.ndarray:
    """除掉背景亮度（close 的核比筆畫粗，估出來的背景不含字），光線不均也不影響後面的門檻"""
    k = max(15, min(gray.shape) // 6) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.divide(gray, bg, scale=255)


def _ink(flat: np.ndarray) -> np.ndarray:
    """墨跡遮罩；碰到影像邊緣的連通塊（標籤外的背景、旁邊的紙）去掉"""
    _, bw = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    h, w = bw.shape
    edge = (st[:, 0] == 0) | (st[:, 1] == 0) | (st[:, 0] + st[:, 2] >= w) | (st[:, 1] + st[:, 3] >= h)
    edge[0] = False
    bw[edge[lab]] = 0
    return bw


def _rows(bw: np.ndarray, min_h: int = 4) -> List[Tuple[int, int, int, int]]:
    """水平投影切出文字列 [(x0, y0, x1, y1)]（影像要是正向）"""
    h, w = bw.shape
    on = (bw > 0).sum(axis=1) > max(2, 0.02 * w)
    rows, y = [], 0
    while y < h:
        if not on[y]:
            y += 1
            continue
        y0 = y
        while y < h and on[y]:
            y += 1
        if y - y0 >= min_h:
            cols = np.nonzero((bw[y0:y] > 0).any(axis=0))[0]
            rows.append((int(cols.min()), y0, int(cols.max()) + 1, y))
    return rows


class LabelTemplate:
    def __init__(self, cfg: LabelCvConfig, folder: str, rec_pred, tag: str = "CV"):
        self.cfg = cfg
        self.rec_pred = rec_pred
        self.tag = tag
        self.refs: List[_Ref] = []
        for p in sorted(glob.glob(os.path.join(folder, cfg.refs))):
            ref = self._load_ref(p)
            if ref is not None:
                self.refs.append(ref)
        lo, hi = cfg.scales
        self.coarse_scales = np.arange(lo, hi + 1e-6, 0.10)
        logger.info(f"[{tag}] 參考圖 {len(self.refs)} 張（{folder}），模板門檻 {cfg.min_score:g}，欄位 {cfg.fields}")

    @property
    def ready(self) -> bool:
        return bool(self.refs) and self.rec_pred is not None and bool(self.cfg.fields)

    def _load_ref(self, path: str) -> Optional[_Ref]:
        img = cv2.imread(path)
        if img is None:
            logger.warning(f"[{self.tag}] 參考圖讀不到：{path}")
            return None
        gray = _flatten(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        rows = [r for r in _rows(_ink(gray)) if r[3] - r[1] >= _MIN_ROW_H]
        try:
            fields = {k: rows[i] for k, i in self.cfg.fields.items()}
        except IndexError:
            logger.warning(f"[{self.tag}] 參考圖 {os.path.basename(path)} 只切出 {len(rows)} 列，不夠 fields 用，略過")
            return None
        first = min(r[1] for r in fields.values())
        top = [r for r in rows if r[3] <= first]
        if not top:
            logger.warning(f"[{self.tag}] 參考圖 {os.path.basename(path)} 欄位上方沒有模板區，略過")
            return None
        anchor = (min(r[0] for r in top), min(r[1] for r in top), max(r[2] for r in top), max(r[3] for r in top))
        return _Ref(os.path.basename(path), gray, anchor, fields)

    @staticmethod
    def _best(g: np.ndarray, ref: _Ref, scales) -> Tuple[float, Optional[Tuple[int, int]], float]:
        """ref 的模板區在 g 上各尺度的最佳 (分數, 左上角, 尺度)"""
        x0, y0, x1, y1 = ref.anchor
        tpl0 = ref.gray[y0:y1, x0:x1]
        best = (-1.0, None, 1.0)
        for s in scales:
            tpl = cv2.resize(tpl0, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if tpl.shape[0] < 4 or tpl.shape[1] < 4 or tpl.shape[0] >= g.shape[0] or tpl.shape[1] >= g.shape[1]:
                continue
            _, sc, _, loc = cv2.minMaxLoc(cv2.matchTemplate(g, tpl, cv2.TM_CCOEFF_NORMED))
            if sc > best[0]:
                best = (float(sc), loc, float(s))
        return best

    # ------------------------------------------------------------------
    def read(self, crop_bgr: np.ndarray) -> Optional[LabelCvResult]:
        """找模板、讀欄位；模板分數不到門檻（不是這種標籤，或看不清楚）回 None"""
        if not self.ready or crop_bgr is None or crop_bgr.size == 0:
            return None
        t0 = time.perf_counter()
        g0 = _flatten(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY))
        # 先粗後細：全部方向 x 參考圖 x 細尺度一次比完要 ~250 ms（實驗機），而貼紙結果是在主迴圈套用的。
        # 粗找：半解析度、4 個方向、0.1 一格的尺度，順便決定方向；一般貼紙多半在這裡就被擋掉
        half = cv2.resize(g0, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        coarse = (-1.0, 0, 1.0)
        for rot, code in _ROT.items():
            g = half if code is None else cv2.rotate(half, code)
            for ref in self.refs:
                sc, _, s = self._best(g, ref, self.coarse_scales * 0.5)
                if sc > coarse[0]:
                    coarse = (sc, rot, s / 0.5)
        if coarse[0] < self.cfg.min_score - _COARSE_SLACK:
            return None
        # 細找：只看粗找選出的方向，全解析度、所有參考圖、粗找尺度前後 0.1 內 0.05 一格
        _, rot, cs = coarse
        g = g0 if _ROT[rot] is None else cv2.rotate(g0, _ROT[rot])
        lo, hi = self.cfg.scales
        fine = np.arange(max(lo, cs - 0.1), min(hi, cs + 0.1) + 1e-6, 0.05)
        best = (-1.0, None, None, 1.0)
        for ref in self.refs:
            sc, loc, s = self._best(g, ref, fine)
            if sc > best[0]:
                best = (sc, ref, loc, s)
        score, ref, loc, s = best
        if ref is None or score < self.cfg.min_score:
            return None

        img = crop_bgr if _ROT[rot] is None else cv2.rotate(crop_bgr, _ROT[rot])
        bw = _ink(_flatten(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        H, W = bw.shape
        ax0, ay0 = ref.anchor[:2]
        texts = {}
        for key, (x0, y0, x1, y1) in ref.fields.items():
            # 參考圖座標 -> crop：以模板區左上角為原點、乘尺度
            tx0, ty0 = loc[0] + (x0 - ax0) * s, loc[1] + (y0 - ay0) * s
            tx1, ty1 = loc[0] + (x1 - ax0) * s, loc[1] + (y1 - ay0) * s
            # 投影微調：預期位置上下各放寬半列，取最高的那一列（印的位置每張差幾 px）
            hh = ty1 - ty0
            b0, b1 = int(max(0, ty0 - 0.5 * hh)), int(min(H, ty1 + 0.5 * hh))
            sub = _rows(bw[b0:b1, :])
            if sub:
                r = max(sub, key=lambda r: r[3] - r[1])
                tx0, ty0, tx1, ty1 = r[0], b0 + r[1], r[2], b0 + r[3]
            pad = 0.15 * (ty1 - ty0)
            q = np.array([[tx0 - pad, ty0 - pad], [tx1 + pad, ty0 - pad],
                          [tx1 + pad, ty1 + pad], [tx0 - pad, ty1 + pad]], np.float32)
            q[:, 0] = q[:, 0].clip(0, W - 1)
            q[:, 1] = q[:, 1].clip(0, H - 1)
            texts[key] = self.rec_pred.predict_text(img, q)
        return LabelCvResult(score=score, rotate=rot, ref=ref.name, texts=texts,
                             ms=1000 * (time.perf_counter() - t0))
