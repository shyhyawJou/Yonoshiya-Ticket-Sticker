"""
debug_sheet.py
==============
訂單除錯總覽圖：一筆訂單收尾時，把這筆訂單看過的

    * 傳票（主票 / 候選票）：裁切圖 + OCR det 框 + rec 文字 + 逐列的模糊比對與
      menus_mapping 對應（傳票名稱 -> 要核對的貼紙）
    * 貼紙：裁切圖 + det 框 + rec 文字 + 模糊比對結果 / 候選 + 對應的傳票品項 + OK/NG 判定
    * 醬料包等「類別即身份」的物件：裁切圖 + 類別 -> object_class_items 對應 + OK/NG 判定

全部排進**一張** canvas_w x canvas_h（預設 4024 x 3036）的圖，只看這張圖就知道
這筆訂單每個環節的結果。

資料流
------
    TrayStateMachine（OCR 結果 / 醬料包判定）  --record_*-->  DebugSheet（依 tray_id 收集）
    LogicEngine._close_order                  --finalize-->  背景 thread 渲染 + 寫檔
    App reset / 切模式                         --discard-->   丟掉這筆訂單收集到的東西

設計原則
--------
* **不能影響主流程**：所有 record_* / finalize 內部都包 try/except，出錯只記 warning；
  渲染與寫檔在背景 thread，主迴圈只付出「縮圖 + 複製資料」的成本。
* **不改判定邏輯**：這裡只讀、只畫，完全不會回頭修改 tray / 貼紙的狀態。
* **文字用 Pillow + NotoSansJP**（cv2.putText 畫不出日文）。沒裝 Pillow 或找不到字型時退回
  ASCII 版本（日文會變 '?'），圖還是會出，只是菜名看不到。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

try:  # Pillow 是選用依賴：沒有的話退回 ASCII
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:  # pragma: no cover - 依環境而定
    _HAS_PIL = False

SHEET_VERSION = "1"
DEFAULT_FONT_REL = "tasks/ocr/NotoSansJP-Bold.otf"

# ---------------------------------------------------------------------------
# 顏色（RGB）。畫 cv2 時用 _bgr() 轉。
# ---------------------------------------------------------------------------
C_BG = (22, 24, 29)
C_PANEL = (34, 37, 44)
C_PANEL_HI = (44, 48, 57)
C_TEXT = (236, 238, 242)
C_DIM = (152, 158, 170)
C_OK = (46, 190, 84)
C_NG = (232, 56, 56)
C_PEND = (255, 152, 0)
C_UNK = (230, 196, 20)
C_NEUTRAL = (110, 116, 128)
C_INFO = (96, 170, 255)

# 傳票逐列配色（同一列的 det 框 / 表格標記同色）
ROW_PALETTE = [
    (255, 99, 99), (99, 190, 255), (120, 220, 120), (255, 190, 70),
    (200, 130, 255), (80, 225, 215), (255, 130, 200), (190, 215, 80),
]

VERDICT_STYLE: Dict[str, Tuple[Tuple[int, int, int], str]] = {
    "OK": (C_OK, "OK"),
    "NG": (C_NG, "NG"),
    "SUSPECT": (C_PEND, "CHECKING"),
    "UNKNOWN": (C_UNK, "UNKNOWN"),
    "UNREAD": (C_NEUTRAL, "UNREAD"),
    "UNTRACKED": (C_NEUTRAL, "UNTRACKED"),
}

STATUS_COLOR = {"OK": C_OK, "NG": C_NG, "NGtoOK": C_PEND}


def _bgr(c: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (int(c[2]), int(c[1]), int(c[0]))


def conf_color(conf: float) -> Tuple[int, int, int]:
    """rec 信心值 -> 框顏色（RGB）。"""
    if conf >= 0.9:
        return (60, 205, 100)
    if conf >= 0.7:
        return (190, 205, 50)
    if conf >= 0.5:
        return (255, 152, 0)
    return (235, 60, 60)


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------
@dataclass
class DebugContext:
    """渲染時查 mapping 用的唯讀資料（由 LogicEngine 建構時提供）。"""
    # 傳票名稱 -> 貼紙組合變體（menus_mapping.yaml 的 name -> sub_items）
    menu_variants: Dict[str, List[List[str]]] = field(default_factory=dict)
    # 偵測類別 -> 品項名稱（menus_mapping.yaml 的 object_class_items）
    object_class_items: Dict[str, str] = field(default_factory=dict)
    profile: str = ""

    def sticker_to_tickets(self) -> Dict[str, List[str]]:
        """貼紙品項 -> 哪些傳票名稱會展開出它（反查 mapping）。"""
        rev: Dict[str, List[str]] = {}
        for ticket, variants in self.menu_variants.items():
            for combo in variants:
                for sub in combo:
                    lst = rev.setdefault(sub, [])
                    if ticket not in lst:
                        lst.append(ticket)
        return rev


@dataclass
class Evidence:
    """一次 OCR（或一次類別直讀）留下的證據。"""
    uid: int
    kind: str                    # "ticket" | "ticket_extra" | "sticker" | "known"
    t: float                     # time.time()
    crop: Optional[np.ndarray]   # BGR，記錄時已縮到 crop_max_side 以內；None = 被 trim 掉了
    boxes: List[dict] = field(default_factory=list)   # [{"poly": [[x,y]*4], "text": str, "conf": float}]
    track: Optional[int] = None  # 貼紙 / 醬料包的追蹤 id，同一個實體的多次 OCR 共用
    data: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 文字層：Pillow 畫日文；沒有 Pillow 退回 cv2.putText（ASCII）
# ---------------------------------------------------------------------------
class _TextLayer:
    def __init__(self, canvas: np.ndarray, font_path: Optional[str]):
        self.canvas = canvas
        self._items: List[tuple] = []
        self._fonts: Dict[int, Any] = {}
        self._cwidths: Dict[Tuple[int, str], float] = {}
        self.font_path = font_path if (font_path and os.path.isfile(font_path)) else None
        self.has_jp = bool(_HAS_PIL and self.font_path)
        if not self.has_jp:
            logger.warning(
                "[debug_sheet] 沒有可用的 Pillow / 日文字型"
                f"（PIL={_HAS_PIL}, font={font_path}），文字退回 ASCII，日文會顯示成 '?'"
            )

    # -- 量測 -------------------------------------------------------------
    def _font(self, size: int):
        f = self._fonts.get(size)
        if f is None:
            f = ImageFont.truetype(self.font_path, size)
            self._fonts[size] = f
        return f

    def _cw(self, size: int, ch: str) -> float:
        """單一字元寬度（快取）。換行 / 截斷要逐字量寬，不快取的話同一個字會被量上萬次
        （實測 20 張貼紙的圖，getlength 佔渲染時間的 60%）。忽略字間距微調（kerning），
        對日文全形字沒有差別，英數字只差 1~2 像素。"""
        key = (size, ch)
        w = self._cwidths.get(key)
        if w is None:
            w = float(self._font(size).getlength(ch)) if self.has_jp else size * 0.55
            self._cwidths[key] = w
        return w

    def width(self, s: str, size: int) -> int:
        return int(sum(self._cw(size, ch) for ch in s))

    def line_h(self, size: int) -> int:
        return int(size * 1.32)

    def wrap(self, s: str, size: int, max_w: int, max_lines: int = 99) -> List[str]:
        """逐字換行（CJK 友善）。超過 max_lines 的部分以 … 截斷。逐字累加寬度，O(n)。"""
        out: List[str] = []
        for para in str(s).split("\n"):
            cur, cur_w = "", 0.0
            for ch in para:
                cw = self._cw(size, ch)
                if cur and cur_w + cw > max_w:
                    out.append(cur)
                    cur, cur_w = ch, cw
                else:
                    cur += ch
                    cur_w += cw
            out.append(cur)
        if len(out) > max_lines:
            out = out[:max_lines]
            out[-1] = out[-1][: max(0, len(out[-1]) - 1)] + "…"
        return out

    def clip(self, s: str, size: int, max_w: int) -> str:
        if self.width(s, size) <= max_w:
            return s
        ell = self._cw(size, "…")
        cur_w, keep = 0.0, 0
        for ch in s:
            cw = self._cw(size, ch)
            if cur_w + cw + ell > max_w:
                break
            cur_w += cw
            keep += 1
        return s[:keep] + "…"

    # -- 繪製 -------------------------------------------------------------
    def add(self, x: int, y: int, s: str, size: int, color=C_TEXT,
            bg: Optional[Tuple[int, int, int]] = None, bg_alpha: float = 0.7, pad: int = 3) -> int:
        """記錄一段文字，回傳這段文字佔用的高度。bg 有值時先在 canvas 上混一塊底色。"""
        if not s:
            return 0
        w, h = self.width(s, size), self.line_h(size)
        if bg is not None:
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(self.canvas.shape[1], x + w + pad)
            y1 = min(self.canvas.shape[0], y + h + pad)
            if x1 > x0 and y1 > y0:
                roi = self.canvas[y0:y1, x0:x1]
                tint = np.empty_like(roi)
                tint[:] = _bgr(bg)
                cv2.addWeighted(tint, bg_alpha, roi, 1.0 - bg_alpha, 0, dst=roi)
        self._items.append((int(x), int(y), s, int(size), color))
        return h

    def flush(self) -> np.ndarray:
        if not self._items:
            return self.canvas
        if self.has_jp:
            img = Image.fromarray(cv2.cvtColor(self.canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img)
            for x, y, s, size, color in self._items:
                draw.text((x, y), s, font=self._font(size), fill=tuple(color))
            self.canvas = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        else:
            for x, y, s, size, color in self._items:
                ascii_s = s.encode("ascii", "replace").decode("ascii")
                cv2.putText(self.canvas, ascii_s, (x, y + int(size * 0.95)),
                            cv2.FONT_HERSHEY_SIMPLEX, size / 36.0, _bgr(color), 1, cv2.LINE_AA)
        self._items = []
        return self.canvas


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _fit_into(img: np.ndarray, max_w: int, max_h: int, max_up: float = 2.5) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    sc = min(max_w / max(w, 1), max_h / max(h, 1), max_up)
    nw, nh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    interp = cv2.INTER_AREA if sc < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp), sc


def _fmt_counts(items: Optional[Sequence[dict]]) -> str:
    """[{名稱: 數量}, ...] -> '名稱×數量, ...'；空 -> '-'"""
    if not items:
        return "-"
    parts = []
    for d in items:
        for k, v in d.items():
            parts.append(f"{k}×{v}" if v is not None else f"{k}×?")
    return ", ".join(parts) if parts else "-"


def _hms(t: Optional[float]) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "--:--:--"


def _draw_boxes(tile: np.ndarray, boxes: List[dict], scale: float, tl: _TextLayer,
                origin: Tuple[int, int], colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                label: bool = True, min_fs: int = 13, max_fs: int = 26) -> None:
    """在已縮放貼上的 crop 上畫 det 框；label=True 時標 `序號 文字 信心`。

    tile 是整張 canvas（座標用 origin 偏移），文字走 TextLayer（日文）。
    """
    ox, oy = origin
    for i, b in enumerate(boxes):
        poly = np.asarray(b.get("poly", []), dtype=np.float32)
        if poly.shape != (4, 2):
            continue
        pts = (poly * scale).astype(np.int32) + np.array([ox, oy], dtype=np.int32)
        color = (colors or {}).get(i) or conf_color(float(b.get("conf", 0.0)))
        thick = 2 if scale < 1.2 else 3
        cv2.polylines(tile, [pts.reshape(-1, 1, 2)], True, _bgr(color), thick, cv2.LINE_AA)
        if label:
            bh = float(poly[:, 1].max() - poly[:, 1].min()) * scale
            fs = int(np.clip(bh * 0.55, min_fs, max_fs))
            txt = f"{i}:{b.get('text', '')} {float(b.get('conf', 0.0)):.2f}"
            lx = int(pts[:, 0].min())
            ly = int(pts[:, 1].min()) - tl.line_h(fs) - 1
            ly = max(oy, ly)
            tl.add(lx, ly, txt, fs, color=(255, 255, 255), bg=color, bg_alpha=0.55, pad=1)


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
class SheetRenderer:
    """把一筆訂單的 Evidence 排成一張總覽圖。純函式風格：不碰任何執行期狀態。"""

    HEADER_H = 330
    FOOTER_H = 64
    MARGIN = 24

    def __init__(self, ctx: DebugContext, font_path: Optional[str], canvas_w: int = 4024, canvas_h: int = 3036):
        self.ctx = ctx
        self.font_path = font_path
        self.W, self.H = int(canvas_w), int(canvas_h)
        self._rev = ctx.sticker_to_tickets()

    # ------------------------------------------------------------------
    def render(self, entries: List[Evidence], summary: Dict[str, Any]) -> np.ndarray:
        canvas = np.empty((self.H, self.W, 3), np.uint8)
        canvas[:] = _bgr(C_BG)
        tl = _TextLayer(canvas, self.font_path)

        tickets = [e for e in entries if e.kind in ("ticket", "ticket_extra")]
        items = [e for e in entries if e.kind in ("sticker", "known")]

        body_y0 = self.HEADER_H
        body_y1 = self.H - self.FOOTER_H
        self._draw_header(canvas, tl, summary, len(tickets), len(items))

        m = self.MARGIN
        if tickets and items:
            split = int(self.W * 0.44)
        elif tickets:
            split = self.W
        else:
            split = 0
        if tickets:
            self._draw_ticket_panel(canvas, tl, tickets, (m, body_y0, split - m // 2, body_y1), summary)
        else:
            self._empty_panel(canvas, tl, (m, body_y0, (split if split else self.W // 3) - m // 2, body_y1),
                              "沒有讀到傳票 (no ticket OCR)") if not items else None
        if items:
            x0 = split + m // 2 if tickets else m
            self._draw_item_panel(canvas, tl, items, (x0, body_y0, self.W - m, body_y1), summary)
        elif tickets:
            pass
        if not tickets and not items:
            self._empty_panel(canvas, tl, (m, body_y0, self.W - m, body_y1),
                              "這筆訂單沒有任何 OCR / 物件判定紀錄")

        self._draw_footer(canvas, tl, summary)
        return tl.flush()

    # ------------------------------------------------------------------
    # header / footer
    # ------------------------------------------------------------------
    def _panel(self, canvas, rect, color=C_PANEL):
        x0, y0, x1, y1 = rect
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _bgr(color), -1)

    def _empty_panel(self, canvas, tl, rect, msg):
        self._panel(canvas, rect)
        x0, y0, x1, y1 = rect
        tl.add(x0 + 30, y0 + 30, msg, 40, C_DIM)

    def _draw_header(self, canvas, tl: _TextLayer, s: Dict[str, Any], n_tickets: int, n_items: int):
        m = self.MARGIN
        cv2.rectangle(canvas, (0, 0), (self.W, self.HEADER_H - 8), _bgr(C_PANEL), -1)
        status = str(s.get("status", "?"))
        color = STATUS_COLOR.get(status, C_NEUTRAL)
        # 大色塊：最後結果
        bw, bh = 520, self.HEADER_H - 8 - 2 * m
        cv2.rectangle(canvas, (m, m), (m + bw, m + bh), _bgr(color), -1)
        fs = 120 if len(status) <= 3 else 96
        tl.add(m + (bw - tl.width(status, fs)) // 2, m + 18, status, fs, (255, 255, 255))
        tl.add(m + 24, m + bh - 92, "TIMEOUT 強制收尾" if s.get("timed_out") else "final_status", 36, (255, 255, 255))
        tl.add(m + 24, m + bh - 48, f"tray={s.get('tray_id', '?')}", 32, (255, 255, 255))

        x = m + bw + 40
        w_avail = self.W - x - m - 520
        y = m - 4
        t0, t1 = s.get("start_time"), s.get("end_time")
        title = (f"單號 {s.get('order_number') or '(未讀到)'}    "
                 f"{t0 or '?'}  →  {t1 or '?'}"
                 f"{'    (' + str(s['duration_sec']) + ' 秒)' if s.get('duration_sec') is not None else ''}")
        y += tl.add(x, y, title, 58, C_TEXT) + 6

        lines = [
            ("應有 expected", _fmt_counts(s.get("expected")), C_TEXT),
            ("已核對 achieved", _fmt_counts(s.get("achieved")), C_OK),
            ("缺少 missing", _fmt_counts(s.get("missing")), C_NG if s.get("missing") else C_DIM),
            ("錯誤 wrong", _fmt_counts(s.get("wrong")), C_NG if s.get("wrong") else C_DIM),
        ]
        for label, value, vc in lines:
            tl.add(x, y, label, 34, C_DIM)
            wrapped = tl.wrap(value, 38, w_avail - 330, max_lines=1)
            tl.add(x + 330, y - 2, wrapped[0], 38, vc)
            y += 52

        # 右側：影片 / 統計
        rx = self.W - m - 500
        ry = m - 4
        tl.add(rx, ry, f"傳票 OCR ×{n_tickets}   貼紙/物件 ×{n_items}", 34, C_INFO)
        ry += 50
        vf = s.get("video_file") or ""
        tl.add(rx, ry, "video", 30, C_DIM)
        for ln in tl.wrap(vf or "-", 28, 500 - 90, max_lines=3):
            tl.add(rx + 90, ry, ln, 28, C_TEXT)
            ry += 36
        ry += 8
        tl.add(rx, ry, f"profile: {self.ctx.profile or '-'}", 30, C_DIM)
        ry += 40
        tl.add(rx, ry, f"sheet v{SHEET_VERSION}  {time.strftime('%Y-%m-%d %H:%M:%S')}", 28, C_DIM)

    def _draw_footer(self, canvas, tl: _TextLayer, s: Dict[str, Any]):
        y = self.H - self.FOOTER_H + 8
        x = self.MARGIN
        cv2.rectangle(canvas, (0, self.H - self.FOOTER_H), (self.W, self.H), _bgr(C_PANEL), -1)
        tl.add(x, y, "det 框顏色 = rec 信心：", 30, C_DIM)
        x += tl.width("det 框顏色 = rec 信心：", 30) + 10
        for name, c in (("≥0.9", conf_color(0.95)), ("≥0.7", conf_color(0.8)),
                        ("≥0.5", conf_color(0.6)), ("<0.5", conf_color(0.2))):
            cv2.rectangle(canvas, (x, y + 6), (x + 30, y + 30), _bgr(c), -1)
            tl.add(x + 38, y, name, 30, C_TEXT)
            x += 38 + tl.width(name, 30) + 28
        x += 30
        tl.add(x, y, "貼紙邊框：", 30, C_DIM)
        x += tl.width("貼紙邊框：", 30) + 10
        for key in ("OK", "NG", "SUSPECT", "UNKNOWN", "UNREAD"):
            c, label = VERDICT_STYLE[key]
            cv2.rectangle(canvas, (x, y + 6), (x + 30, y + 30), _bgr(c), -1)
            tl.add(x + 38, y, label, 30, C_TEXT)
            x += 38 + tl.width(label, 30) + 28
        note = "傳票逐列同色 = 該列的 det 框；「近似候選」是除錯時用 token_set_ratio 重算的參考，不是正式比對結果"
        tl.add(self.W - self.MARGIN - tl.width(note, 26), y + 2, note, 26, C_DIM)

    # ------------------------------------------------------------------
    # 傳票
    # ------------------------------------------------------------------
    def _draw_ticket_panel(self, canvas, tl, tickets: List[Evidence], rect, summary):
        x0, y0, x1, y1 = rect
        shown = tickets[-3:] if len(tickets) > 3 else tickets      # 最多 3 張，保留最新
        hidden = len(tickets) - len(shown)
        gap = 16
        tile_h = (y1 - y0 - gap * (len(shown) - 1)) // len(shown)
        for i, ev in enumerate(shown):
            ty0 = y0 + i * (tile_h + gap)
            self._draw_ticket_tile(canvas, tl, ev, (x0, ty0, x1, ty0 + tile_h), tickets.index(ev) + 1)
        if hidden > 0:
            tl.add(x0 + 10, y1 - 40, f"(另有 {hidden} 筆較舊的傳票 OCR 未顯示)", 28, C_PEND, bg=C_BG)

    def _draw_ticket_tile(self, canvas, tl: _TextLayer, ev: Evidence, rect, seq: int):
        x0, y0, x1, y1 = rect
        self._panel(canvas, rect)
        d = ev.data
        is_extra = ev.kind == "ticket_extra"
        title_h = 64
        outcome = d.get("outcome", "")
        ok = bool(d.get("accepted"))
        oc = C_OK if ok else C_NG
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + title_h), _bgr(C_PANEL_HI), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + 12, y0 + title_h), _bgr(oc), -1)
        head = f"#{seq} {'候選傳票' if is_extra else '傳票'}  {_hms(ev.t)}   單號={d.get('order_number') or '-'}   det×{len(ev.boxes)}"
        tl.add(x0 + 24, y0 + 10, head, 36, C_TEXT)

        pad = 14
        crop_w = int((x1 - x0) * 0.50) - 2 * pad
        crop_h = (y1 - y0) - title_h - 2 * pad
        ix, iy = x0 + pad, y0 + title_h + pad
        row_color: Dict[int, Tuple[int, int, int]] = {}
        rows = d.get("rows") or []
        for ri, r in enumerate(rows):
            for bi in r.get("box_idx") or []:
                row_color[int(bi)] = ROW_PALETTE[ri % len(ROW_PALETTE)]
        if ev.crop is not None:
            img, sc = _fit_into(ev.crop, crop_w, crop_h, max_up=1.6)
            h, w = img.shape[:2]
            canvas[iy:iy + h, ix:ix + w] = img
            cv2.rectangle(canvas, (ix - 2, iy - 2), (ix + w + 1, iy + h + 1), _bgr(oc), 2)
            _draw_boxes(canvas, ev.boxes, sc, tl, (ix, iy), colors=row_color or None)
        else:
            tl.add(ix, iy, "(裁切圖已釋放)", 30, C_DIM)

        # ---- 右側：解析表 ----
        # OrderRequirements 記下的「實際套用」mapping（含換湯等替換），格式「品名×數量 → 貼紙×數量（備註）」；
        # 依品名分組，同名多列依序消耗。沒有（舊資料 / 解析失敗）才退回用 menu_variants 重算。
        applied: Dict[str, List[str]] = {}
        for ln in d.get("mapping_lines") or []:
            applied.setdefault(ln.split("×", 1)[0], []).append(ln)
        tx = x0 + int((x1 - x0) * 0.50)
        tw = x1 - tx - pad - 6
        ty = y0 + title_h + pad
        ty += tl.add(tx, ty, "OCR 文字 → 模糊比對 → mapping（要核對的貼紙）", 34, C_INFO) + 6
        if not rows:
            tl.add(tx, ty, "(解析器沒有產生任何列)", 32, C_NG)
            ty += 44
        for ri, r in enumerate(rows):
            if ty > y1 - 150:
                tl.add(tx, ty, f"… 另有 {len(rows) - ri} 列放不下", 28, C_PEND)
                ty += 36
                break
            col = ROW_PALETTE[ri % len(ROW_PALETTE)]
            cv2.rectangle(canvas, (tx, ty + 6), (tx + 14, ty + 36), _bgr(col), -1)
            status = r.get("status", "ok")
            name = r.get("name")
            line1 = f"{ri + 1}. 「{r.get('raw', '')}」"
            ty += tl.add(tx + 24, ty, tl.clip(line1, 34, tw - 24), 34, C_TEXT if status == "ok" else C_DIM)
            if status == "ok":
                sc_txt = f"{float(r.get('score', 0)):.0f}分" if r.get("score") is not None else ""
                q = r.get("qty")
                qsrc = r.get("qty_src") or ""
                line2 = f"→ {name}   {sc_txt}   ×{q if q is not None else '?'}   [{qsrc}]"
                ty += tl.add(tx + 24, ty, tl.clip(line2, 32, tw - 24), 32, C_OK if q is not None else C_PEND)
                queue_ = applied.get(name)
                if queue_:
                    ln = queue_.pop(0)
                    mp = "mapping: " + (ln.split(" → ", 1)[1] if " → " in ln else ln)
                else:
                    mp = self._mapping_line(name)
                if mp:
                    for ln in tl.wrap(mp, 28, tw - 24, max_lines=2):
                        ty += tl.add(tx + 24, ty, ln, 28, C_INFO)
            else:
                why = {"no_match": "沒有比對到菜名", "low_score": f"分數過低 ({r.get('score', 0):.0f})",
                       "ghost_dropped": "幽靈列防護：丟棄", "qty_unconfirmed": "數量未確認",
                       "name_none": "沒有比對到菜名"}.get(status, status)
                extra = f" → {name}" if name else ""
                ty += tl.add(tx + 24, ty, tl.clip(f"× {why}{extra}", 30, tw - 24), 30, C_NG)
            ty += 8

        # ---- 下方：整張票的結論 ----
        by = y1 - 8
        notes: List[Tuple[str, Tuple[int, int, int]]] = []
        combos = d.get("combos") or []
        if combos:
            txt = " | ".join(", ".join(f"{k}×{v}" for k, v in c.items()) for c in combos[:4])
            notes.append((f"組合假設: {txt}" + (" …" if len(combos) > 4 else ""), C_INFO))
        subs = d.get("substitutions") or []
        if subs:
            notes.append((f"替換: {subs}", C_PEND))
        if d.get("unconfirmed_qty"):
            notes.append((f"數量未確認: {d['unconfirmed_qty']}", C_PEND))
        notes.append((f"結果: {outcome}", oc))
        lines: List[Tuple[str, Tuple[int, int, int]]] = []
        for text, c in notes:
            for ln in tl.wrap(text, 28, tw, max_lines=2):
                lines.append((ln, c))
        by -= len(lines) * tl.line_h(28)
        for ln, c in lines:
            by += tl.add(tx, by, ln, 28, c)

    def _mapping_line(self, ticket_name: Optional[str]) -> str:
        if not ticket_name:
            return ""
        variants = self.ctx.menu_variants.get(ticket_name)
        if not variants:
            return f"mapping: {ticket_name} (無套餐拆分，貼紙就是它自己)"
        primary = "、".join(variants[0])
        if len(variants) == 1:
            return f"mapping: {primary}"
        others = " | ".join("、".join(v) for v in variants[1:3])
        return f"mapping: {primary}   (另有 {len(variants) - 1} 種組合: {others}{' …' if len(variants) > 3 else ''})"

    # ------------------------------------------------------------------
    # 貼紙 / 醬料包
    # ------------------------------------------------------------------
    def _group_tracks(self, items: List[Evidence]) -> List[List[Evidence]]:
        groups: Dict[Any, List[Evidence]] = {}
        order: List[Any] = []
        for e in items:
            key = e.track if e.track is not None else ("u", e.uid)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(e)
        return [groups[k] for k in order]

    @staticmethod
    def _verdict_rank(v: str) -> int:
        return {"NG": 0, "SUSPECT": 1, "UNKNOWN": 2, "UNREAD": 3, "UNTRACKED": 4, "OK": 5}.get(v, 9)

    def _draw_item_panel(self, canvas, tl: _TextLayer, items: List[Evidence], rect, summary):
        x0, y0, x1, y1 = rect
        groups = self._group_tracks(items)
        # NG 排前面，同類保持時間順序
        groups.sort(key=lambda g: self._verdict_rank(g[-1].data.get("verdict", "")))
        cap = 24
        hidden = max(0, len(groups) - cap)
        groups = groups[:cap]
        n = len(groups)
        W, H = x1 - x0, y1 - y0
        gap = 14
        best = (1, n)
        best_score = -1.0
        for cols in range(1, n + 1):
            rows = -(-n // cols)
            cw, ch = (W - gap * (cols - 1)) / cols, (H - gap * (rows - 1)) / rows
            score = min(cw / 1.0, ch / 1.25)   # 偏好接近 4:5 的格子
            if score > best_score:
                best_score, best = score, (cols, rows)
        cols, rows = best
        cw = (W - gap * (cols - 1)) // cols
        ch = (H - gap * (rows - 1)) // rows
        for i, g in enumerate(groups):
            r, c = divmod(i, cols)
            cx0, cy0 = x0 + c * (cw + gap), y0 + r * (ch + gap)
            self._draw_item_tile(canvas, tl, g, (cx0, cy0, cx0 + cw, cy0 + ch), i + 1)
        if hidden:
            tl.add(x1 - 520, y1 - 40, f"(另有 {hidden} 個物件未顯示)", 28, C_PEND, bg=C_BG)

    def _draw_item_tile(self, canvas, tl: _TextLayer, group: List[Evidence], rect, idx: int):
        x0, y0, x1, y1 = rect
        last = group[-1]
        d = last.data
        verdict = d.get("verdict", "UNREAD")
        vcolor, vlabel = VERDICT_STYLE.get(verdict, (C_NEUTRAL, verdict))
        self._panel(canvas, rect)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), _bgr(vcolor), 6)
        w = x1 - x0
        title_h = 54
        cv2.rectangle(canvas, (x0, y0), (x1, y0 + title_h), _bgr(vcolor), -1)
        kind_txt = "醬料包/物件" if last.kind == "known" else "貼紙"
        tl.add(x0 + 14, y0 + 6, f"{vlabel}   #{idx} {kind_txt}  {_hms(last.t)}", 34, (255, 255, 255))
        if len(group) > 1:
            tag = f"OCR×{len(group)}"
            tl.add(x1 - 14 - tl.width(tag, 30), y0 + 9, tag, 30, (255, 255, 255))

        pad = 12
        crop_box_h = int((y1 - y0 - title_h) * 0.48)
        crop_src = next((e for e in reversed(group) if e.crop is not None), None)
        cy = y0 + title_h + pad
        if crop_src is not None:
            img, sc = _fit_into(crop_src.crop, w - 2 * pad - 8, crop_box_h, max_up=2.2)
            h, ww = img.shape[:2]
            ox = x0 + (w - ww) // 2
            canvas[cy:cy + h, ox:ox + ww] = img
            _draw_boxes(canvas, crop_src.boxes, sc, tl, (ox, cy), label=True, min_fs=14, max_fs=24)
            cy += h + pad
        else:
            cy += tl.add(x0 + pad, cy, "(無裁切圖)", 28, C_DIM) + pad

        # ---- 文字區 ----
        tw = w - 2 * pad - 6
        lines = self._item_lines(group)
        avail = y1 - cy - 6
        fs = 30
        while fs > 17:
            wrapped = [(ln, c, sz) for ln, c, sz in self._wrap_lines(tl, lines, fs, tw)]
            if sum(tl.line_h(sz) for _, _, sz in wrapped) <= avail:
                break
            fs -= 2
        wrapped = list(self._wrap_lines(tl, lines, fs, tw))
        for ln, c, sz in wrapped:
            if cy + tl.line_h(sz) > y1 - 4:
                break
            cy += tl.add(x0 + pad, cy, ln, sz, c)

    @staticmethod
    def _wrap_lines(tl: _TextLayer, lines, fs, tw):
        for text, color, rel in lines:
            sz = max(16, fs + rel)
            for ln in tl.wrap(text, sz, tw, max_lines=3):
                yield ln, color, sz

    def _item_lines(self, group: List[Evidence]) -> List[Tuple[str, Tuple[int, int, int], int]]:
        """回傳 [(文字, 顏色, 相對字級)]。"""
        last = group[-1]
        d = last.data
        out: List[Tuple[str, Tuple[int, int, int], int]] = []
        verdict = d.get("verdict", "")
        vcolor = VERDICT_STYLE.get(verdict, (C_NEUTRAL, ""))[0]
        if last.kind == "known":
            cls = d.get("cls_name")
            name = d.get("item_name")
            out.append((f"類別 {cls}", C_TEXT, 0))
            out.append((f"mapping(object_class_items) → {name}", C_INFO, 0))
        else:
            raw = d.get("raw")
            conf = d.get("conf")
            out.append((f"rec: 「{raw if raw else '(沒讀到字)'}」" + (f"  conf {conf:.2f}" if conf is not None else ""), C_TEXT, 0))
            matched = d.get("matched")
            if matched:
                exact = "，一字不差" if d.get("exact") else ""
                out.append((f"模糊比對 → {matched}  ({d.get('score', 0):.0f}分{exact})", C_OK if verdict == "OK" else C_PEND, 0))
            else:
                out.append(("模糊比對 → (無)", C_NG, 0))
            top = d.get("top") or []
            if top:
                out.append(("近似候選: " + " / ".join(f"{n} {s:.0f}" for n, s in top), C_DIM, -4))
            rep = d.get("reported")
            if rep and rep != matched:
                out.append((f"對外回報名稱 → {rep}（純異寫統一成主變體）", C_DIM, -4))

        # 反查 mapping：這個貼紙 / 物件品項是哪個傳票品項展開出來的
        item = d.get("reported") or d.get("matched") or d.get("item_name")
        if item:
            ticket_names = self._rev.get(item) or []
            mine = [t for t in ticket_names if t in (d.get("ticket_names") or [])]
            show = mine or ticket_names[:3]
            expected = d.get("expected")
            in_expected = expected is not None and item in expected
            what = "物件" if last.kind == "known" else "貼紙"
            if show:
                out.append((f"mapping: 傳票 {' / '.join(show)} → {what} {item}", C_INFO, 0))
            elif in_expected:
                out.append((f"mapping: {item} 在本訂單待核對清單內", C_INFO, 0))
            elif expected is not None:
                out.append((f"mapping: 「{item}」不在本訂單待核對清單（menus_mapping 往回對照也找不到對應的傳票品項）", C_NG, 0))

        note = d.get("note") or ""
        out.append((f"判定 {verdict}: {note}" if note else f"判定 {verdict}", vcolor, 0))
        exp = d.get("expected")
        if exp is not None:
            chk = d.get("checked") or []
            out.append((f"待核對 {len(exp)} 項 / 已核對 {len(chk)}", C_DIM, -4))
        for evt in (d.get("events") or [])[-3:]:
            out.append((f"事件 {evt}", C_PEND, -2))
        if len(group) > 1:
            hist = []
            for k, e in enumerate(group[-5:-1], 1):
                dd = e.data
                hist.append(f"{dd.get('raw') or dd.get('cls_name') or '?'}→{dd.get('matched') or dd.get('item_name') or '-'}[{dd.get('verdict', '?')}]")
            out.append(("前次: " + "  ".join(hist), C_DIM, -4))
        return out


# ---------------------------------------------------------------------------
# 收集 + 非同步輸出
# ---------------------------------------------------------------------------
class DebugSheet:
    """對外門面：收集證據、收尾時非同步渲染並存檔。"""

    def __init__(self, cfg, root_dir: str | Path, project_dir: str | Path,
                 context: Optional[DebugContext] = None,
                 crop_fn: Optional[Callable[[np.ndarray, Sequence[float]], np.ndarray]] = None):
        self.cfg = cfg
        self.enabled: bool = bool(getattr(cfg, "enabled", False))
        self.ctx = context or DebugContext()
        self.crop_fn = crop_fn
        base = Path(getattr(cfg, "root_dir", "") or root_dir)
        self.out_root = base
        self.folder = str(getattr(cfg, "folder", "debug_sheet") or "debug_sheet")
        fp = getattr(cfg, "font_path", "") or ""
        self.font_path = str(fp) if fp else str(Path(project_dir) / DEFAULT_FONT_REL)
        self.canvas_w = int(getattr(cfg, "canvas_w", 4024))
        self.canvas_h = int(getattr(cfg, "canvas_h", 3036))
        self.quality = int(getattr(cfg, "jpeg_quality", 85))
        self.save_json = bool(getattr(cfg, "save_json", True))
        self.mode = str(getattr(cfg, "mode", "all"))
        self.max_entries = int(getattr(cfg, "max_entries", 80))
        self.crop_max_side = int(getattr(cfg, "crop_max_side", 1600))
        self.slow_record_s = float(getattr(cfg, "slow_record_ms", 50.0)) / 1000.0
        self.slow_render_s = float(getattr(cfg, "slow_render_s", 5.0))

        self._lock = threading.Lock()
        self._evidence: Dict[str, List[Evidence]] = {}
        self._next_uid = 0
        self._next_track = 0
        self._frame: Optional[np.ndarray] = None

        self._queue: "queue.Queue[Optional[tuple]]" = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None
        self._renderer: Optional[SheetRenderer] = None
        self.dropped = 0
        # --- 耗時統計 ---
        # _cost：每筆訂單在「主迴圈」累積的成本（會直接拖慢辨識，最需要盯）。
        # _totals：整個程式生命週期的背景渲染統計。
        # last_timing：最近一張圖的完整耗時（測試與除錯用）。
        self._cost: Dict[str, dict] = {}
        self._totals = {"n": 0, "render_sum": 0.0, "render_max": 0.0, "total_sum": 0.0, "total_max": 0.0}
        self.last_timing: Dict[str, Any] = {}
        if self.enabled:
            logger.info(
                f"[debug_sheet] 啟用 mode={self.mode} canvas={self.canvas_w}x{self.canvas_h} "
                f"輸出={self.out_root}/<日期>/{self.folder}/ font={self.font_path}"
            )

    # ------------------------------------------------------------------
    # 收集
    # ------------------------------------------------------------------
    def add_cost(self, tray_id: str, seconds: float, label: str) -> None:
        """記一筆「主迴圈」花掉的時間。單次太慢會立刻警告。"""
        if not self.enabled:
            return
        with self._lock:
            c = self._cost.setdefault(tray_id, {"n": 0, "total": 0.0, "max": 0.0, "max_label": "", "by": {}})
            c["n"] += 1
            c["total"] += seconds
            c["by"][label] = c["by"].get(label, 0.0) + seconds
            if seconds > c["max"]:
                c["max"], c["max_label"] = seconds, label
        if seconds > self.slow_record_s:
            logger.warning(
                f"[debug_sheet][timing] 主迴圈單次記錄偏慢: {label} {seconds * 1000:.1f}ms "
                f"(門檻 {self.slow_record_s * 1000:.0f}ms) tray={tray_id}"
            )

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        """每幀 update() 開頭給一次當下的原圖參考（不複製），醬料包裁切用。"""
        self._frame = frame

    def track_uid(self, obj: Any) -> int:
        """幫追蹤物件（TrackedItem）配一個穩定 id。

        不能直接用 id(obj)：物件被回收後位址會被別的物件重用，兩個不同的貼紙會被誤併成同一個。
        改成第一次看到時配一個遞增編號，存在物件自己身上（TrackedItem 是一般 dataclass，可以設屬性）。
        物件不能設屬性時（__slots__ 之類）退回「每次都是新編號」：圖仍會出，只是同一個實體的
        多次 OCR 不會被併成同一格。
        """
        uid = getattr(obj, "_dbg_uid", None)
        if uid is None:
            with self._lock:
                self._next_track += 1
                uid = self._next_track
            try:
                setattr(obj, "_dbg_uid", uid)
            except Exception:
                pass
        return uid

    @staticmethod
    def _own(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """主迴圈只做複製（大約 1~7ms），**不縮圖**：INTER_AREA 縮一張傳票要 25~50ms，
        會直接卡掉辨識主迴圈一兩幀。縮圖留到背景 thread（_shrink_entries）。
        一定要複製：呼叫端的 frame_crop 之後可能被重用 / 改寫。"""
        if crop is None or crop.size == 0:
            return None
        return crop.copy()

    def _shrink_entries(self, entries: List[Evidence]) -> None:
        """背景 thread：把過大的裁切圖縮到 crop_max_side 以內（框座標同步縮放），省渲染時間與記憶體。"""
        for e in entries:
            if e.crop is None:
                continue
            h, w = e.crop.shape[:2]
            side = max(h, w)
            if side <= self.crop_max_side:
                continue
            sc = self.crop_max_side / side
            e.crop = cv2.resize(e.crop, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
            for b in e.boxes:
                b["poly"] = [[p[0] * sc, p[1] * sc] for p in b["poly"]]

    @staticmethod
    def _boxes(dt_boxes, rec_res) -> List[dict]:
        out: List[dict] = []
        if dt_boxes is None or rec_res is None:
            return out
        for box, rec in zip(dt_boxes, rec_res):
            try:
                poly = np.asarray(box, dtype=np.float32).reshape(-1, 2)[:4]
                if poly.shape != (4, 2):
                    continue
                text, conf = rec[0], float(rec[1])
            except Exception:
                continue
            out.append({"poly": poly.tolist(), "text": str(text), "conf": conf})
        return out

    def _add(self, tray_id: str, ev: Evidence) -> Evidence:
        with self._lock:
            lst = self._evidence.setdefault(tray_id, [])
            lst.append(ev)
            if ev.kind in ("ticket", "ticket_extra"):
                # 傳票解析失敗會一直重試，每次都是一張 8MB 的大圖；圖上最多只畫最新 3 張，
                # 所以只留最新 4 張的裁切圖，更舊的只留文字紀錄（json 還在）。
                with_crop = [e for e in lst if e.kind in ("ticket", "ticket_extra") and e.crop is not None]
                for old in with_crop[:-4]:
                    old.crop = None
            if len(lst) > self.max_entries:
                # 釋放最舊、而且不是該追蹤物件最新一筆的貼紙裁切圖，文字紀錄保留
                latest_of_track = {}
                for e in lst:
                    if e.track is not None:
                        latest_of_track[e.track] = e.uid
                for e in lst:
                    if (e.crop is not None and e.kind in ("sticker", "known")
                            and latest_of_track.get(e.track) != e.uid):
                        e.crop = None
                        break
        return ev

    def record_ocr(self, tray_id: str, kind: str, crop: Optional[np.ndarray], dt_boxes, rec_res,
                   data: Dict[str, Any], track: Optional[int] = None) -> Optional[Evidence]:
        if not self.enabled:
            return None
        try:
            boxes = self._boxes(dt_boxes, rec_res)
            small = self._own(crop)
            with self._lock:
                self._next_uid += 1
                uid = self._next_uid
            ev = Evidence(uid=uid, kind=kind, t=time.time(), crop=small, boxes=boxes, track=track, data=dict(data))
            return self._add(tray_id, ev)
        except Exception as e:  # 除錯功能不能影響主流程
            logger.warning(f"[debug_sheet] record_ocr 失敗（略過）: {e!r}")
            return None

    def record_known(self, tray_id: str, xywhr, data: Dict[str, Any], track: Optional[int] = None) -> Optional[Evidence]:
        """醬料包等不走 OCR 的物件：用當下的原圖依旋轉框裁一張。"""
        if not self.enabled:
            return None
        try:
            crop = None
            if self._frame is not None and self.crop_fn is not None and xywhr is not None:
                try:
                    crop = self.crop_fn(self._frame, xywhr)
                except Exception as e:
                    logger.debug(f"[debug_sheet] 醬料包裁切失敗: {e!r}")
            small = crop if (crop is not None and crop.size) else None   # crop_fn 回傳的本來就是新陣列
            with self._lock:
                self._next_uid += 1
                uid = self._next_uid
            ev = Evidence(uid=uid, kind="known", t=time.time(), crop=small, boxes=[], track=track, data=dict(data))
            return self._add(tray_id, ev)
        except Exception as e:
            logger.warning(f"[debug_sheet] record_known 失敗（略過）: {e!r}")
            return None

    def annotate(self, tray_id: str, track: Optional[int], text: str) -> None:
        """在某個貼紙 / 物件最新一筆證據上加一行事件（例如「錯誤品項已被移走」）。"""
        if not self.enabled or track is None:
            return
        try:
            with self._lock:
                for e in reversed(self._evidence.get(tray_id, [])):
                    if e.track == track:
                        e.data.setdefault("events", []).append(f"{_hms(time.time())} {text}")
                        return
        except Exception as e:
            logger.warning(f"[debug_sheet] annotate 失敗（略過）: {e!r}")

    def discard(self, tray_id: Optional[str] = None) -> None:
        """App reset / 切模式：這筆訂單作廢，不出圖。"""
        with self._lock:
            if tray_id is None:
                self._evidence.clear()
            else:
                self._evidence.pop(tray_id, None)

    # ------------------------------------------------------------------
    # 收尾 -> 非同步渲染
    # ------------------------------------------------------------------
    def should_render(self, summary: Dict[str, Any]) -> bool:
        if self.mode == "all":
            return True
        return bool(summary.get("status") != "OK" or summary.get("missing") or summary.get("wrong"))

    def finalize(self, tray_id: str, summary: Dict[str, Any]) -> Optional[str]:
        """取走這筆訂單的證據並排進渲染佇列。回傳預定的輸出路徑（模式不符 / 停用時 None）。"""
        if not self.enabled:
            return None
        t_fin = time.perf_counter()
        with self._lock:
            entries = self._evidence.pop(tray_id, [])
            cost = self._cost.pop(tray_id, None) or {"n": 0, "total": 0.0, "max": 0.0, "max_label": "", "by": {}}
        try:
            if not self.should_render(summary):
                return None
            day = time.strftime("%Y%m%d")
            stamp = time.strftime("%Y%m%d_%H%M%S")
            order = str(summary.get("order_number") or "NA").replace("/", "-")
            name = f"{stamp}_{tray_id}_No{order}_{summary.get('status', 'NA')}"
            path = self.out_root / day / self.folder / f"{name}.jpg"
            self._ensure_worker()
            cost["finalize"] = time.perf_counter() - t_fin     # 取走證據 + 組檔名，不含渲染（在背景）
            cost["queue_depth"] = self._queue.qsize()
            try:
                self._queue.put_nowait((entries, dict(summary), path, cost, time.perf_counter()))
            except queue.Full:
                self.dropped += 1
                logger.warning(f"[debug_sheet] 渲染佇列已滿，丟棄這張總覽圖（累積 {self.dropped}）: {name}")
                return None
            return str(path)
        except Exception as e:
            logger.warning(f"[debug_sheet] finalize 失敗（略過）: {e!r}")
            return None

    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="debug-sheet", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                entries, summary, path, cost, t_enq = job
                self._write_one(entries, summary, path, cost, t_enq)
            except Exception as e:
                logger.error(f"[debug_sheet] 渲染/寫檔失敗: {e!r}")
            finally:
                self._queue.task_done()

    def render_to_array(self, entries: List[Evidence], summary: Dict[str, Any]) -> np.ndarray:
        """同步渲染（測試與離線重畫用）。"""
        if self._renderer is None:
            self._renderer = SheetRenderer(self.ctx, self.font_path, self.canvas_w, self.canvas_h)
        return self._renderer.render(entries, summary)

    def _write_one(self, entries: List[Evidence], summary: Dict[str, Any], path: Path,
                   cost: Optional[dict] = None, t_enq: Optional[float] = None) -> None:
        cost = cost or {}
        t0 = time.perf_counter()
        queue_wait = (t0 - t_enq) if t_enq is not None else 0.0

        self._shrink_entries(entries)
        img = self.render_to_array(entries, summary)
        t1 = time.perf_counter()

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("cv2.imencode 失敗")
        t2 = time.perf_counter()

        # 先寫 .tmp 再改名：rclone / aibox_uploader 掃資料夾時不會撿到寫一半的檔案
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, path)
        t3 = time.perf_counter()

        if self.save_json:
            self._dump_json(entries, summary, path.with_suffix(".json"))
        t4 = time.perf_counter()

        render_s, encode_s, write_s, json_s = t1 - t0, t2 - t1, t3 - t2, t4 - t3
        total_s = t4 - t0
        main_total = float(cost.get("total", 0.0)) + float(cost.get("finalize", 0.0))
        self.last_timing = {
            "path": str(path), "bytes": len(buf), "entries": len(entries),
            "queue_wait_s": queue_wait, "render_s": render_s, "encode_s": encode_s,
            "write_s": write_s, "json_s": json_s, "bg_total_s": total_s,
            "main_n": int(cost.get("n", 0)), "main_total_s": main_total,
            "main_max_s": float(cost.get("max", 0.0)), "main_max_label": cost.get("max_label", ""),
            "main_finalize_s": float(cost.get("finalize", 0.0)),
            "main_by_label_s": dict(cost.get("by", {})),
            "queue_depth": int(cost.get("queue_depth", 0)),
        }
        tot = self._totals
        tot["n"] += 1
        tot["render_sum"] += render_s
        tot["render_max"] = max(tot["render_max"], render_s)
        tot["total_sum"] += total_s
        tot["total_max"] = max(tot["total_max"], total_s)

        by = " ".join(f"{k}={v * 1000:.1f}ms" for k, v in sorted(cost.get("by", {}).items(), key=lambda kv: -kv[1]))
        logger.info(
            f"[debug_sheet][timing] 單號={summary.get('order_number')} {summary.get('status')} {path.name} "
            f"{img.shape[1]}x{img.shape[0]} {len(buf) / 1e6:.2f}MB 證據{len(entries)}筆\n"
            f"    主迴圈(影響辨識): 累計 {main_total * 1000:.1f}ms / 記錄 {cost.get('n', 0)} 次"
            f"，單次最大 {float(cost.get('max', 0.0)) * 1000:.1f}ms({cost.get('max_label', '-')})"
            f"，finalize {float(cost.get('finalize', 0.0)) * 1000:.1f}ms"
            f"{('，分項 ' + by) if by else ''}\n"
            f"    背景(獨立 thread): 佇列等待 {queue_wait:.2f}s + 渲染 {render_s:.2f}s + 編碼 {encode_s:.2f}s"
            f" + 寫檔 {write_s:.2f}s + json {json_s:.2f}s = {total_s:.2f}s"
            f"（佇列深度 {cost.get('queue_depth', 0)}）\n"
            f"    累計 {tot['n']} 張：背景平均 {tot['total_sum'] / tot['n']:.2f}s / 最大 {tot['total_max']:.2f}s，"
            f"渲染平均 {tot['render_sum'] / tot['n']:.2f}s / 最大 {tot['render_max']:.2f}s"
        )
        if total_s > self.slow_render_s:
            logger.warning(
                f"[debug_sheet][timing] 背景出圖偏慢 {total_s:.1f}s（門檻 {self.slow_render_s:g}s）："
                f"渲染 {render_s:.1f}s 編碼 {encode_s:.1f}s 寫檔 {write_s:.1f}s。"
                f"持續偏慢請考慮 mode: ng_only 或調低 jpeg_quality"
            )

    @staticmethod
    def _dump_json(entries: List[Evidence], summary: Dict[str, Any], path: Path) -> None:
        doc = {
            "version": SHEET_VERSION,
            "summary": summary,
            "entries": [
                {
                    "uid": e.uid, "kind": e.kind, "time": _hms(e.t), "track": e.track,
                    "boxes": [{"text": b["text"], "conf": round(b["conf"], 4)} for b in e.boxes],
                    "data": e.data,
                }
                for e in entries
            ],
        }
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1, default=str)
        os.replace(tmp, path)

    def shutdown(self, timeout: float = 10.0) -> None:
        """把佇列裡還沒寫的圖寫完再結束（切模式 / 程式結束時呼叫）。"""
        t = self._thread
        if t is None or not t.is_alive():
            return
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            return
        t.join(timeout=timeout)
