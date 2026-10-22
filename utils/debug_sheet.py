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

# 「有問題」的判定：NG（錯誤品項成立）、SUSPECT（疑似，確認中）、UNKNOWN（讀不出菜名）。
# 這些是除錯最需要的證據：圖上不能被併掉（見 SheetRenderer._group_tracks），記憶體吃緊時也不能被釋放。
PROBLEM_VERDICTS = ("NG", "SUSPECT", "UNKNOWN")


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


def _count_tokens(items: Optional[Sequence[dict]]) -> List[str]:
    """[{名稱: 數量}, ...] -> ['名稱×數量', ...]"""
    parts: List[str] = []
    for d in items or []:
        for k, v in d.items():
            parts.append(f"{k}×{v}" if v is not None else f"{k}×?")
    return parts


def _fmt_counts(items: Optional[Sequence[dict]]) -> str:
    """[{名稱: 數量}, ...] -> '名稱×數量, ...'；空 -> '-'"""
    return ", ".join(_count_tokens(items)) or "-"


def _wrap_tokens(tl: "_TextLayer", tokens: List[str], size: int, max_w: int, sep: str = ", ") -> List[str]:
    """依「項目」換行：不會把『牛丼小×2』從中間切開。單一項目比一行還長才退回逐字換行。"""
    lines: List[str] = []
    cur = ""
    for tok in tokens:
        cand = tok if not cur else cur + sep + tok
        if tl.width(cand, size) <= max_w:
            cur = cand
            continue
        if cur:
            lines.append(cur + sep.rstrip())
            cur = ""
        if tl.width(tok, size) > max_w:
            parts = tl.wrap(tok, size, max_w)
            lines.extend(parts[:-1])
            cur = parts[-1]
        else:
            cur = tok
    if cur:
        lines.append(cur)
    return lines or ["-"]


def _flow_tokens(tl: "_TextLayer", tokens: List[Tuple[str, Tuple[int, int, int]]], x: int, y: int, w: int,
                 y_max: int, sizes: Tuple[int, ...] = (24, 20, 17), draw: bool = True) -> Tuple[int, int, int]:
    """把 (文字, 顏色) 一個一個排進 w 寬的區域，放不下就換行。

    字級由大到小試，全部放得下就用該字級；都放不下就用最小字級盡量放。
    回傳 (放不下的個數, 用到的最底 y, 用的字級)。每個 token 自己是一段文字、不會被從中間換行。
    draw=False 只算排版不畫（先量一次，決定要不要留位置寫提示，再畫一次；不能畫兩遍）。
    """
    gap = 14
    tokens = [(tl.clip(t, sizes[0], w), c) for t, c in tokens]

    def layout(fs: int):
        pos, cx, cy, lh = [], 0, 0, tl.line_h(fs)
        for text, col in tokens:
            tw = tl.width(text, fs)
            if cx and cx + tw > w:
                cx, cy = 0, cy + lh
            pos.append((cx, cy, text, col))
            cx += tw + gap
        return pos, cy + lh

    fs, pos = sizes[-1], []
    for cand in sizes:
        pos, total = layout(cand)
        fs = cand
        if total <= y_max - y:
            break
    drawn, bottom = 0, y
    for cx, cy, text, col in pos:
        if y + cy + tl.line_h(fs) > y_max:
            break
        if draw:
            tl.add(x + cx, y + cy, text, fs, col)
        drawn += 1
        bottom = y + cy + tl.line_h(fs)
    return len(tokens) - drawn, bottom, fs


def _hms(t: Optional[float]) -> str:
    return time.strftime("%H:%M:%S", time.localtime(t)) if t else "--:--:--"


def _draw_boxes(tile: np.ndarray, boxes: List[dict], scale: float, tl: _TextLayer,
                origin: Tuple[int, int], colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
                label: bool = True, min_fs: int = 13, max_fs: int = 26, label_mode: str = "full") -> None:
    """在已縮放貼上的 crop 上畫 det 框。

    label_mode：
      "full"  = 標 `序號:文字 信心`（貼紙用，一張圖上只有一兩個框）。
      "index" = 只標序號（傳票用：傳票一張就有幾十個框，文字標在圖上會互相疊、擋住傳票本身的字；
                文字改在右側表格對照，見 SheetRenderer._draw_ticket_tile）。
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
            if label_mode == "index":
                fs = int(np.clip(bh * 0.50, 17, 26))   # 只有編號：夠大才讀得到，又不至於擋住傳票上的字
                txt = str(i)
            else:
                fs = int(np.clip(bh * 0.55, min_fs, max_fs))
                txt = f"{i}:{b.get('text', '')} {float(b.get('conf', 0.0)):.2f}"
            lx = int(pts[:, 0].min())
            ly = int(pts[:, 1].min()) - tl.line_h(fs) - 1
            ly = max(oy, ly)
            tl.add(lx, ly, txt, fs, color=(255, 255, 255), bg=color,
                   bg_alpha=0.92 if label_mode == "index" else 0.55, pad=1)


class _Seg(list):
    """一格的證據（時間序）。本身就是 list，額外記「同一個追蹤物件的前一段 / 後一段」。"""
    problem: bool = False
    prev: "Optional[_Seg]" = None
    next: "Optional[_Seg]" = None


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
class SheetRenderer:
    """把一筆訂單的 Evidence 排成一張總覽圖。純函式風格：不碰任何執行期狀態。"""

    HEADER_H = 330
    FOOTER_H = 64
    MARGIN = 24

    # 所有字級、間距、格子大小都是照這個寬度設計的。canvas_w / canvas_h 不是直接拿來排版，而是
    # 「先照 REF_W 排好（高度依長寬比），最後縮放到指定尺寸」：改小畫布只會讓字跟檔案一起變小，
    # 版面不會擠壞（以前字級固定、格子變窄，標題就會互相壓到）。
    REF_W = 4024

    def __init__(self, ctx: DebugContext, font_path: Optional[str], canvas_w: int = 4024, canvas_h: int = 3036,
                 tiles_per_page: int = 24, max_pages: int = 4):
        self.ctx = ctx
        self.font_path = font_path
        self.out_w, self.out_h = int(canvas_w), int(canvas_h)
        self.W = self.REF_W
        self.H = max(1, round(self.REF_W * self.out_h / self.out_w))
        self.tiles_per_page = max(1, int(tiles_per_page))
        self.max_pages = max(1, int(max_pages))
        self._rev = ctx.sticker_to_tickets()
        self._ticket_seq: Dict[int, int] = {}      # id(傳票證據) -> 第幾筆傳票（跨頁編號一致）

    # ------------------------------------------------------------------
    def plan(self, entries: List[Evidence]) -> Tuple[List[dict], dict]:
        """決定每一頁放什麼。塞不下的格子往後面的頁放，不丟掉。

        * 貼紙 / 醬料包：全部先排序（NG、疑似、Unknown … OK 最後），再依 tiles_per_page 切頁，
          所以有問題的永遠在最前面幾頁。
        * 傳票：第 1 頁放最新 3 張；更舊的、還留有裁切圖的依序放到後面的頁（每頁 3 張）。
          裁切圖已經被釋放的舊傳票沒有「圖」可放，只留在 .json。
        * 最多 max_pages 頁，超過的格子只留在 .json，最後一頁會註明數量。
        """
        tickets = [e for e in entries if e.kind in ("ticket", "ticket_extra")]
        items = [e for e in entries if e.kind in ("sticker", "known")]
        segs = self._group_tracks(items)
        segs.sort(key=lambda g: self._verdict_rank(g[-1].data.get("verdict", "")))

        cap = self.tiles_per_page
        first = tickets[-3:]
        rest = [e for e in tickets[:-3] if e.crop is not None]
        t_chunks: List[List[Evidence]] = []
        while rest:
            t_chunks.append(rest[-3:])
            rest = rest[:-3]

        n_pages = min(self.max_pages, max(1, -(-len(segs) // cap), 1 + len(t_chunks)))
        shown_segs = segs[:cap * n_pages]
        pages: List[dict] = []
        for p in range(n_pages):
            if p == 0:
                page_tickets = first
            else:
                page_tickets = t_chunks[p - 1] if p - 1 < len(t_chunks) else []
            pages.append({"tickets": page_tickets, "segs": shown_segs[p * cap:(p + 1) * cap], "offset": p * cap})
        shown_tickets = sum(len(pg["tickets"]) for pg in pages)
        info = {
            "n_tickets": len(tickets), "n_tiles": len(segs), "n_reads": len(items),
            "hidden_tickets": len(tickets) - shown_tickets,
            "hidden_tiles": len(segs) - len(shown_segs),
            "all_tickets": tickets,
        }
        return pages, info

    def iter_pages(self, entries: List[Evidence], summary: Dict[str, Any]):
        """逐頁渲染（generator）：寫完一頁就能釋放，不會同時握著好幾張 36MB 的畫布。

        一筆訂單渲染成 1 張或多張圖：第 1 張放傳票與最重要的格子，塞不下的往後放。
        """
        pages, info = self.plan(entries)
        self._ticket_seq = {id(e): i + 1 for i, e in enumerate(info["all_tickets"])}
        for i, pg in enumerate(pages):
            yield self._render_page(pg, summary, i + 1, len(pages), info)

    def render_pages(self, entries: List[Evidence], summary: Dict[str, Any]) -> List[np.ndarray]:
        return list(self.iter_pages(entries, summary))

    def render(self, entries: List[Evidence], summary: Dict[str, Any]) -> np.ndarray:
        """只取第 1 張（相容舊呼叫端 / 測試）。"""
        return self.render_pages(entries, summary)[0]

    def _render_page(self, pg: dict, summary: Dict[str, Any], page: int, pages: int, info: dict) -> np.ndarray:
        canvas = np.empty((self.H, self.W, 3), np.uint8)
        canvas[:] = _bgr(C_BG)
        tl = _TextLayer(canvas, self.font_path)

        tickets, segs = pg["tickets"], pg["segs"]
        body_y1 = self.H - self.FOOTER_H
        # 標頭高度依內容決定（項目多就換行加高），傳回高度當作內容區的起點
        body_y0 = self._draw_header(canvas, tl, summary, info["n_tickets"], info["n_tiles"], info["n_reads"],
                                    page, pages)

        m = self.MARGIN
        if tickets and segs:
            split = int(self.W * 0.44)
        elif tickets:
            split = self.W
        else:
            split = 0
        if tickets:
            # 只有第 1 頁提示「更舊的傳票沒有圖」（有圖的舊傳票已經排到後面幾頁）
            hidden_t = info["hidden_tickets"] if page == 1 else 0
            self._draw_ticket_panel(canvas, tl, tickets, (m, body_y0, split - m // 2, body_y1), summary, hidden_t)
        if segs:
            x0 = split + m // 2 if tickets else m
            note = None
            if page < pages:
                note = f"（續：第 {page + 1} 頁）"
            elif info["hidden_tiles"] > 0:
                note = (f"(另有 {info['hidden_tiles']} 個物件未顯示：已達 max_pages={self.max_pages}，"
                        f"完整紀錄見 .json)")
            self._draw_item_panel(canvas, tl, segs, (x0, body_y0, self.W - m, body_y1), summary,
                                  offset=pg["offset"], note=note)
        elif page == 1 and not tickets:
            self._empty_panel(canvas, tl, (m, body_y0, self.W - m, body_y1),
                              "這筆訂單沒有任何 OCR / 物件判定紀錄")

        self._draw_footer(canvas, tl, summary)
        img = tl.flush()
        if (self.out_w, self.out_h) != (self.W, self.H):
            img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return img

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

    def _draw_header(self, canvas, tl: _TextLayer, s: Dict[str, Any], n_tickets: int, n_tiles: int,
                     n_reads: int = 0, page: int = 1, pages: int = 1) -> int:
        """畫標頭並回傳標頭高度。項目一多就換行、把標頭加高，**不再截斷**成「…」。"""
        m = self.MARGIN
        bw = 520
        x = m + bw + 40
        right_w = 500
        rx = self.W - m - right_w
        w_val = rx - 20 - (x + 330)               # 值欄位：標籤欄右邊 ~ 右欄左邊
        VAL_FS, ROW_PITCH, SUB_PITCH = 38, 52, 46

        # ---- 左欄：應有 / 已核對 / 缺少 / 錯誤。依項目換行，最多 3 / 2 行，再多才用「…（共 N 項）」----
        rows = []
        for label, key, color, max_lines in (
            ("應有 expected", "expected", C_TEXT, 3),
            ("已核對 achieved", "achieved", C_OK, 2),
            ("缺少 missing", "missing", C_NG if s.get("missing") else C_DIM, 2),
            ("錯誤 wrong", "wrong", C_NG if s.get("wrong") else C_DIM, 2),
        ):
            toks = _count_tokens(s.get(key))
            lines = _wrap_tokens(tl, toks, VAL_FS, w_val)
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                suffix = f"…（共 {len(toks)} 項，見 .json）"
                last = lines[-1]
                while last and tl.width(last + suffix, VAL_FS) > w_val:
                    last = last[:-1]
                lines[-1] = last + suffix
            rows.append((label, lines, color))
        title_h = tl.line_h(58) + 6
        left_h = (m - 4) + title_h + sum(ROW_PITCH + SUB_PITCH * (len(ls) - 1) for _, ls, _ in rows)

        # ---- 右欄：頁碼 / 統計 / video / profile / 版本。每一項都依欄寬換行或 clip，不會被畫布邊緣切掉 ----
        right = []                                   # (文字, 字級, 顏色, x 偏移, 後面空多少)
        if pages > 1:
            right.append((f"第 {page} / {pages} 頁", 44, C_PEND, 0, 8))
        right.append((tl.clip(f"傳票 {n_tickets} 筆   貼紙/物件 {n_tiles} 格", 30, right_w), 30, C_INFO, 0, 0))
        if n_reads and n_reads != n_tiles:
            right.append((tl.clip(f"（OCR 讀取 {n_reads} 次，同一張重讀併成一格）", 24, right_w), 24, C_DIM, 0, 0))
        right.append(("", 0, C_DIM, 0, 8))
        video_lines = tl.wrap(s.get("video_file") or "-", 28, right_w - 90, max_lines=3)
        profile_lines = tl.wrap(f"profile: {self.ctx.profile or '-'}", 28, right_w, max_lines=3)
        sheet_line = f"sheet v{SHEET_VERSION}  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        right_h = (m - 4) + sum((tl.line_h(sz) if sz else 0) + gap for _, sz, _, _, gap in right) \
            + max(len(video_lines) * 36, 40) + 8 + len(profile_lines) * tl.line_h(28) + tl.line_h(28)

        hh = max(self.HEADER_H, int(max(left_h, right_h)) + 20)

        # ---------------- 繪製 ----------------
        cv2.rectangle(canvas, (0, 0), (self.W, hh - 8), _bgr(C_PANEL), -1)
        status = str(s.get("status", "?"))
        color = STATUS_COLOR.get(status, C_NEUTRAL)
        bh = hh - 8 - 2 * m
        cv2.rectangle(canvas, (m, m), (m + bw, m + bh), _bgr(color), -1)
        fs = 120 if len(status) <= 3 else 96
        tl.add(m + (bw - tl.width(status, fs)) // 2, m + 18, status, fs, (255, 255, 255))
        tl.add(m + 24, m + bh - 92, "TIMEOUT 強制收尾" if s.get("timed_out") else "final_status", 36, (255, 255, 255))
        tl.add(m + 24, m + bh - 48, f"tray={s.get('tray_id', '?')}", 32, (255, 255, 255))

        y = m - 4
        t0, t1 = s.get("start_time"), s.get("end_time")
        title = (f"單號 {s.get('order_number') or '(未讀到)'}    "
                 f"{t0 or '?'}  →  {t1 or '?'}"
                 f"{'    (' + str(s['duration_sec']) + ' 秒)' if s.get('duration_sec') is not None else ''}")
        y += tl.add(x, y, title, 58, C_TEXT) + 6
        for label, lines, vc in rows:
            tl.add(x, y, label, 34, C_DIM)
            for k, ln in enumerate(lines):
                tl.add(x + 330, y - 2 + k * SUB_PITCH, ln, VAL_FS, vc)
            y += ROW_PITCH + SUB_PITCH * (len(lines) - 1)

        ry = m - 4
        for text, sz, col, xo, gap in right:
            if sz:
                ry += tl.add(rx + xo, ry, text, sz, col)
            ry += gap
        tl.add(rx, ry, "video", 30, C_DIM)
        for ln in video_lines:
            tl.add(rx + 90, ry, ln, 28, C_TEXT)
            ry += 36
        ry += 8 + (0 if len(video_lines) else 4)
        for ln in profile_lines:
            ry += tl.add(rx, ry, ln, 28, C_DIM)
        tl.add(rx, ry, sheet_line, 28, C_DIM)
        return hh

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
    def _draw_ticket_panel(self, canvas, tl, tickets: List[Evidence], rect, summary, hidden: int = 0):
        x0, y0, x1, y1 = rect
        gap = 16
        tile_h = (y1 - y0 - gap * (len(tickets) - 1)) // len(tickets)
        for i, ev in enumerate(tickets):
            ty0 = y0 + i * (tile_h + gap)
            self._draw_ticket_tile(canvas, tl, ev, (x0, ty0, x1, ty0 + tile_h), self._ticket_seq.get(id(ev), i + 1))
        if hidden > 0:
            tl.add(x0 + 10, y1 - 40, f"(另有 {hidden} 筆較舊的傳票 OCR 沒有圖可顯示，見 .json)", 28, C_PEND, bg=C_BG)

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
        crop_bottom = iy
        if ev.crop is not None:
            img, sc = _fit_into(ev.crop, crop_w, crop_h, max_up=1.6)
            h, w = img.shape[:2]
            crop_bottom = iy + h
            canvas[iy:iy + h, ix:ix + w] = img
            cv2.rectangle(canvas, (ix - 2, iy - 2), (ix + w + 1, iy + h + 1), _bgr(oc), 2)
            _draw_boxes(canvas, ev.boxes, sc, tl, (ix, iy), colors=row_color or None, label_mode="index")
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
            bx = r.get("box_idx") or []
            box_lab = ("框 " + "、".join(str(b) for b in bx)) if bx else ""
            lab_w = tl.width(box_lab, 26) + 16 if box_lab else 0
            row_y = ty
            ty += tl.add(tx + 24, ty, tl.clip(line1, 34, tw - 24 - lab_w), 34, C_TEXT if status == "ok" else C_DIM)
            if box_lab:      # 圖上的框只有編號，這裡告訴你這一列用到哪幾個框
                tl.add(tx + tw - lab_w + 16, row_y + 6, box_lab, 26, C_DIM)
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

        # ---- 下方：整張票的結論（先算高度，上面的框對照要避開它）----
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

        # ---- 框編號對照：圖上的框只標編號，每個框的 rec 文字與信心列在傳票圖的下方（那裡本來就是空的）----
        # 全部的框都列（不只是沒歸到任何一列的）：歸了列的框，列裡只有合併後的文字，看不到各框自己的信心。
        # 文字顏色 = 圖上那個框的顏色（歸了列 = 該列顏色，沒歸列 = 依信心），一眼對得上。
        if ev.boxes:
            ly = crop_bottom + 14
            y_max = y1 - pad
            if y_max - ly > tl.line_h(24) + tl.line_h(17):
                ly += tl.add(ix, ly, "框編號對照（文字顏色 = 圖上框的顏色）", 24, C_DIM) + 4
                toks = []
                for i, bx in enumerate(ev.boxes):
                    txt = str(bx.get("text", ""))
                    txt = txt if len(txt) <= 14 else txt[:13] + "…"
                    toks.append((f"{i}:{txt}({float(bx.get('conf', 0.0)):.2f})",
                                 row_color.get(i) or conf_color(float(bx.get("conf", 0.0)))))
                if _flow_tokens(tl, toks, ix, ly, crop_w, y_max, draw=False)[0]:
                    # 放不下：讓出最後一行寫提示（其餘框的文字都在 .json）
                    left, bottom, _ = _flow_tokens(tl, toks, ix, ly, crop_w, y_max - tl.line_h(22))
                    tl.add(ix, bottom + 2, f"…另 {left} 個框見 .json", 22, C_PEND)
                else:
                    _flow_tokens(tl, toks, ix, ly, crop_w, y_max)

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
    def _group_tracks(self, items: List[Evidence]) -> List["_Seg"]:
        """把證據分組成「一格」。

        同一個追蹤物件（同一張貼紙實體）的多次 OCR 併成一格，**但在「有問題 / 正常」之間切換時要拆開**。

        為什麼要拆：NG 之後用 OK 貼紙把它蓋住，追蹤器會把新貼紙當成同一個物件重讀，判定從 NG 變 OK。
        若併成一格只顯示最後一筆，NG 那張貼紙的裁切圖與 det 框就從圖上消失了（只剩一行小字歷史），
        但這筆訂單其實有發生過錯誤（CSV 記為 NGtoOK）。拆開之後 NG 一格、OK 一格，互相註明。

        只在 problem <-> 正常 切換時拆：OK / 沒讀到字 / 對不到追蹤物件 之間的重讀仍然併成一格，
        同一張貼紙讀很多次不會把圖面塞滿。
        """
        by_track: Dict[Any, List[Evidence]] = {}
        order: List[Any] = []
        for e in items:
            key = e.track if e.track is not None else ("u", e.uid)
            if key not in by_track:
                by_track[key] = []
                order.append(key)
            by_track[key].append(e)

        out: List[_Seg] = []
        for key in order:
            segs: List[_Seg] = []
            for e in by_track[key]:
                is_problem = e.data.get("verdict") in PROBLEM_VERDICTS
                if segs and (segs[-1].problem == is_problem):
                    segs[-1].append(e)
                else:
                    seg = _Seg([e])
                    seg.problem = is_problem
                    if segs:                       # 同一個追蹤物件的前後兩段互相連結，圖上才能互相註明
                        seg.prev, segs[-1].next = segs[-1], seg
                    segs.append(seg)
            out.extend(segs)
        return out

    @staticmethod
    def _verdict_rank(v: str) -> int:
        return {"NG": 0, "SUSPECT": 1, "UNKNOWN": 2, "UNREAD": 3, "UNTRACKED": 4, "OK": 5}.get(v, 9)

    def _draw_item_panel(self, canvas, tl: _TextLayer, segs: List["_Seg"], rect, summary,
                         offset: int = 0, note: Optional[str] = None):
        """畫這一頁的貼紙 / 物件格。segs 已經分組、排序、切好頁；編號 = offset + 頁內順序，跨頁連續。"""
        x0, y0, x1, y1 = rect
        n = len(segs)
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
        for i, g in enumerate(segs):
            r, c = divmod(i, cols)
            cx0, cy0 = x0 + c * (cw + gap), y0 + r * (ch + gap)
            self._draw_item_tile(canvas, tl, g, (cx0, cy0, cx0 + cw, cy0 + ch), offset + i + 1)
        if note:
            tl.add(x1 - 10 - tl.width(note, 30), y1 - 44, note, 30, C_PEND, bg=C_BG)

    @staticmethod
    def _title_plan(tl: _TextLayer, vlabel: str, idx: int, kind_txt: str, hms_txt: str, tag: str,
                    w: int) -> Tuple[str, int, int]:
        """貼紙格的標題列：左邊「判定 #編號 種類 時間」、右邊「OCR×n」。

        格子窄（格數多 / 畫布小）或判定名稱長（UNREAD、CHECKING、UNKNOWN…）時，左右兩邊會撞在一起。
        依序嘗試：縮小字級 → 去掉時間 → 去掉種類，挑第一個「左 + 右 + 間隔」放得下的，保證不重疊。
        回傳 (左邊文字, 字級, 右邊標籤字級)。
        """
        avail = w - 28
        full = f"{vlabel}   #{idx} {kind_txt}  {hms_txt}"
        no_time = f"{vlabel}   #{idx} {kind_txt}"
        short = f"{vlabel} #{idx}"
        options = ([(full, fs) for fs in (34, 30, 26, 22)] + [(no_time, fs) for fs in (26, 22)]
                   + [(short, 22), (short, 18)])
        for text, fs in options:
            tag_fs = max(18, fs - 6)
            need = tl.width(text, fs) + ((16 + tl.width(tag, tag_fs)) if tag else 0)
            if need <= avail:
                return text, fs, tag_fs
        return short, 18, 18

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
        tag = f"OCR×{len(group)}" if len(group) > 1 else ""
        left_txt, tfs, tag_fs = self._title_plan(tl, vlabel, idx, kind_txt, _hms(last.t), tag, w)
        tl.add(x0 + 14, y0 + 6, left_txt, tfs, (255, 255, 255))
        if tag:
            tl.add(x1 - 14 - tl.width(tag, tag_fs), y0 + 6 + (tl.line_h(tfs) - tl.line_h(tag_fs)) // 2, tag,
                   tag_fs, (255, 255, 255))

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
        # 同一個追蹤物件前後判定不同（例如 NG 之後被 OK 貼紙蓋住）：兩格互相註明
        nxt = getattr(group, "next", None)
        prv = getattr(group, "prev", None)
        if nxt:
            nd = nxt[-1].data
            out.append((f"之後這個位置的追蹤物件改判為 {nd.get('verdict')}"
                        f"「{nd.get('raw') or nd.get('item_name') or '?'}」（{_hms(nxt[-1].t)}）："
                        f"原本的貼紙可能被蓋住或換掉", C_PEND, -2))
        if prv:
            pd = prv[-1].data
            out.append((f"這個位置先前判為 {pd.get('verdict')}「{pd.get('raw') or pd.get('item_name') or '?'}」"
                        f"（{_hms(prv[-1].t)}），之後被讀成這張", C_PEND, -2))
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
        self.tiles_per_page = max(1, int(getattr(cfg, "tiles_per_page", 24)))
        self.max_pages = max(1, int(getattr(cfg, "max_pages", 4)))   # 上限 99 在載入設定時檢查（config.py）
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
                f"每頁 {self.tiles_per_page} 格、最多 {self.max_pages} 頁 "
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
                # 釋放最舊、而且不是該追蹤物件最新一筆的貼紙裁切圖，文字紀錄保留。
                # 「有問題」的判定（NG / 疑似 / Unknown）不釋放：NG 之後常被同一個位置的 OK 讀取蓋過，
                # 它就不再是「最新一筆」，不保護的話長訂單裡最先被釋放的就是它。
                latest_of_track = {}
                for e in lst:
                    if e.track is not None:
                        latest_of_track[e.track] = e.uid
                for e in lst:
                    if (e.crop is not None and e.kind in ("sticker", "known")
                            and latest_of_track.get(e.track) != e.uid
                            and e.data.get("verdict") not in PROBLEM_VERDICTS):
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
            return str(self.page_path(path, 1))
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

    def _get_renderer(self) -> SheetRenderer:
        if self._renderer is None:
            self._renderer = SheetRenderer(self.ctx, self.font_path, self.canvas_w, self.canvas_h,
                                           self.tiles_per_page, self.max_pages)
        return self._renderer

    def render_pages(self, entries: List[Evidence], summary: Dict[str, Any]) -> List[np.ndarray]:
        """同步渲染所有頁（測試與離線重畫用）。塞不下的格子會依序放到後面的頁。"""
        return self._get_renderer().render_pages(entries, summary)

    def render_to_array(self, entries: List[Evidence], summary: Dict[str, Any]) -> np.ndarray:
        """只取第 1 頁（相容舊呼叫端）。"""
        return self.render_pages(entries, summary)[0]

    @staticmethod
    def page_path(path: Path, page: int) -> Path:
        """每一頁都帶固定 2 碼頁碼，格式一致：<名稱>_p01.jpg、<名稱>_p02.jpg…（單頁訂單也是 _p01）。

        path 是不含頁碼的基底（<名稱>.jpg）。頁數上限 99，由 max_pages 設定保證（config.py 載入時檢查）。
        .json 是整筆訂單共用的一份，不帶頁碼（見 json_path_for）。
        """
        return path.with_name(f"{path.stem}_p{page:02d}{path.suffix}")

    @staticmethod
    def json_path_for(sheet_path) -> Path:
        """任何一頁的圖片路徑 -> 這筆訂單共用的 .json 路徑（去掉檔名最後固定 4 個字元的 _pNN）。"""
        p = Path(sheet_path)
        return p.with_name(p.stem[:-4] + ".json")

    def _write_one(self, entries: List[Evidence], summary: Dict[str, Any], path: Path,
                   cost: Optional[dict] = None, t_enq: Optional[float] = None) -> None:
        cost = cost or {}
        t0 = time.perf_counter()
        queue_wait = (t0 - t_enq) if t_enq is not None else 0.0

        self._shrink_entries(entries)
        path.parent.mkdir(parents=True, exist_ok=True)

        render_s = encode_s = write_s = 0.0
        paths: List[Path] = []
        total_bytes = 0
        shape = None
        t_mark = time.perf_counter()
        # 逐頁渲染、編碼、寫檔：寫完一頁就釋放，記憶體只握一張畫布
        for page_no, img in enumerate(self._get_renderer().iter_pages(entries, summary), 1):
            t_a = time.perf_counter()
            render_s += t_a - t_mark
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                raise RuntimeError("cv2.imencode 失敗")
            t_b = time.perf_counter()
            encode_s += t_b - t_a
            page_path = self.page_path(path, page_no)
            tmp = page_path.with_name(page_path.name + ".tmp")
            # 先寫 .tmp 再改名：rclone / aibox_uploader 掃資料夾時不會撿到寫一半的檔案
            with open(tmp, "wb") as f:
                f.write(buf.tobytes())
            os.replace(tmp, page_path)
            write_s += time.perf_counter() - t_b
            paths.append(page_path)
            total_bytes += len(buf)
            shape = shape or img.shape
            t_mark = time.perf_counter()
        n_pages = len(paths)

        t_j = time.perf_counter()
        if self.save_json:
            self._dump_json(entries, summary, path.with_suffix(".json"), [p.name for p in paths])
        json_s = time.perf_counter() - t_j

        total_s = time.perf_counter() - t0
        main_total = float(cost.get("total", 0.0)) + float(cost.get("finalize", 0.0))
        self.last_timing = {
            "path": str(paths[0]), "paths": [str(p) for p in paths], "pages": n_pages,
            "bytes": total_bytes, "entries": len(entries),
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
        pages_txt = f"（共 {n_pages} 頁）" if n_pages > 1 else ""
        logger.info(
            f"[debug_sheet][timing] 單號={summary.get('order_number')} {summary.get('status')} {paths[0].name}{pages_txt} "
            f"{shape[1]}x{shape[0]} {total_bytes / 1e6:.2f}MB 證據{len(entries)}筆\n"
            f"    主迴圈(影響辨識): 累計 {main_total * 1000:.1f}ms / 記錄 {cost.get('n', 0)} 次"
            f"，單次最大 {float(cost.get('max', 0.0)) * 1000:.1f}ms({cost.get('max_label', '-')})"
            f"，finalize {float(cost.get('finalize', 0.0)) * 1000:.1f}ms"
            f"{('，分項 ' + by) if by else ''}\n"
            f"    背景(獨立 thread): 佇列等待 {queue_wait:.2f}s + 渲染 {render_s:.2f}s + 編碼 {encode_s:.2f}s"
            f" + 寫檔 {write_s:.2f}s + json {json_s:.2f}s = {total_s:.2f}s"
            f"（佇列深度 {cost.get('queue_depth', 0)}）\n"
            f"    累計 {tot['n']} 筆訂單：背景平均 {tot['total_sum'] / tot['n']:.2f}s / 最大 {tot['total_max']:.2f}s，"
            f"渲染平均 {tot['render_sum'] / tot['n']:.2f}s / 最大 {tot['render_max']:.2f}s"
        )
        if total_s > self.slow_render_s:
            logger.warning(
                f"[debug_sheet][timing] 背景出圖偏慢 {total_s:.1f}s（{n_pages} 頁，門檻 {self.slow_render_s:g}s）："
                f"渲染 {render_s:.1f}s 編碼 {encode_s:.1f}s 寫檔 {write_s:.1f}s。"
                f"持續偏慢請考慮 mode: ng_only、調低 jpeg_quality 或 max_pages"
            )

    @staticmethod
    def _dump_json(entries: List[Evidence], summary: Dict[str, Any], path: Path,
                   files: Optional[List[str]] = None) -> None:
        doc = {
            "version": SHEET_VERSION,
            "pages": len(files or []) or 1,
            "files": files or [],
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
