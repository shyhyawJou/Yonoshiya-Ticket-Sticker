"""吉野家大傳票的純 CV 版面分析（實驗中，app 尚未接上）。

det 模型框傳票文字有兩個老問題：沒看過的字框不好，以及数欄的數字太小常漏框
（或跟菜名黏成「1N豚丼・並盛」）。這裡改成先用框線把表格結構找出來，
文字框只在已知的格子裡找墨跡，不靠 det：

    □ | 数 |   メニュー名   | 金額 |
    □ |  1 | 牛丼・アタマ    |  661 |    <- 品項列：左框外有方框、数欄有數字
      |    | ▲▲取消▲▲      |      |    <- 段落標題：沒有方框、沒有數量
    □ |  1 | 牛丼・大        |  726 |

1. 表格骨架：長橫線（表格上框 / 表頭底線 / 表格底線）與從上框畫到底線的直欄線
   （左框、数欄右界、金額欄左界、右框）。條碼的直條沒有橫線穿過，自然不會入選。
2. 方向：数欄是最窄的那一欄，在左邊就是正向，在右邊就是倒過來。
3. 文字列：表頭那一列的高度就是列距，表頭底線到表格底線切成等高的列；
   紙張彎曲時每個 x 位置各自等分（_LineY / _Grid），字依中心點分到列與欄。
   段落標題（■■追加■■ / ●テイクアウト● / ▲▲取消▲▲）也是一列。
4. 每列三個彼此獨立的訊號：左框外的方框、数欄有墨跡、金額欄有墨跡。
   方框或数量任一成立就是品項列 —— 方框被筆打勾、被手指壓住都還有数量撐著。
5. 品名框取品名格內墨跡的實際左右邊界，空一格兩格的縮排自然吸收。
6. 單號：表格上方偏右那一塊，字最高的那一行（「伝票No」的深色橢圓標籤丟掉）。

紅筆先抹掉再二值化：現場會在方框或菜名上畫紅筆，不抹掉會混進文字框。

所有座標都在「轉正 + 去斜後」的影像 (TicketLayout.image) 上；
TicketLayout.matrix 是原始 crop -> 該影像的 2x3 仿射矩陣。
"""
from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np

Box = Tuple[float, float, float, float]  # (x0, y0, x1, y1)

# 去斜角度的搜尋範圍與細找步距（度）。mmr 的旋轉框已經大致對齊紙張，但疊在別張紙上時實測可斜到 5 度。
_SKEW_RANGE = 6.0
_SKEW_STEP = 0.125
# 長橫線的一段至少要有多長（crop 寬的比例）；斷成幾段的會再合併
_HLINE_FRAC = 0.20
# 方向可信的條件之一：表格高度離整數個列距最多差多少列
_MAX_MISFIT = 0.3
# 紅筆：R 比 G、B 都高出這麼多就當成紅筆抹掉
_RED_DELTA = 40


@dataclass
class CvRow:
    kind: str                         # "item" 品項列 / "header" 段落標題或說明
    k: int                            # 表頭下第幾列（從 0 起，空白列也算），對應 TicketLayout.cell_of
    y0: float
    y1: float
    name_box: Optional[Box] = None
    qty_box: Optional[Box] = None
    amount_box: Optional[Box] = None
    check_box: Optional[Box] = None
    name_quad: Optional[np.ndarray] = None  # 給 rec 的四點框（跟著列的斜度），見 pad_quad
    qty_quad: Optional[np.ndarray] = None
    has_check: bool = False
    has_qty: bool = False
    has_amount: bool = False


@dataclass
class TicketLayout:
    ok: bool
    reason: str = ""
    rotate: int = 0                   # 原始 crop 順時針轉幾度才是正向（0 / 90 / 180 / 270）
    angle: float = 0.0                # 去斜角度（度）
    image: Optional[np.ndarray] = None
    matrix: Optional[np.ndarray] = None
    cols: Tuple[float, ...] = ()      # 左框, 数欄右界, 金額欄左界, 右框（表格垂直中點 y_mid 的 x）
    col_slopes: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)   # 各欄線的 dx/dy（長表格 + 透視時不是 0）
    y_mid: float = 0.0
    y_top: float = 0.0
    y_header: float = 0.0
    y_bottom: float = 0.0
    text_h: float = 0.0
    rows: List[CvRow] = field(default_factory=list)
    order_no_box: Optional[Box] = None
    rec_image: Optional[np.ndarray] = None   # image 把框線塗成紙色；直接拿 CV 框給 rec 時用這張
    grid: Optional["_Grid"] = None
    # 方向可信（可以拿來取代 det 的翻轉比對）：表頭那列的高度剛好是列距、表格左邊有方框。
    # 版面判錯方向時這兩條不會同時成立 —— 倒過來看的話方框在表格右邊，根本不在搜尋範圍
    orientation_ok: bool = False
    orientation_note: str = ""

    def col_x(self, i: int, y):
        """第 i 條欄線（0 左框、1 数欄右界、2 金額欄左界、3 右框）在高度 y 的 x；y 可以是陣列。

        長傳票的欄線上下兩端可以差到 10 px（20261028_171039：相機透視，左右兩邊的直線
        斜向不同，去斜只能轉正其中一邊）。数欄的「1」緊貼欄線，用一個固定的 x 切欄會分錯。
        """
        return self.cols[i] + self.col_slopes[i] * (np.asarray(y, dtype=np.float64) - self.y_mid)

    def cell_of(self, x: float, y: float) -> Optional[Tuple[int, str]]:
        """image 上一點落在哪一格：(第幾列, "qty" / "name" / "amt")；表格外為 None。

        給 det 框分列用：det 讀字比較準，但會漏框、會把數量跟菜名框成一個；
        版面交給 CV，每個 det 框依中心點歸到 CV 的列與欄。
        """
        if self.grid is None:
            return None
        left, x_qty, x_amt, right = (float(self.col_x(i, y)) for i in range(4))
        if left < x < x_qty:
            col = "qty"
        elif x_qty <= x < x_amt:
            col = "name"
        elif x_amt <= x < right:
            col = "amt"
        else:
            return None
        r = self.grid.row_at(x, y)
        return (int(r), col) if 0 <= r < self.grid.n else None


# ----------------------------------------------------------------------------
# 前處理
# ----------------------------------------------------------------------------
def _erase_red(bgr: np.ndarray) -> np.ndarray:
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    red = (r - np.maximum(g, b) > _RED_DELTA) & (r > 90)
    if not red.any():
        return bgr
    out = bgr.copy()
    red = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    paper = np.median(bgr[~red].reshape(-1, 3), axis=0) if (~red).any() else (255, 255, 255)
    out[red] = paper
    return out


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 7)


def _rotate_quarter(img: np.ndarray, cw: int) -> np.ndarray:
    return {0: lambda i: i,
            90: lambda i: cv2.rotate(i, cv2.ROTATE_90_CLOCKWISE),
            180: lambda i: cv2.rotate(i, cv2.ROTATE_180),
            270: lambda i: cv2.rotate(i, cv2.ROTATE_90_COUNTERCLOCKWISE)}[cw](img)


def _quarter_matrix(cw: int, w: int, h: int) -> np.ndarray:
    """原始 (w, h) 影像 -> 順時針轉 cw 度後的 3x3 矩陣。"""
    return np.array({
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, h - 1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]],
    }[cw], dtype=np.float64)


def _line_masks(mask: np.ndarray, v_len: int):
    """只留長橫線 / 長直線的遮罩（先垂直於線的方向加粗 1 px，斜一點也不會斷）。"""
    h, w = mask.shape
    th = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
    tv = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    hm = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(w * _HLINE_FRAC)), 1)))
    vm = cv2.morphologyEx(tv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len)))
    return hm, vm


def _skew_by_search(mask: np.ndarray) -> float:
    """逐角度轉轉看，取框線投影最尖的角度（度）。慢（約 26 次旋轉 + 形態學），只當備援。"""
    h, w = mask.shape
    v_len = max(15, int(h * 0.035))
    center = (w / 2, h / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        hm, vm = _line_masks(rot, v_len)
        return float((hm.sum(axis=1, dtype=np.float64) ** 2).sum() + (vm.sum(axis=0, dtype=np.float64) ** 2).sum())

    coarse = max(np.arange(-_SKEW_RANGE, _SKEW_RANGE + 1e-6, 0.5), key=score)
    return float(max(np.arange(coarse - 0.5, coarse + 0.5 + 1e-6, _SKEW_STEP), key=score))


def _estimate_skew(mask: np.ndarray) -> float:
    """去斜角度（度，給 cv2.getRotationMatrix2D 用）。

    只看框線不看文字：品項多的長傳票整片都是字，拿全圖投影估角度會被字列帶偏
    （實測 20261022_150725 估成 1.75 度，其實是 0 度，轉完直欄線全斷）。

    做法：先用「加粗 2 px、短一點」的開運算撈出框線（斜到 4 度左右都撈得到），
    每條線用 cv2.fitLine 擬合角度，依長度加權取中位數 —— 一次算完，不用逐角度轉。
    撈不到兩條以上的線（斜太多、整張很糊）才退回逐角度搜尋。
    """
    h, w = mask.shape
    angles, weights = [], []
    for horizontal in (True, False):
        thick = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5) if horizontal else (5, 1)))
        k = (max(10, int(w * 0.15)), 1) if horizontal else (1, max(15, int(h * 0.06)))
        op = cv2.morphologyEx(thick, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(op, 8)
        for i in range(1, n):
            x, y, bw, bh, _ = stats[i]
            length = bw if horizontal else bh
            if length < (0.3 * w if horizontal else 0.1 * h):
                continue
            # 用原遮罩的像素（不是加粗後的）擬合
            sub = (lab[y:y + bh, x:x + bw] == i) & (mask[y:y + bh, x:x + bw] > 0)
            ys, xs = np.nonzero(sub)
            if len(xs) < 10:
                continue
            step = max(1, len(xs) // 400)  # 擬合直線不需要每個像素，抽樣 400 點就夠準
            xs, ys = xs[::step], ys[::step]
            vx, vy, _, _ = cv2.fitLine(np.stack([xs, ys], 1).astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            if horizontal:
                a = np.degrees(np.arctan2(vy, vx))
            else:  # 直線換算成「相對垂直方向」的傾斜，跟橫線同號
                a = -np.degrees(np.arctan2(vx, vy))
            a = (a + 90) % 180 - 90
            if abs(a) <= _SKEW_RANGE:
                angles.append(a); weights.append(float(length))
    if len(angles) < 2:
        return _skew_by_search(mask)
    order = np.argsort(angles)
    cum = np.cumsum(np.array(weights)[order])
    return float(np.array(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


# ----------------------------------------------------------------------------
# 表格骨架
# ----------------------------------------------------------------------------
def _cluster_segments(segs, gap: float, kink: float):
    """同一條線被切成好幾段時合併。segs: (位置, 起點, 終點)。

    兩種斷法：同一高度被遮住一段（迴紋針、印刷斷點），以及紙張微折 ——
    折線兩側的同一條框線會差 7~9 px（20261022_154841），只有「頭尾相接、
    沒有並排」的兩段才用 kink 這個較寬的容許值，免得把相鄰兩條框線併成一條。
    """
    out = []
    for pos, a0, a1 in sorted(segs, key=lambda t: t[1]):
        for L in out:
            overlap = min(a1, L[2]) - max(a0, L[1])
            near = a0 <= L[2] + gap and a1 >= L[1] - gap
            if near and (abs(pos - L[0]) <= 3 or (abs(pos - L[0]) <= kink and overlap <= 10)):
                L[0] = (L[0] * (L[2] - L[1]) + pos * (a1 - a0)) / max(1, (L[2] - L[1]) + (a1 - a0))
                L[1] = min(L[1], a0); L[2] = max(L[2], a1)
                break
        else:
            out.append([pos, a0, a1])
    return sorted(tuple(L) for L in out)


def _h_lines(hm: np.ndarray) -> List[Tuple[float, int, int]]:
    """長橫線 (y, x0, x1)。hm = _line_masks 的橫線遮罩。

    粗細用「面積 / 長度」（平均粗細）判斷，不用外框高度：線稍微斜一點，外框就比線本身高很多。
    """
    h, w = hm.shape
    n, _, stats, cents = cv2.connectedComponentsWithStats(hm, 8)
    segs = [(float(cents[i][1]), int(stats[i][0]), int(stats[i][0] + stats[i][2]))
            for i in range(1, n)
            if stats[i][4] / max(1, stats[i][2]) <= max(10, 0.03 * h)]  # 模糊的傳票框線二值化後可到 9~10 px
    return _cluster_segments(segs, gap=0.2 * w, kink=10)


def _v_lines(vm: np.ndarray) -> List[Tuple[float, int, int, float]]:
    """直線 (x, y0, y1, 斜率 dx/dy)，x 是線段中點 (y0+y1)/2 的位置。vm = _line_masks 的直線遮罩。

    長表格的欄線會斜（20261028_171039：表格高 360 px，相機透視讓右邊的欄線上下差 8 px，
    去斜只能轉正其中一邊）。以前的傳票表格只有 100 px 高，斜一點看不出來，所以兩個地方
    都沒考慮到：
      * 粗細用外框寬判斷 —— 3 px 的線斜 8 px，外框就有 11 px，被當成太粗丟掉。
        改用平均粗細（面積 / 長度）
      * 斷段接回同一條線時跟「目前的平均位置」比 —— 斜的長線越接越偏。改成用前一段的
        位置與斜率推算這一段應該在哪，再擬合整條線的斜率
    """
    h, w = vm.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(vm, 8)
    segs = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area / max(1, bh) > max(10, 0.03 * w):
            continue
        k = 0.0
        if bh >= 40:  # 夠長的一段自己就量得出斜率
            ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
            step = max(1, len(xs) // 400)
            k = float(np.polyfit(ys[::step], xs[::step], 1)[0])
        segs.append((float(cents[i][0]), y, y + bh, k))
    segs.sort(key=lambda t: t[1])
    gap = 0.05 * h
    chains: List[list] = []   # [[(x, y0, y1, k), ...]]
    for seg in segs:
        x, y0, y1, k = seg
        for c in chains:
            lx, ly0, ly1, lk = c[-1]
            pred = lx + lk * ((y0 + y1) / 2 - (ly0 + ly1) / 2)
            if abs(x - pred) <= 3 and y0 <= max(e[2] for e in c) + gap and y1 >= c[0][1] - gap:
                c.append(seg)
                break
        else:
            chains.append([seg])
    out = []
    for c in chains:
        y0, y1 = min(e[1] for e in c), max(e[2] for e in c)
        ym = (y0 + y1) / 2
        mids = np.array([(e[1] + e[2]) / 2 for e in c])
        xs = np.array([e[0] for e in c])
        wts = np.array([max(1, e[2] - e[1]) for e in c], dtype=np.float64)
        if len(c) >= 2 and np.ptp(mids) > 20:
            slope = float(np.polyfit(mids, xs, 1, w=np.sqrt(wts))[0])
        else:
            slope = float(np.average([e[3] for e in c], weights=wts))
        out.append((float(np.average(xs + slope * (ym - mids), weights=wts)), y0, y1, slope))
    return sorted(out)


# 欄界在表格寬中的位置（数欄右界、金額欄左界），量自 special_test 與 10 月現場傳票。
# 範圍給寬一點，只用來從直線裡認出哪一條是哪一條；多組可行時取最接近典型值的。
_QTY_FRAC = (0.05, 0.22)
_AMT_FRAC = (0.60, 0.85)
_QTY_AT, _AMT_AT = 0.10, 0.73


def _fit_cols(xs: List[float], w: int, mirrored: bool):
    """從直線 x 裡挑出 (左框, 数欄右界, 金額欄左界, 右框)，回傳 (誤差, cols)；挑不出來回 None。

    mirrored=True 表示影像是倒的（数欄在右）。cols 一律是影像上由左到右的四條線。
    正向的表格用倒過來的比例套不進去（金額欄在右邊佔 0.27，超出数欄的範圍），反之亦然，
    所以這也是判斷方向的依據。
    """
    best = None
    for i, L in enumerate(xs):
        for R in xs[i + 1:]:
            tw = R - L
            if tw < 0.5 * w:
                continue
            inner = [x for x in xs if L + 3 < x < R - 3]
            fr = [(((R - x) if mirrored else (x - L)) / tw, x) for x in inner]
            qty = [(abs(f - _QTY_AT), x) for f, x in fr if _QTY_FRAC[0] < f < _QTY_FRAC[1]]
            amt = [(abs(f - _AMT_AT), x) for f, x in fr if _AMT_FRAC[0] < f < _AMT_FRAC[1]]
            if not qty or not amt:
                continue
            eq, q = min(qty)
            ea, a = min(amt)
            cand = (eq + ea, (L, a, q, R) if mirrored else (L, q, a, R))
            if best is None or cand[0] < best[0]:
                best = cand
    return best


def _find_table(hm: np.ndarray, vm: np.ndarray):
    """回傳 (cols, mirrored, y_top, y_header, y_bottom, 斜率, y_mid) 或 None。

    cols 是影像上由左到右的四條欄線在表格垂直中點 y_mid 的 x；斜率是各自的 dx/dy。
    """
    h, w = hm.shape
    hl = _h_lines(hm)
    # 貼著影像邊緣的直線是紙張邊緣 / 背景，不是欄線（右框可能離邊緣只有 5 px，只排除最外 3 px）
    vl = [v for v in _v_lines(vm) if 3 < v[0] < w - 3]
    tol = max(4.0, 0.012 * h)
    cands = []
    for i, (yt, tx0, tx1) in enumerate(hl):
        for yb, bx0, bx1 in hl[i + 1:]:
            if yb - yt < 0.06 * h:
                continue
            # 欄線模糊時會斷好幾段（20261019_150949 的数欄線只剩 40%），涵蓋 40% 就收；
            # 但至少兩條要「從上框畫到底線」，加上比例要套得進去，背景的紙邊湊不出來
            ym = (yt + yb) / 2
            inside = [(x + k * (ym - (y0 + y1) / 2), y0, y1, k) for x, y0, y1, k in vl
                      if y0 >= yt - 2 * tol and y1 <= yb + 2 * tol
                      and (min(y1, yb) - max(y0, yt)) >= 0.4 * (yb - yt)]
            xs, slope_of = [], {}
            for x, _, _, k in sorted(inside):  # 同一條線的兩段 x 很近，合併
                if not xs or x - xs[-1] >= 6:
                    xs.append(x); slope_of[x] = k
            anchored = [x for x, y0, y1, _ in inside if abs(y0 - yt) <= 2 * tol and abs(y1 - yb) <= 2 * tol]
            for mirrored in (False, True):
                fit = _fit_cols(xs, w, mirrored)
                if fit is None:
                    continue
                err, cols = fit
                L, R = cols[0], cols[3]
                tw = R - L
                # 上框、底線都要橫過表格的一半以上（光線不均時只抓得到一部分）
                if (min(tx1, R) - max(tx0, L)) < 0.5 * tw or (min(bx1, R) - max(bx0, L)) < 0.5 * tw:
                    continue
                if sum(1 for c in cols if any(abs(c - x) < 6 for x in anchored)) < 2:
                    continue
                cands.append((yb - yt, err, cols, mirrored, yt, yb, tuple(slope_of[c] for c in cols)))
    if not cands:
        return None
    tallest = max(c[0] for c in cands)
    _, _, cols, mirrored, yt, yb, slopes = min((c for c in cands if c[0] >= 0.9 * tallest), key=lambda c: c[1])
    L, R = cols[0], cols[3]
    tw = R - L
    # 表頭底線：倒過來時表頭在下面，取靠下框那條
    spans = [y for y, x0, x1 in hl if yt + 4 < y < yb - 4 and (min(x1, R) - max(x0, L)) >= 0.5 * tw]
    y_header = (spans[-1] if mirrored else spans[0]) if spans else None
    return cols, mirrored, yt, y_header, yb, slopes, (yt + yb) / 2


# ----------------------------------------------------------------------------
# 列與格子
# ----------------------------------------------------------------------------
class _LineY:
    """一條橫框線的 y(x)。

    沿線每隔一段取樣，中間內插、兩端線性外插。紙張微彎、微折、相機透視時，
    同一條框線左右兩端可以差到 7~9 px —— 整張表格用同一個 y 切列，
    就會把一列切成兩半、金額欄跑到下一列（20261019_190438、20261022_165725）。
    """

    def __init__(self, line_mask: np.ndarray, y_est: float, x0: float, x1: float, tol: float):
        H = line_mask.shape[0]
        xs, ys = [], []
        for xc in np.linspace(x0 + 3, x1 - 3, 16):
            ya, yb = int(max(0, y_est - tol)), int(min(H, y_est + tol + 1))
            hit = np.nonzero(line_mask[ya:yb, int(xc) - 3:int(xc) + 4].any(axis=1))[0]
            if len(hit):
                xs.append(float(xc)); ys.append(ya + float(np.median(hit)))
        if len(xs) < 2:
            xs, ys = [x0, x1], [y_est, y_est]
        self.xs, self.ys = np.array(xs), np.array(ys)
        k = min(4, len(xs))
        self.slope_l = float(np.polyfit(self.xs[:k], self.ys[:k], 1)[0])
        self.slope_r = float(np.polyfit(self.xs[-k:], self.ys[-k:], 1)[0])

    def __call__(self, x):
        """x 可以是單一數值或 numpy 陣列（分格時一次算全部連通元件）。"""
        xa = np.asarray(x, dtype=np.float64)
        y = np.interp(xa, self.xs, self.ys)
        y = np.where(xa < self.xs[0], self.ys[0] + self.slope_l * (xa - self.xs[0]), y)
        y = np.where(xa > self.xs[-1], self.ys[-1] + self.slope_r * (xa - self.xs[-1]), y)
        return float(y) if y.ndim == 0 else y


class _Grid:
    """表頭底線到表格底線之間等分成 n 列，每個 x 位置各自等分。

    不用投影找列：模糊的傳票列與列之間的空白只降到 1~7 px，門檻怎麼調都會黏列或斷字。
    改用結構 —— 表頭那一列的高度就是列距，POS 每列等高，所以表頭底線到表格底線
    正好是整數個列距（實測 26 張誤差都在 ±0.13 列以內）。
    """

    def __init__(self, top: _LineY, header: _LineY, bottom: _LineY, xc: float):
        self.header, self.bottom = header, bottom
        p0 = header(xc) - top(xc)
        ratio = (bottom(xc) - header(xc)) / p0
        self.n = max(1, int(round(ratio)))
        self.misfit = abs(ratio - self.n)   # 表格高度離整數個列距差多少（實測 ±0.13 以內）
        self.pitch = (bottom(xc) - header(xc)) / self.n

    def row_at(self, x: float, y: float) -> float:
        """(x, y) 落在第幾列（浮點數，0 = 表頭下第一列的上緣）。"""
        yh = self.header(x)
        return (y - yh) / ((self.bottom(x) - yh) / self.n)

    def y_at(self, x: float, k: float) -> float:
        yh = self.header(x)
        return yh + k * (self.bottom(x) - yh) / self.n


# 文字範圍：同一段字裡字與字的空隙（含全形空白「牛皿　ＦＰ４食」約 1 個字高）不會超過這個
_WORD_GAP = 1.6


def _text_extent(comps, th: float):
    """一格裡的連通元件 -> 真正是字的那些。

    格子裡除了字，還有灰塵、紙纖維、框線抹掉後的殘段；全部取聯集的話，一顆離字很遠的雜點
    就會把框拉到整格那麼寬，rec 讀到一大段空白就容易讀錯（det 的框貼著字，沒有這個問題）。
      1. 只用「像字的」元件分段：長邊至少半個字高。實測（字高 18 px）雜點是一串長寬 2~7 px 的
         小點、間距 2~10 px 一路排到格子邊；字的筆畫（連扁扁的「ー」寬 17 px）都超過 9 px。
         不能看高度：「ー」只有 5 px 高，用高度判斷會把「黒カレー」切成「黒カレ」
      2. 由左到右，空隙超過 _WORD_GAP 個字高就斷開，取墨跡最多的那一段
      3. 濁點、「・」這類小元件只要落在那一段的範圍內就加回來
    """
    big = sorted((c for c in comps if max(c[2], c[3]) >= 0.5 * th), key=lambda c: c[0])
    if not big:
        return comps
    runs, cur = [], [big[0]]
    for c in big[1:]:
        if c[0] - max(e[0] + e[2] for e in cur) > _WORD_GAP * th:
            runs.append(cur); cur = [c]
        else:
            cur.append(c)
    runs.append(cur)
    best = max(runs, key=lambda r: sum(e[4] for e in r))
    lo = min(e[0] for e in best) - 0.2 * th
    hi = max(e[0] + e[2] for e in best) + 0.2 * th
    return [c for c in comps if lo <= c[5] <= hi]


def _cell_quad(comps, grid: _Grid, k: int, th: float) -> Tuple[Box, np.ndarray]:
    """一格裡的連通元件 -> (文字外框, 跟著該列斜度走的四點框)。只框真正是字的範圍（見 _text_extent）。"""
    comps = _text_extent(comps, th)
    x0 = min(c[0] for c in comps); x1 = max(c[0] + c[2] for c in comps)
    y0 = min(c[1] for c in comps); y1 = max(c[1] + c[3] for c in comps)
    # 上下緣以「離該列上緣多遠」表示，再沿著列的斜度畫回去
    t_off = min(c[1] - grid.y_at(c[5], k) for c in comps)
    b_off = max(c[1] + c[3] - grid.y_at(c[5], k) for c in comps)
    quad = np.array([[x0, grid.y_at(x0, k) + t_off], [x1, grid.y_at(x1, k) + t_off],
                     [x1, grid.y_at(x1, k) + b_off], [x0, grid.y_at(x0, k) + b_off]], dtype=np.float32)
    return (float(x0), float(y0), float(x1), float(y1)), quad


def _is_checkbox(c, th: float) -> bool:
    """方框：大小約一個字高、接近正方形。被打勾時不一定中空，所以不看內部。"""
    w, h = c[2], c[3]
    return 0.3 * th <= w <= 1.3 * th and 0.3 * th <= h <= 1.3 * th and 0.6 <= w / max(h, 1) <= 1.6


def _find_order_no(gray: np.ndarray, cols, y_top: float, th: float) -> Optional[Box]:
    """表格上方偏右、字最高的那一行（單號）。

    旁邊「●伝票No●」的深色橢圓標籤比單號的字矮（約六成），只取最高那群字。
    """
    left, _, _, right = cols
    tw = right - left
    H, W = gray.shape
    x0, x1 = int(max(0, left + 0.40 * tw)), int(min(W, right + 0.08 * tw))
    y0, y1 = int(max(0, y_top - 6.0 * th)), int(max(0, y_top - 0.4 * th))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    sub = gray[y0:y1, x0:x1]
    _, bw_img = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw_img, 8)
    rh, rw = bw_img.shape
    # 碰到搜尋範圍邊緣的是紙張外的背景（crop 邊緣那條暗帶），不是字
    big = [tuple(int(v) for v in stats[i]) for i in range(1, n)
           if stats[i][4] >= 4 and 0.5 * th <= stats[i][3] <= 3.5 * th
           and stats[i][0] > 0 and stats[i][1] > 0
           and stats[i][0] + stats[i][2] < rw and stats[i][1] + stats[i][3] < rh]
    if not big:
        return None
    tall = max(c[3] for c in big)
    core = [c for c in big if c[3] >= 0.8 * tall]
    cy = float(np.median([c[1] + c[3] / 2 for c in core]))
    core = [c for c in core if abs(c[1] + c[3] / 2 - cy) < 0.5 * tall]
    lx0 = min(c[0] for c in core); lx1 = max(c[0] + c[2] for c in core)
    ly0 = min(c[1] for c in core); ly1 = max(c[1] + c[3] for c in core)
    return (float(x0 + lx0), float(y0 + ly0), float(x0 + lx1), float(y0 + ly1))


# ----------------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------------
def analyze(crop_bgr: np.ndarray) -> TicketLayout:
    """分析一張大傳票 crop（任意 0/90/180/270 方向）。"""
    if crop_bgr is None or crop_bgr.size == 0:
        return TicketLayout(ok=False, reason="空影像")
    clean = _erase_red(crop_bgr)
    h0, w0 = clean.shape[:2]
    # 先二值化再去斜：反過來的話，內插會把 1 px 的框線抹淡，adaptiveThreshold 就抓不到了
    mask0 = _ink_mask(cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY))

    found = None
    for base in (0, 90):
        m = _rotate_quarter(mask0, base)
        hh, ww = m.shape
        angle = _estimate_skew(m)
        R = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
        mask = cv2.warpAffine(m, R, (ww, hh), flags=cv2.INTER_NEAREST)
        hm, vm = _line_masks(mask, max(15, int(hh * 0.035)))
        t = _find_table(hm, vm)
        if t is not None:
            found = (base, angle, R, mask, hm, vm, t)
            break
    if found is None:
        return TicketLayout(ok=False, reason="找不到表格框線")

    base, angle, R, mask, hm, vm, t = found
    rotate = base
    if t[1]:
        # 数欄（最窄那欄）在右邊 = 整張倒過來。去斜是繞影像中心轉，跟轉 180 度可以交換，
        # 所以已經去斜的遮罩直接轉 180 度即可，角度與 R 都沿用
        rotate = (base + 180) % 360
        mask, hm, vm = (cv2.rotate(x, cv2.ROTATE_180) for x in (mask, hm, vm))
        t = _find_table(hm, vm)
    hh, ww = mask.shape
    gray = cv2.warpAffine(cv2.cvtColor(_rotate_quarter(clean, rotate), cv2.COLOR_BGR2GRAY), R, (ww, hh),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 給 rec 的影像用原圖（沒抹紅筆）：紅筆畫過菜名時，抹掉會連字的筆畫一起抹斷
    # （sp_20261001_135649 的とん汁被讀成豚丼・並）。紅筆只需要在找框線與分格時排除。
    image = cv2.warpAffine(_rotate_quarter(crop_bgr, rotate), R, (ww, hh),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    matrix = (np.vstack([R, [0, 0, 1]]) @ _quarter_matrix(rotate, w0, h0))[:2]
    if t is None:
        return TicketLayout(ok=False, reason="轉正後找不到表格框線", rotate=rotate, angle=angle,
                            image=image, matrix=matrix)
    cols, mirrored, yt, yh, yb, slopes, y_mid = t
    if mirrored:
        return TicketLayout(ok=False, reason="轉正後数欄仍在右邊", rotate=rotate, angle=angle,
                            image=image, matrix=matrix)
    if yh is None:
        return TicketLayout(ok=False, reason="找不到表頭底線", rotate=rotate, angle=angle,
                            image=image, matrix=matrix, cols=cols)
    left, x_qty, x_amt, right = cols

    # 去掉框線再找字，否則欄線會被當成「1」
    lines = cv2.dilate(cv2.bitwise_or(hm, vm), np.ones((3, 3), np.uint8))
    text_mask = cv2.subtract(mask, lines)
    rec_image = image.copy()
    rec_image[lines > 0] = np.median(image.reshape(-1, 3), axis=0)

    # 三條橫框線各自沿線取樣 y(x)，列界跟著紙張的彎曲走
    tol = 0.35 * (yh - yt)
    grid = _Grid(_LineY(hm, yt, left, right, tol), _LineY(hm, yh, left, right, tol),
                 _LineY(hm, yb, left, right, tol), (left + right) / 2)
    th = 0.62 * grid.pitch  # 字高約為列距的六成
    tall = 0.45 * th        # 「這格有字」：至少有一個元件這麼高（框線殘段、雜點是扁的）

    # 每個連通元件依「中心點在第幾列、哪一欄」分格。逐元件判斷而不是逐格切矩形，
    # 列是斜的或彎的也不會切到字
    # 一張傳票有上千個連通元件（大多是雜點），先用 numpy 一次濾掉、算好欄與列，
    # 剩下的才逐個收進格子（逐個用 Python 判斷時這段佔了版面分析一半的時間）
    _, _, stats, cents = cv2.connectedComponentsWithStats(text_mask, 8)
    stats, cents = stats[1:], cents[1:]
    cx, cy = cents[:, 0], cents[:, 1]
    # 欄界取每個元件自己那個高度的位置（欄線是斜的，見 TicketLayout.col_x）
    lx, qx, ax, rx = (cols[i] + slopes[i] * (cy - y_mid) for i in range(4))
    # 雜點；或太大（迴紋針、手指）
    keep = (stats[:, 4] >= 3) & (stats[:, 3] <= 1.3 * grid.pitch) & (cx > lx - 2.6 * th) & (cx < rx)
    col_names = np.array(["chk", "qty", "name", "amt", ""])
    col_idx = np.select([cx < lx - 1, (cx > lx) & (cx < qx), (cx > qx) & (cx < ax), cx > ax],
                        [0, 1, 2, 3], default=4)
    rk = np.full(len(cx), -1.0)
    if keep.any():
        rk[keep] = grid.row_at(cx[keep], cy[keep])
    keep &= (col_idx < 4) & (rk >= 0) & (rk < grid.n)
    cells: dict = {}
    for i in np.nonzero(keep)[0]:
        x, y, w_, h_, area = (int(v) for v in stats[i])
        cells.setdefault((int(rk[i]), str(col_names[col_idx[i]])), []).append(
            (x, y, w_, h_, area, float(cx[i]), float(cy[i])))

    rows: List[CvRow] = []
    xc = (left + right) / 2
    for k in range(grid.n):
        has = {c: any(e[3] >= tall for e in cells.get((k, c), [])) for c in ("qty", "name", "amt")}
        if not any(has.values()):
            continue  # 空白列
        row = CvRow(kind="header", k=k, y0=grid.y_at(xc, k), y1=grid.y_at(xc, k + 1),
                    has_qty=has["qty"], has_amount=has["amt"])
        if has["name"]:
            row.name_box, row.name_quad = _cell_quad(cells[(k, "name")], grid, k, th)
        if has["qty"]:
            row.qty_box, row.qty_quad = _cell_quad(cells[(k, "qty")], grid, k, th)
        if has["amt"]:
            row.amount_box, _ = _cell_quad(cells[(k, "amt")], grid, k, th)
        boxes = [e for e in cells.get((k, "chk"), []) if _is_checkbox(e, th)]
        if boxes:
            e = max(boxes, key=lambda e: e[4])
            row.check_box = (float(e[0]), float(e[1]), float(e[0] + e[2]), float(e[1] + e[3]))
        rows.append(row)

    # 方框一定排成一直行、大小一樣；跟大多數對不上的是雜點（表格外的筆跡、別張紙）
    found = [r.check_box for r in rows if r.check_box is not None]
    if found:
        mx = float(np.median([(b[0] + b[2]) / 2 for b in found]))
        ms = float(np.median([b[3] - b[1] for b in found]))
        for r in rows:
            b = r.check_box
            if b is not None and (abs((b[0] + b[2]) / 2 - mx) > 0.4 * th
                                  or abs((b[3] - b[1]) - ms) > 0.35 * ms):
                r.check_box = None
    for r in rows:
        r.has_check = r.check_box is not None
        r.kind = "item" if (r.has_check or r.has_qty) else "header"

    order_box = _find_order_no(gray, cols, yt, th)
    n_item = sum(1 for r in rows if r.kind == "item")
    n_chk = sum(1 for r in rows if r.has_check)
    if n_item == 0:
        o_ok, o_note = False, "沒有品項列"
    elif n_chk == 0:
        o_ok, o_note = False, "表格左邊找不到方框"
    elif grid.misfit > _MAX_MISFIT:
        o_ok, o_note = False, f"表格高度不是整數個列距（差 {grid.misfit:.2f} 列）"
    else:
        o_ok, o_note = True, f"方框 {n_chk}/{n_item}、列距誤差 {grid.misfit:.2f}"
    return TicketLayout(ok=True, rotate=rotate, angle=angle, image=image, matrix=matrix,
                        orientation_ok=o_ok, orientation_note=o_note,
                        cols=cols, col_slopes=slopes, y_mid=y_mid, y_top=yt, y_header=yh, y_bottom=yb,
                        text_h=th, rows=rows, order_no_box=order_box, rec_image=rec_image, grid=grid)


def pad_quad(quad: np.ndarray, th: float, shape, px: float = 0.1, py: float = 0.1) -> np.ndarray:
    """CV 緊貼的四點框 -> 給 rec 的四點框：沿框自己的方向外擴一點。

    外擴量實測（131 個品名格，rec_image 已抹框線）：0.1/0.1 最好；0.25/0.2 次之；
    再大就會吃到上下兩列的字，0.6/0.5 時比對成功數掉到三分之一。
    """
    H, W = shape[:2]
    q = np.asarray(quad, dtype=np.float32).copy()
    along = q[1] - q[0]
    along /= max(1e-6, float(np.linalg.norm(along)))
    down = q[3] - q[0]
    down /= max(1e-6, float(np.linalg.norm(down)))
    q[0] += (-along * px - down * py) * th
    q[1] += (along * px - down * py) * th
    q[2] += (along * px + down * py) * th
    q[3] += (-along * px + down * py) * th
    q[:, 0] = np.clip(q[:, 0], 0, W - 1)
    q[:, 1] = np.clip(q[:, 1], 0, H - 1)
    return q


# ----------------------------------------------------------------------------
# OCR 線程 -> 解析器的交接
# ----------------------------------------------------------------------------
# 吉野家大傳票開 CV 輔助時，AsyncOCR 在 det 之前先跑 analyze 決定方向（省掉 det 的翻轉比對），
# 解析器接著又需要同一份版面。frame_crop 從 OCR callback 一路原封不動傳到 OrderParser.parse
# （同一個 ndarray 物件），所以用物件本身當鑰匙交接，不用改中間每一層的參數。
# 只留最近一份；拿到的不是同一個物件（被複製過）就當沒有，解析器自己重算 —— 結果一樣，只是慢一點。
_handoff_lock = threading.Lock()
_handoff: Optional[tuple] = None   # (weakref(影像), TicketLayout)


def rebase(layout: TicketLayout, applied_cw: int, w0: int, h0: int) -> TicketLayout:
    """版面是對原始 crop（w0 x h0）算的；crop 之後被順時針轉了 applied_cw 度，換算成對轉完的影像。"""
    if layout.matrix is None or applied_cw % 360 == 0:
        return layout
    m = np.vstack([layout.matrix, [0, 0, 1]]) @ np.linalg.inv(_quarter_matrix(applied_cw % 360, w0, h0))
    return replace(layout, matrix=m[:2], rotate=(layout.rotate - applied_cw) % 360)


def hand_off(img: np.ndarray, layout: TicketLayout) -> None:
    global _handoff
    with _handoff_lock:
        _handoff = (weakref.ref(img), layout)


def take_over(img: np.ndarray) -> Optional[TicketLayout]:
    """取出 hand_off 交給這張影像的版面（只能取一次）；沒有就回 None。"""
    global _handoff
    with _handoff_lock:
        entry, _handoff = _handoff, None
    if entry is not None and entry[0]() is img:
        return entry[1]
    return None
