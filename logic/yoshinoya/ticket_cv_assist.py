"""吉野家大傳票的 CV 輔助解析（ticket_cv_assist: true 時由 order_parser.OrderParser 呼叫）。

分工：版面交給 CV，字交給 det。
  * CV（ticket_cv_layout.analyze）用表格框線與左邊的方框決定：哪幾列是品項、哪些是
    段落標題（■■追加■■ / ●テイクアウト● / ▲▲取消▲▲）、数欄在哪、單號在哪。
  * det 框依中心點分到 CV 的列與欄，字照樣用 det+rec 讀到的。
  * 只有 det 讀不好的格子才用 CV 框補讀，log 一律標 [CV補救]：
      品名  det 漏框 / 只框到一部分 / det 讀到的字跟菜單對得不完整（分數 < 100）
            —— 補讀後兩種讀法取比對分數高的，同分用 det
      數量  det 框 + predict_quantity（只解碼 1~9 的正方形 crop），一致就用、不一致取信心高的，
            都讀不到但数欄有墨跡就當 1（跟 legacy 的預設 1 一樣，只是這裡有方框 / 墨跡證明這列是品項）
      單號  CV 單號框附近的 det 框；沒有才用 CV 框讀

為什麼不直接全用 CV 框讀：rec 是拿 det 的切圖訓練的，CV 框給 rec 平均讀得比 det 框差
（2026-10 的 41 張測試：只用 CV 框 27 張全對、現行 det 32 張、這個分工 38/39）。

取消段落目前只記 log，品項照常核對（之後要通知前端刪除，再在這裡接）。
找不到表格（紙張被切到、疊在別張上）回傳 None，由呼叫端走原本的 legacy 解析。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger

from logic.yoshinoya.ticket_cv_layout import TicketLayout, analyze, pad_quad, take_over

# 段落標題的 rec 常讀歪（■■追加■■ -> 「豚皿追力皿皿」），只認關鍵的那個字
_CANCEL_RE = re.compile(r"取\s*消")
_SECTION_RE = re.compile(r"追|取\s*消")
# det 把数欄的數字跟菜名框成一個時的前綴（「1N豚丼・並盛」的 N 是数欄的欄線）
_LEAD_QTY_RE = re.compile(r"^\s*([1-9]\d?)\s*[N|]?\s*")
_DIGITS_RE = re.compile(r"[1-9]\d*")
# det 框寬不到 CV 看到的品名墨跡寬的這個比例 = 只框到一部分（「チーズ」只框到「ズ」）
_PARTIAL_COVER = 0.5
# 與 legacy parse() 相同的門檻
_MIN_DET_SCORE, _MIN_DET_SCORE_NEGI = 0.5, 0.3
_MIN_MATCH_SCORE = 40


@dataclass
class _DetBox:
    idx: int          # dt_boxes 的 index（除錯總覽畫框用）
    x0: float
    x1: float
    text: str
    conf: float
    cross_qty: bool   # 左緣跨進数欄（数量跟菜名被框成一個）


@dataclass
class CvParseResult:
    items: List[Tuple[str, int]]
    order_number: Optional[str]
    debug_rows: List[dict]
    menu_bottom_line: Optional[tuple]       # crop 座標的兩端點，給 overlay
    layout: TicketLayout
    rescued: List[str] = field(default_factory=list)   # 補救過的格子，log 摘要用


class CvAssist:
    def __init__(self, folded, menus_ticket, fuzzy_matcher, rec_pred):
        self.folded = folded
        self.menus_ticket = menus_ticket
        self.fuzzy_matcher = fuzzy_matcher
        self.rec_pred = rec_pred

    # ------------------------------------------------------------------
    def parse(self, frame_crop: np.ndarray, dt_boxes: list, rec_res: list) -> Optional[CvParseResult]:
        t0 = time.perf_counter()
        # OCR 線程在 det 之前已經算過（用來決定方向）就直接沿用，見 ticket_cv_layout.hand_off
        layout = take_over(frame_crop)
        reused = layout is not None
        if layout is None:
            layout = analyze(frame_crop)
        t_layout = 1000 * (time.perf_counter() - t0)
        layout_txt = "沿用 OCR 線程的版面" if reused else f"版面 {t_layout:.0f} ms"
        if not layout.ok:
            logger.info(f"[CV輔助] 找不到表格（{layout.reason}，{layout_txt}），改用原本的解析")
            return None

        cells, det_order = self._assign_det_boxes(layout, dt_boxes, rec_res)
        rows_dbg: List[dict] = []
        items: List[Tuple[str, int]] = []
        rescued: List[str] = []
        cancelled = False
        for n, r in enumerate(layout.rows, 1):
            boxes = sorted(cells.get((r.k, "name"), []), key=lambda b: b.x0)
            if r.kind == "header":
                text = "".join(b.text for b in boxes)
                if not text and r.name_quad is not None:   # 只為了認出「取消」
                    text, _ = self._read_cv(layout, r.name_quad)
                if _SECTION_RE.search(text):
                    cancelled = bool(_CANCEL_RE.search(text))
                    if cancelled:
                        logger.info(f"[CV輔助] 第{n}列「{text}」：以下品項在取消段落")
                rows_dbg.append({"raw": text, "name": None, "score": 0.0, "qty": None, "qty_src": "",
                                 # 除錯總覽圖對不認得的 status 直接印出文字（「× 段落標題」）
                                 "status": "段落標題", "box_idx": [b.idx for b in boxes]})
                continue

            name, match, score, name_src, lead_qty = self._read_name(layout, r, boxes, n, rescued)
            qty, qty_src = self._read_qty(layout, r, cells.get((r.k, "qty"), []), lead_qty, n, rescued)
            status = "ok" if (match is not None and score > _MIN_MATCH_SCORE) else (
                "no_match" if match is None else "low_score")
            src_txt = qty_src if name_src == "det" else f"{qty_src}；品名{name_src}"
            rows_dbg.append({"raw": name, "name": match, "score": float(score), "qty": qty,
                             "qty_src": src_txt + ("；取消段落" if cancelled else ""),
                             "status": status, "box_idx": [b.idx for b in boxes]})
            if status != "ok":
                logger.info(f"[CV輔助] 第{n}列品項沒對到菜單：「{name}」（{match} {score:.0f}）")
                continue
            if cancelled:
                logger.warning(f"[CV輔助] 第{n}列 {match}×{qty} 在取消段落（目前仍照常核對）")
            logger.info(f"[訂單餐點確認]: OCR合併結果 -> '{name}' 對應 -> '{match}' (score={score:.1f}) ... "
                        f"數量: {qty} (來源={qty_src}{'' if name_src == 'det' else '，品名' + name_src})")
            items.append((match, qty))

        order_number = det_order
        if order_number is None and layout.order_no_box is not None:
            x0, y0, x1, y1 = layout.order_no_box
            text, conf = self._read_cv(layout, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32))
            from logic.yoshinoya.order_parser import normalize_order_number
            order_number = normalize_order_number(text)
            if order_number:
                rescued.append("單號")
                logger.info(f"[CV補救] 單號：det 沒讀到，CV 框讀到「{text}」→ {order_number}")
        if order_number is None:
            logger.warning("[CV輔助] 單號未確認（det 與 CV 框都沒讀到）")

        n_items = sum(1 for r in layout.rows if r.kind == "item")
        logger.info(
            f"[CV輔助] 表格 {len(layout.rows)} 列（品項 {n_items}、標題 {len(layout.rows) - n_items}），"
            f"CV 補救 {len(rescued)} 格{('：' + '、'.join(rescued)) if rescued else ''}；"
            f"{layout_txt}、解析合計 {1000 * (time.perf_counter() - t0):.0f} ms"
        )
        logger.info(f"[訂單解析結果]: {items}, 編號: {order_number}")
        return CvParseResult(items=items, order_number=order_number, debug_rows=rows_dbg,
                             menu_bottom_line=self._bottom_line(layout), layout=layout, rescued=rescued)

    # ------------------------------------------------------------------
    def _assign_det_boxes(self, layout: TicketLayout, dt_boxes, rec_res):
        """det 框換到 CV 影像座標，依中心點分到 (列, 欄)；單號另外挑。"""
        M = np.asarray(layout.matrix, dtype=np.float64)
        ob, th = layout.order_no_box, layout.text_h
        cells: dict = {}
        orders = []
        from logic.yoshinoya.order_parser import normalize_order_number
        for idx, (box, (text, conf)) in enumerate(zip(dt_boxes, rec_res)):
            if conf < (_MIN_DET_SCORE_NEGI if text.startswith("ねぎ") else _MIN_DET_SCORE):
                continue
            p = np.asarray(box, dtype=np.float64).reshape(-1, 2) @ M[:, :2].T + M[:, 2]
            cx, cy = p[:, 0].mean(), p[:, 1].mean()
            no = normalize_order_number(text.strip())
            if no and ob is not None and ob[0] - 2 * th < cx < ob[2] + 2 * th and ob[1] - th < cy < ob[3] + th:
                orders.append((float(conf), no))
                continue
            cell = layout.cell_of(cx, cy)
            if cell is not None:
                # 跨進数欄：左緣在数欄右界（該高度的位置，欄線可能是斜的）左邊
                cells.setdefault(cell, []).append(_DetBox(idx, float(p[:, 0].min()), float(p[:, 0].max()),
                                                          text.strip(), float(conf),
                                                          p[:, 0].min() < float(layout.col_x(1, cy)) - 2))
        return cells, (max(orders)[1] if orders else None)

    def _resolve(self, text: str):
        if not text:
            return None, 0.0
        match, score = self.folded.resolve(text, self.menus_ticket, self.fuzzy_matcher)
        return match, float(score or 0.0)

    def _read_cv(self, layout: TicketLayout, quad) -> Tuple[str, float]:
        q = pad_quad(quad, layout.text_h, layout.rec_image.shape)
        return self.rec_pred.predict_text(layout.rec_image, q)

    def _read_name(self, layout, r, boxes: List[_DetBox], n: int, rescued: List[str]):
        """回傳 (文字, 菜名, 分數, 來源, 黏在品名前面的數量文字)。"""
        lead_qty = None
        parts = []
        for b in boxes:
            t = b.text
            if b.cross_qty:  # 「1N豚丼・並盛」：前面的數字其實是数欄
                m = _LEAD_QTY_RE.match(t)
                if m:
                    lead_qty = (m.group(1), b.conf)
                    t = t[m.end():]
            parts.append(t)
        det_text = "".join(parts)
        det_match, det_score = self._resolve(det_text)

        reason = None
        if not boxes:
            reason = "det 漏框"
        elif r.name_box is not None:
            ink_w = r.name_box[2] - r.name_box[0]
            cover = (max(b.x1 for b in boxes) - min(b.x0 for b in boxes)) / max(1.0, ink_w)
            if cover < _PARTIAL_COVER:
                reason = f"det 只框到 {cover:.0%}"
        if reason is None and det_score < 100:
            reason = f"det 比對分數 {det_score:.0f}"
        if reason is None or r.name_quad is None:
            return det_text, det_match, det_score, "det", lead_qty

        cv_text, cv_conf = self._read_cv(layout, r.name_quad)
        cv_match, cv_score = self._resolve(cv_text)
        if cv_score > det_score:
            rescued.append(f"第{n}列品名")
            logger.info(f"[CV補救] 第{n}列品名：{reason}，CV 框讀到「{cv_text}」→ {cv_match} ({cv_score:.0f})"
                        f"{'' if not boxes else f'；det 為「{det_text}」→ {det_match} ({det_score:.0f})'}")
            return cv_text, cv_match, cv_score, "CV補救", lead_qty
        if reason != f"det 比對分數 {det_score:.0f}":
            logger.info(f"[CV輔助] 第{n}列品名：{reason}，CV 框讀到「{cv_text}」({cv_score:.0f}) 沒比較好，沿用 det")
        return det_text, det_match, det_score, "det", lead_qty

    def _read_qty(self, layout, r, boxes: List[_DetBox], lead_qty, n: int, rescued: List[str]):
        """回傳 (數量, 來源)。"""
        cands = []  # (數量, 信心, 來源)
        if lead_qty:
            cands.append((int(lead_qty[0]), lead_qty[1], "det(黏在品名前)"))
        for b in boxes:
            m = _DIGITS_RE.search(b.text)
            if m:
                cands.append((int(m.group()), b.conf, "det"))
        det_vals = {v for v, _, _ in cands}
        if r.qty_box is not None:
            x0, y0, x1, y1 = r.qty_box
            side = 1.15 * max(layout.text_h, y1 - y0)
            # predict_quantity 內部會把 cx 往右挪 10 px（配合方法三從表頭「数」推算的位置），這裡先扣回來
            t, c = self.rec_pred.predict_quantity(layout.rec_image, cx=(x0 + x1) / 2 - 10, cy=(y0 + y1) / 2,
                                                  h=side, w=side, is_flip=False)
            m = _DIGITS_RE.search(t or "")
            if m:
                cands.append((int(m.group()), float(c), "CV"))

        if not cands:
            if r.has_qty:
                rescued.append(f"第{n}列數量")
                logger.info(f"[CV補救] 第{n}列數量：det 與 CV 框都讀不出數字，但数欄有字 → 1")
                return 1, "預設1(数欄有字)"
            return 1, "無，預設1"
        if len({v for v, _, _ in cands}) == 1:
            if not det_vals:
                rescued.append(f"第{n}列數量")
                logger.info(f"[CV補救] 第{n}列數量：det 沒讀到，CV 框讀到 {cands[0][0]}")
                return cands[0][0], "CV補救"
            return cands[0][0], cands[0][2]
        v, c, src = max(cands, key=lambda x: x[1])
        logger.warning(f"[CV輔助] 第{n}列數量讀法不一致 {[(x[0], round(x[1], 2), x[2]) for x in cands]}，"
                       f"取信心最高的 {v}（{src}）")
        if src == "CV":
            rescued.append(f"第{n}列數量")
            return v, "CV補救(分歧)"
        return v, f"{src}(分歧)"

    @staticmethod
    def _bottom_line(layout: TicketLayout):
        """表格底線換回 crop 座標，給畫面 overlay（跟 legacy 的 pop_menu_bottom_line 同格式）。"""
        if layout.grid is None:
            return None
        left, right = layout.cols[0], layout.cols[3]
        Minv = np.linalg.inv(np.vstack([layout.matrix, [0, 0, 1]]))
        pts = []
        for x in (left, right):
            p = Minv @ np.array([x, layout.grid.bottom(x), 1.0])
            pts.append((float(p[0]), float(p[1])))
        return tuple(pts)
