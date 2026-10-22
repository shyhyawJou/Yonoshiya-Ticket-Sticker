"""
readers/debug_hooks.py
======================
把「傳票 / 貼紙 / 醬料包的判定過程」交給除錯總覽圖（utils/debug_sheet.py）收集。

兩個 reader（readers/ticket.py、readers/sticker.py）和 TrayStateMachine 共用同一份 ReaderDebugHooks：
它們只在判定**做完之後**呼叫一行 `self.debug.xxx(...)`，這裡負責把當下的解析列、比對結果、
待核對清單整理成一包資料交給 DebugSheet。

原則
----
* **只讀不寫**：不會改 tray / 貼紙 / 解析器的任何狀態，所以不會影響判定。
* **任何錯誤都吞掉**（記 warning）：除錯功能壞掉不能讓 OCR 流程跟著壞。
* **停用時近乎零成本**：`on()` 為 False 時所有方法立刻返回。
* **耗時要量**：在主迴圈花掉的時間用 `_timed` 累計到該筆訂單，出圖時印 [debug_sheet][timing]。
"""
from __future__ import annotations

import functools
import time
from collections import Counter
from typing import Callable, List, Optional

from loguru import logger


def _timed(label: str, tray_arg: int):
    """量測被包住的方法在「主迴圈」花的時間，累計到該筆訂單。

    tray_arg = tray_id 在位置參數裡的 index（不含 self）。量測本身出錯不外洩。
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            if not self.on():
                return fn(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return fn(self, *args, **kwargs)
            finally:
                try:
                    self.sheet.add_cost(args[tray_arg], time.perf_counter() - t0, label)
                except Exception:
                    pass
        return wrapper
    return deco


class ReaderDebugHooks:
    def __init__(self, sheet=None, order_parser=None, sticker_matcher=None, requirements=None,
                 confirm_fn: Optional[Callable[[], int]] = None):
        self.sheet = sheet                      # utils.debug_sheet.DebugSheet；None = 停用
        self.order_parser = order_parser
        self.sticker_matcher = sticker_matcher
        self.requirements = requirements        # OrderRequirements：取實際套用的 mapping / 替換
        self.confirm_fn = confirm_fn or (lambda: 0)

    # ------------------------------------------------------------------
    def on(self) -> bool:
        return self.sheet is not None and bool(getattr(self.sheet, "enabled", False))

    # ------------------------------------------------------------------
    # 傳票
    # ------------------------------------------------------------------
    def rows(self) -> List[dict]:
        """取出剛剛那次 parse() 的逐列比對細節；兩種解析器各自的格式在這裡統一成 dict。"""
        parser = self.order_parser
        rows = getattr(parser, "last_debug_rows", None)       # 吉野家 legacy 解析器
        if rows:
            return [dict(r) for r in rows]
        out: List[dict] = []
        for r in getattr(parser, "last_rows", None) or []:    # Suzumo 表格解析器
            if r.name is None:
                status = "no_match"
            elif r.quantity is None:
                status = "qty_unconfirmed"
            else:
                status = "ok"
            out.append({
                "raw": r.name_raw, "name": r.name, "score": float(r.name_score or 0),
                "qty": r.quantity, "qty_src": r.quantity_source, "status": status, "box_idx": [],
            })
        return out

    def ticket_fields(self, tray) -> dict:
        """傳票被接受之後的訂單需求：組合假設、實際套用的 mapping、替換、數量未確認。"""
        req = self.requirements
        return {
            "combos": [dict(c) for c in (getattr(tray, "expected_combos", None) or [])],
            "expected_display": list(getattr(tray, "expected_items_display", None) or []),
            # OrderRequirements 記下的「實際套用」結果（含換湯這類替換），比重算準確
            "mapping_lines": list(getattr(req, "last_mapping_lines", None) or []),
            "substitutions": list(getattr(req, "last_substitutions", None) or []),
            "unconfirmed_qty": list(getattr(req, "last_unconfirmed_quantity", None) or []),
        }

    @_timed("ticket", 0)
    def ticket(self, tray_id, kind, frame_crop, dt_boxes, rec_res, order_number, parsed_items,
               *, accepted: bool, outcome: str, tray):
        if not self.on():
            return None
        try:
            data = {
                "order_number": order_number,
                "rows": self.rows(),
                "parsed": [[n, q] for n, q in (parsed_items or [])],
                "accepted": bool(accepted),
                "outcome": outcome,
            }
            if accepted:
                data.update(self.ticket_fields(tray))
            return self.sheet.record_ocr(tray_id, kind, frame_crop, dt_boxes, rec_res, data)
        except Exception as e:
            logger.warning(f"[debug_sheet] 記錄傳票失敗（略過）: {e!r}")
            return None

    def set(self, ev, tray=None, **fields) -> None:
        """更新先前記下的傳票紀錄（候選傳票先記「處理中」，各出口再補結果）。"""
        if ev is None:
            return
        try:
            ev.data.update(fields)
            if tray is not None and ev.data.get("accepted"):
                ev.data.update(self.ticket_fields(tray))
        except Exception as e:
            logger.warning(f"[debug_sheet] 更新傳票紀錄失敗（略過）: {e!r}")

    # ------------------------------------------------------------------
    # 貼紙 / 醬料包
    # ------------------------------------------------------------------
    def top(self, raw, k: int = 3) -> list:
        """貼紙文字的近似候選 top-k。只是除錯參考（token_set_ratio 重算），不是正式比對結果。"""
        if not raw:
            return []
        try:
            from rapidfuzz import process, fuzz
            sm = self.sticker_matcher
            fmap = sm.folded.folded_map(sm.vocab_sticker or sm.menus_sticker)
            res = process.extract(sm._fold(raw), list(fmap.keys()), scorer=fuzz.token_set_ratio, limit=k)
            return [(fmap[key], float(score)) for key, score, _ in res]
        except Exception:
            return []

    def verdict(self, tray, ts, match_status, has_text: bool):
        """這次貼紙 / 物件判定之後的結論 (verdict, 說明)。要在 settle() 之後呼叫。"""
        confirm = self.confirm_fn()
        if ts is None:
            return "UNTRACKED", "OCR 結果對不到追蹤中的貼紙（已離開或位置漂移）"
        if match_status == "MATCHED":
            item = ts.item_name
            return "OK", (f"核對成功（{item} {Counter(tray.checked_items).get(item, 0)}"
                          f"/{Counter(tray.expected_items).get(item, 0)}）")
        if ts.is_wrong_item:
            return "NG", f"錯誤品項成立: {ts.item_name}"
        if ts.is_unknown_item:
            return "UNKNOWN", f"讀不出菜名: {ts.item_name}"
        if ts.wrong_reads:
            reads = "、".join(f"{text}({kind})" for kind, text in ts.wrong_reads)
            return "SUSPECT", f"確認中 {len(ts.wrong_reads)}/{confirm}: {reads}"
        if not has_text:
            return "UNREAD", "沒讀到文字"
        return "UNREAD", "比對不到菜名（尚未成立錯誤品項）"

    @staticmethod
    def order_context(tray) -> dict:
        return {
            "expected": list(tray.expected_items),
            "checked": list(tray.checked_items),
            "ticket_names": [k for d in (tray.expected_items_display or []) for k in d],
        }

    @_timed("sticker", 1)
    def sticker(self, tray, tray_id, ts, frame_crop, dt_boxes, rec_res, matched_item, match_status) -> None:
        if not self.on():
            return
        try:
            sm = self.sticker_matcher
            raw = conf = None
            if rec_res is not None and len(rec_res) > 0:
                idx = 0   # 跟 StickerMatcher 一樣：取最靠上方的框當主標題
                if dt_boxes is not None and len(dt_boxes) == len(rec_res) and len(dt_boxes) > 1:
                    idx = min(range(len(dt_boxes)), key=lambda i: min(pt[1] for pt in dt_boxes[i]))
                raw, conf = rec_res[idx][0], float(rec_res[idx][1])
            verdict, note = self.verdict(tray, ts, match_status, bool(raw))
            reported = None
            if ts is not None and (ts.is_checked or ts.is_wrong_item or ts.is_unknown_item):
                reported = ts.item_name
            data = {
                "route": "ocr",
                "cls_name": getattr(ts, "cls_name", None),
                "raw": raw, "conf": conf,
                "matched": matched_item, "reported": reported or matched_item,
                "status": match_status,
                "score": float(getattr(sm, "last_match_score", 0) or 0),
                "exact": bool(getattr(sm, "last_match_exact", False)),
                "top": self.top(raw),
                "verdict": verdict, "note": note,
            }
            data.update(self.order_context(tray))
            self.sheet.record_ocr(tray_id, "sticker", frame_crop, dt_boxes, rec_res, data,
                                  track=self.sheet.track_uid(ts) if ts is not None else None)
        except Exception as e:
            logger.warning(f"[debug_sheet] 記錄貼紙失敗（略過）: {e!r}")

    @_timed("known", 1)
    def known(self, tray, tray_id, ts, item_name, matched_item, match_status) -> None:
        """醬料包等類別即身份的物件：沒有 OCR 裁切圖，用當下原圖依旋轉框裁一張。"""
        if not self.on():
            return
        try:
            verdict, note = self.verdict(tray, ts, match_status, True)
            data = {
                "route": "class",
                "cls_name": ts.cls_name, "item_name": item_name,
                "matched": matched_item,
                "reported": ts.item_name if (ts.is_checked or ts.is_wrong_item) else matched_item,
                "status": match_status, "verdict": verdict, "note": note,
            }
            data.update(self.order_context(tray))
            self.sheet.record_known(tray_id, ts.xywhr, data, track=self.sheet.track_uid(ts))
        except Exception as e:
            logger.warning(f"[debug_sheet] 記錄物件失敗（略過）: {e!r}")

    @_timed("event", 0)
    def event(self, tray_id, ts, text: str) -> None:
        if not self.on():
            return
        try:
            self.sheet.annotate(tray_id, self.sheet.track_uid(ts), text)
        except Exception as e:
            logger.warning(f"[debug_sheet] 記錄事件失敗（略過）: {e!r}")


# 沒有注入除錯收集器時 reader 用這一份：所有方法都是 no-op（on() 為 False）
NO_DEBUG = ReaderDebugHooks(None)
