"""
readers/stub.py
===============
客戶聯（stub / small_order_ticket）的 OCR 結果套用與主票交叉驗證。

遵循 readers/ 架構：
- apply_stub_result  : 套用客戶聯 OCR 結果、更新 fallback_rects 與 order_json，若主票已就緒則執行交叉驗證
- verify_single_stub : 當主票與客戶聯皆已讀出時，比對單號與品項並記錄 log（不影響主單狀態與完成判定）
"""

from __future__ import annotations

import json
import time
from typing import List
from loguru import logger

from logic.models import Tray
from logic.readers.base import find_task_item
from logic.yoshinoya.stub_verifier import verify_stub


class StubReader:
    def __init__(self, stub_parser=None, rec_pred=None):
        self.stub_parser = stub_parser
        self.rec_pred = rec_pred

    def apply_stub_result(
        self,
        tray: Tray,
        tray_id: str,
        dt_boxes,
        rec_res,
        is_flip,
        task_bbox,
        target=None,
        frame_crop=None,
    ) -> None:
        stub = tray.stub
        if stub is None:
            return

        target_item = find_task_item([stub.item], task_bbox, target) or stub.item
        target_item.is_ocr_busy = False
        target_item.last_result_mono = time.monotonic()

        if self.stub_parser is None:
            return

        stub.attempts += 1
        stub.ocr_res = list(rec_res) if rec_res else []

        parse = self.stub_parser.parse(
            dt_boxes,
            rec_res,
            img=frame_crop,
            rec_pred=self.rec_pred,
            is_flip=is_flip,
        )
        stub.fallback_rects = list(getattr(parse, "fallback_rects", []) or [])

        if not parse.ok:
            max_att = self.stub_parser.cfg.max_attempts
            if stub.attempts >= max_att:
                target_item.is_checked = True
                logger.warning(
                    f"[stub] tray={tray_id} 客戶聯 {stub.attempts}/{max_att} 次仍未讀出有效內容，放棄重試"
                )
            else:
                target_item.stable_frames = 0
                logger.info(
                    f"[stub] tray={tray_id} 客戶聯未讀出內容（{stub.attempts}/{max_att}），等待重送"
                )
            return

        stub.result = parse
        target_item.is_checked = True
        stub.order_json = parse.to_order_json()

        items_str = (
            "、".join(
                f"{it.name or it.raw_name}({it.pos_code or '-'})×{it.quantity if it.quantity is not None else '?'}"
                for it in parse.items
            )
            or "（無品項）"
        )
        logger.info(
            f"[客戶聯解析結果] tray={tray_id} 單號={parse.ticket_no} "
            f"桌號={parse.table_no} 時間={parse.time_str}｜{items_str}"
        )
        logger.info(
            f"[客戶聯 JSON] tray={tray_id} {json.dumps(stub.order_json, ensure_ascii=False)}"
        )

        self.verify_single_stub(tray)

    def verify_single_stub(self, tray: Tray) -> None:
        stub = tray.stub
        if stub is None or stub.result is None:
            return
        if tray.ticket is None or not tray.ticket.is_checked:
            return

        parse = stub.result
        main_items: List[tuple] = []
        if tray.ticket is not None and tray.ticket.parsed_items:
            main_items.extend(tray.ticket.parsed_items)
        for et in tray.extra_tickets:
            if et.is_ticket_merged and not et.has_notified_missing and et.parsed_items:
                main_items.extend(et.parsed_items)

        v = verify_stub(parse, tray.order_number, main_items)
        stub.verification = v

        if v.matched:
            logger.info(
                f"[stub_verify] tray={tray.id} 客戶聯與主票一致 "
                f"(單號={parse.ticket_no}, 品項={parse.as_pairs()})"
            )
        else:
            logger.warning(
                f"[stub_verify] tray={tray.id} 客戶聯與主票不一致: "
                f"stub單號={parse.ticket_no} vs 主票={tray.order_number} (no_match={v.ticket_no_match}), "
                f"stub少={v.missing_in_stub}, stub多={v.extra_in_stub}, "
                f"數量未確認={v.qty_unconfirmed_in_stub}"
            )
