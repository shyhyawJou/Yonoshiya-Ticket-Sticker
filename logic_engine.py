from __future__ import annotations

from typing import Dict, List, Optional
import re
import time
from collections import Counter

import numpy as np
from loguru import logger

from schema import Detection
from config import Config
from mqtt_bus import MqttBus
from ocr_engine import StandaloneRecDLA
from mmr_engine import Rotated_RTMDET
from csv_writer import CsvWriter

# === Step 1 重構：幾何運算與文字比對 ===
from logic.geometry import PolygonXYXY
from logic.text_utils import normalize_text, FuzzyMatcher

# === Step 2 重構：訂單解析 / 貼紙比對 ===
from logic.order_parser import OrderParser
from logic.sticker_matcher import StickerMatcher

# === Step 3 重構：tray 資料結構 / 追蹤 / 狀態機 ===
from logic.models import Tray, TrayState, TrackedItem  # noqa: F401  (對外相容：舊呼叫端可能 import 這些名稱)
from logic.image_utils import encode_image_base64
from logic.tray_tracker import TrayTracker
from logic.single_order_tracker import SingleOrderTracker
from logic.tray_state_machine import TrayStateMachine


class LogicEngine:
    """
    薄 Facade：組裝底下幾個各自獨立、各自可測試的元件。

        TrayTracker / SingleOrderTracker
                            - 擁有 trays 字典本身，依 cfg.mode 二選一：
                              tray 模式追蹤多筆 tray 盤位置；
                              single 模式不追蹤 tray 盤，固定只維護一筆訂單。
        TrayStateMachine    - 狀態轉換、OCR task 生成、OCR 結果套用（易變層）
                              兩種模式共用同一份，因為它只操作外部傳入的
                              trays 字典，不在乎字典是誰生出來的。
        OrderParser         - 訂單 OCR 解析（穩定層，可獨立測試）
        StickerMatcher      - 貼紙 OCR 比對（穩定層，可獨立測試）

    LogicEngine 本身只保留：
        - 建構期的 wiring（把設定值分配給正確的元件）
        - update() / apply_ocr_result() 的呼叫順序膠水邏輯
        - tray 消失後的收尾工作（截圖 + CSV log），因為這段需要
          mmr + csv 這兩個屬於「儲存 / 影像」的基礎設施依賴，
          放在 tracker 或 state machine 裡都不合適。
    """

    def __init__(self, cfg: Config, bus: MqttBus, mmr: Rotated_RTMDET, rec_path: str, dict_path: str):
        self.cfg = cfg
        self.bus = bus
        self.mmr = mmr

        #### 2026/06/24 by Chris
        self.menus_ticket = [normalize_text(m, ignore_space=True) for m in cfg.menus_ticket]
        self.menus_sticker = [normalize_text(m, ignore_space=True) for m in cfg.menus_sticker]
        self.menus_with_digits = {
            m for m in self.menus_ticket
            if re.search(r'[1-9]', m)
        }
        self.special_cases: Dict[str, List[str]] = {
            normalize_text(sc.name, ignore_space=True): [
                normalize_text(sub, ignore_space=True) for sub in sc.sub_items
            ]
            for sc in cfg.menus_mapping
        }
        self.rec_pred = StandaloneRecDLA(
            rec_path=rec_path,
            dict_path=dict_path
        )

        # 文字模糊比對工具，OrderParser / StickerMatcher 共用同一份設定
        self.fuzzy_matcher = FuzzyMatcher(
            special_prefixes=["ねぎ"],
            prefix_len=2,
            special_score_threshold=35,
            default_score_threshold=40,
        )

        # 訂單 OCR 解析 / 貼紙比對：穩定層，與 tray 概念解耦
        self.order_parser = OrderParser(
            menus_ticket=self.menus_ticket,
            menus_with_digits=self.menus_with_digits,
            fuzzy_matcher=self.fuzzy_matcher,
            rec_pred=self.rec_pred,
        )
        self.sticker_matcher = StickerMatcher(
            menus_sticker=self.menus_sticker,
            fuzzy_matcher=self.fuzzy_matcher,
        )

        # === 參數讀取設定 ===
        self.N = int(cfg.stability.n_settle_frame)
        self.K = int(cfg.stability.k_container_fail)
        self.K_new = int(cfg.stability.k_container_new)
        self.tray_missing_frame = int(cfg.stability.tray_missing_frame)
        # 貼紙消失多久算「真的離開了」，過期後從 tray.stickers 移除追蹤紀錄
        # （修正：已核對貼紙被拿走又放回，因幽靈 bbox 一直留著而被誤判成第二個新物件）
        # 舊 config 若還沒加這個欄位，退回用 tray_missing_frame 當預設值。
        self.sticker_missing_frame = int(getattr(cfg.stability, "sticker_missing_frame", cfg.stability.tray_missing_frame))
        self.roi_strink = cfg.placement.roi_strink
        self.iou_assign = cfg.placement.iou_assign
        self.iou_candidate = cfg.placement.iou_candidate
        self.drift_iou_thresh = cfg.placement.drift_iou_thresh

        self.ticket_leave_frame = int(cfg.stability.ticket_leave_frame)

        # === mode 切換：決定 trays 字典由哪個 tracker 擁有與維護 ===
        # "tray"   -> TrayTracker：偵測/追蹤 tray 盤位置，可同時存在多筆訂單
        # "single" -> SingleOrderTracker：不追蹤 tray 盤，畫面預設只跑一筆訂單
        # 不管哪一種，對外都是 self.tracker.trays: Dict[str, Tray]，
        # 下面的 TrayStateMachine / apply_ocr_result 等完全不需要因此改動。
        self.mode: str = getattr(cfg, "mode", "tray")

        if self.mode == "single":
            self.tracker = SingleOrderTracker(
                bus=bus,
                sticker_missing_frame=self.sticker_missing_frame,
                tray_missing_frame=self.tray_missing_frame,
                frame_width=cfg.runtime.camera.width,
                frame_height=cfg.runtime.camera.height,
                ticket_leave_frame=self.ticket_leave_frame,
            )
            logger.info("LogicEngine 啟動於 [single] 模式：單一訂單，不追蹤 tray 盤")
        else:
            self.tracker = TrayTracker(
                bus=bus,
                iou_assign=self.iou_assign,
                iou_candidate=self.iou_candidate,
                drift_iou_thresh=self.drift_iou_thresh,
                k_container_fail=self.K,
                k_container_new=self.K_new,
                roi_shrink=self.roi_strink,
                sticker_missing_frame=self.sticker_missing_frame,
            )
            logger.info("LogicEngine 啟動於 [tray] 模式：追蹤多筆 tray 盤")

        # tray 狀態機（易變層）：操作 tracker.trays，但不擁有它
        self.state_machine = TrayStateMachine(
            bus=bus,
            order_parser=self.order_parser,
            sticker_matcher=self.sticker_matcher,
            special_cases=self.special_cases,
            n_settle_frame=self.N,
        )

        self.csv = CsvWriter(log_dir="/mnt/reserved/csv_uploaded")

        self.last_order_number: Dict[str, Optional[str]] = {}

    # ------------------------------------------------------------
    # 對外相容 API：trays 字典實際上由 TrayTracker 持有
    # ------------------------------------------------------------
    @property
    def trays(self) -> Dict[str, Tray]:
        return self.tracker.trays

    # ------------------------------------------------------------
    # tray 生命週期操作，委派給 TrayTracker
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        self.tracker.reset(tray_id)

    def reset_all(self):
        self.tracker.reset_all()

    def reset_tray_states(self, tray_id: str):
        self.tracker.reset_tray_states(tray_id)

    def remove_tray(self, tray_id: str, ts_utc: str) -> bool:
        return self.tracker.remove_tray(tray_id, ts_utc)

    def set_ocr_busy(self, tray_id: str, item_type: str, bbox: PolygonXYXY):
        """由 app.py 確認已將任務送給 OCR 後呼叫，正式鎖定狀態"""
        self.state_machine.set_ocr_busy(self.trays, tray_id, item_type, bbox)

    # ------------------------------------------------------------
    # 主要流程
    # ------------------------------------------------------------
    def update(self, frame: np.ndarray, detections: List[Detection]):
        """
        1. 追蹤 tray盤 (餐盤)
        2. 分配並追蹤盤內的 ticket (訂單) 與 sticker (貼紙)，計算穩定度
        3. 依據狀態機生成 OCR Task
        4. 處理消失過久的 tray（截圖 + CSV log + 移除）
        """
        ts_utc = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        tray_movements: dict = {}

        tray_dets: List[Detection] = []
        ticket_dets: List[Detection] = []
        sticker_dets: List[Detection] = []

        for d in detections:
            if d.cls_name == "tray":
                tray_dets.append(d)
            elif d.cls_name == "order_ticket":
                ticket_dets.append(d)
            elif d.cls_name == "order_sticker":
                sticker_dets.append(d)
            elif d.cls_name in {
                "sesame_front",
                "sesame_back",
                "wafusauce_front",
                "wafusauce_back",
                "cream_front",
                "cream_back",
            }:
                sticker_dets.append(d)

        self.tracker.update_tray_positions(tray_dets, ts_utc)
        self.tracker.update_ticket_and_stickers(ticket_dets, sticker_dets)
        
        # ========================================================
        # 3. 【核心修改：觸發貼紙消失處理】
        # ========================================================
        # 在位置剛更新完、生成任務前，立刻找出有沒有剛消失的貼紙。
        # 它會幫你發送 -1 給前端，把狀態從 COMPLETED 退回 WAITING_STICKERS，
        # 並將貼紙標記為 has_notified_missing = True，讓 tracker 下一幀可以乾淨移除。
        #self.state_machine.handle_missing_items(self.trays)

        tasks = self.state_machine.generate_tasks(self.trays)

        self._finalize_missing_trays(frame, ts_utc)

        return tasks, tray_movements

    def apply_ocr_result(
        self,
        tray_id: str,
        item_type: str,
        frame_crop: np.ndarray,
        dt_boxes: list,
        rec_res: list,
        is_flip: bool,
        task_bbox: PolygonXYXY,
    ):
        """
        1. 處理 OCR 結果
        2. 檢查 tray盤狀態
        3. 發布 mqtt 訊息
        """
        self.state_machine.apply_ocr_result(
            self.trays, tray_id, item_type, frame_crop, dt_boxes, rec_res, is_flip, task_bbox,
            last_order_number=self.last_order_number.get(tray_id),
        )

        # === single 模式專屬修正 ===
        # TrayStateMachine._apply_ticket_result 解析失敗時，只會把
        # stable_frames 歸零、state 打回 WAITING_TICKET，並不會把
        # tray.ticket 設回 None（這在 tray 模式下沒問題，因為前端本來
        # 就不是靠「ticket 被鎖定」這件事拿到觸發訊號）。
        # 但 single 模式的 NEW_TRAY_DETECTED 是在 tray.ticket 從 None
        # 變成有值時才會發送（見 SingleOrderTracker），如果失敗後
        # tray.ticket 還留著舊物件，下次重新穩定就不會被當成「第一次
        # 鎖定」，前端就收不到訊號、卡在等待單據。
        # 這裡把它清成 None，讓下一次偵測穩定時視為重新鎖定，
        # 自然重新觸發 NEW_TRAY_DETECTED。
        if self.mode == "single" and item_type == "ticket":
            tray = self.trays.get(tray_id)
            if tray is not None and tray.ticket is not None and tray.state == TrayState.WAITING_TICKET:
                tray.ticket = None

    # ------------------------------------------------------------
    # tray 消失後的收尾：截圖 + CSV log + 移除
    # 這段用到 mmr（影像裁切）與 csv（落地紀錄），屬於基礎設施依賴，
    # 不放進 TrayTracker 或 TrayStateMachine，留在 Facade 這一層。
    # ------------------------------------------------------------
    def _finalize_missing_trays(self, frame: np.ndarray, ts_utc: str):
        trays_to_remove = []
        for tray_id, tray in self.trays.items():
            if tray.missing_count > self.tray_missing_frame:
                trays_to_remove.append(tray_id)

                # 記住這筆訂單的編號，供下次重新鎖定時比對是否為同一張舊單
                if tray.order_number:
                    self.last_order_number[tray_id] = tray.order_number
                    
                final_tray_capture_b64 = ""
                ticket_capture_b64 = ""

                last_xywhr = tray.xywhr
                if last_xywhr is not None:
                    cx, cy, w, h, r = last_xywhr
                    warped_img, _ = self.mmr.crop_by_angle(frame, cx, cy, w, h, r)
                    final_tray_capture_b64 = encode_image_base64(warped_img)

                ticket_capture = tray.ticket_crop
                if ticket_capture is not None:
                    ticket_capture_b64 = encode_image_base64(ticket_capture)

                expected_counts = Counter(tray.expected_items)
                expected_list = [{item: count} for item, count in expected_counts.items()]

                check_counts = Counter(tray.checked_items)
                check_list = [{item: count} for item, count in check_counts.items()]

                log_payload = {
                    "tray_id": tray_id,
                    "order_number": tray.order_number,
                    "start_time": tray.start_time_str,
                    "end_time": ts_utc,
                    "expected_item": expected_list,
                    "expected_item_count": len(expected_list),
                    "checked_item": check_list,
                    "ticket_capture": ticket_capture_b64,
                    "final_tray_capture": final_tray_capture_b64
                }
                self.csv.log(log_payload)

        for tid in trays_to_remove:
            self.remove_tray(tid, ts_utc)