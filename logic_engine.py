from __future__ import annotations

from typing import Dict, List, Optional
import re
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2
from loguru import logger

from utils.mqtt_bus import MqttBus
from utils.schema import Detection
from utils.csv_writer import CsvWriter
from utils.perf_stats import LOOP_PROFILER
from utils.debug_sheet import DebugContext, DebugSheet

from config import Config, OrderGuardCfg
from ocr_engine import StandaloneRecDLA
from mmr_engine import Rotated_RTMDET

# === Step 1 重構：幾何運算與文字比對 ===
from logic.geometry import PolygonXYXY
from logic.text_utils import normalize_text, FuzzyMatcher

# === Step 2 重構：訂單解析 / 貼紙比對 ===
from logic.yoshinoya.order_parser import OrderParser
from logic.readers.sticker_matcher import StickerMatcher

# === Step 3 重構：tray 資料結構 / 追蹤 / 狀態機 ===
from logic.models import Tray, TrayState, TrackedItem  # noqa: F401  (對外相容：舊呼叫端可能 import 這些名稱)
from logic.image_utils import encode_image_base64
from logic.tray_tracker import TrayTracker
from logic.single_order_tracker import SingleOrderTracker
from logic.preset_roi_tracker import PresetRoiTracker
from logic.tray_state_machine import TrayStateMachine

# [訂單開始] / [訂單結束] 的「第 N 筆」：當天的流水號，日期一換（跟 log 每天換檔同一個時間點）
# 就從 1 重來，每天的 log 檔都從第 1 筆開始。號碼在訂單開始時決定並記在 Tray 上，跨過半夜的
# 那筆，結束那行仍印開始時的號碼，兩行照樣配得起來。放在模組層級，切換模式重建 LogicEngine
# 時號碼接續、不會重來。
# AI box 斷電後時鐘會先回到舊日期、等 chrony 校時再跳回來，那天號碼會多重來一次；
# 這個號碼只用在 log，CSV / MQTT 都不用，接受。
_order_seq = {"date": None, "n": 0}


def _next_order_seq() -> int:
    today = time.strftime("%Y%m%d")
    if today != _order_seq["date"]:
        _order_seq["date"], _order_seq["n"] = today, 0
    _order_seq["n"] += 1
    return _order_seq["n"]


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

    def __init__(self, cfg: Config, bus: MqttBus, mmr: Rotated_RTMDET, rec_path: str, dict_path: str, on_recording_start=None, on_recording_stop=None, get_video_filename=None,
                 save_frame=None):
        self.cfg = cfg
        self.bus = bus
        self.mmr = mmr

        self.on_recording_start = on_recording_start or (lambda reason="": None)
        self.on_recording_stop = on_recording_stop or (lambda reason="": None)
        self.get_video_filename = get_video_filename or (lambda session_id: "")
        # save_frame(tag)：請 App 存一張當下的原始全幀。醬料包不走 OCR、沒有「送出 OCR」那一幀，
        # 所以在它下 OK/NG 結論時由狀態機呼叫（None = 不存）。
        self.save_frame = save_frame

        #### 2026/06/24 by Chris
        self.menus_ticket = [normalize_text(m, ignore_space=True) for m in cfg.menus_ticket]
        self.menus_sticker = [normalize_text(m, ignore_space=True) for m in cfg.menus_sticker]
        # 部署詞彙（config includes 的 vocab_ticket / vocab_sticker）。
        # 有設就用它當模糊比對候選；沒設則退回 menus_*（舊行為）。
        self.vocab_ticket = [normalize_text(m, ignore_space=True)
                             for m in getattr(cfg, "vocab_ticket", [])]
        self.vocab_sticker = [normalize_text(m, ignore_space=True)
                              for m in getattr(cfg, "vocab_sticker", [])]
        self.menus_with_digits = {
            m for m in self.menus_ticket
            if re.search(r'[1-9]', m)
        }
        # 傳票名稱 -> 多個「貼紙組合」變體，由 config.load_config 建好
        # （代表菜名已折疊、同名多列已保留成多個變體）。
        #
        # 舊版在這裡用 dict comprehension 從 cfg.menus_mapping 現建，同名後者
        # 覆蓋前者，導致 menus_mapping.yaml 裡 28 組變體只有最後一列生效。
        # 保留 fallback 是為了讓舊的 config 物件（或測試替身）也還能跑。
        self.menu_variants: Dict[str, List[List[str]]] = getattr(cfg, "menu_variants", None) or {
            normalize_text(sc.name, ignore_space=True): [
                [normalize_text(sub, ignore_space=True) for sub in sc.sub_items]
            ]
            for sc in cfg.menus_mapping
        }
        # 菜名等價層（同菜異寫）。舊 config 沒有這兩個索引時為 None，
        # 比對會自動退回「只用代表菜名清單」的舊行為。
        self.ticket_index = getattr(cfg, "ticket_index", None)
        self.sticker_index = getattr(cfg, "sticker_index", None)
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
        # 傳票解析器依品牌（ticket_parser）：吉野家 legacy 是預設，Suzumo 的 table 才載入 szm 那份
        parser_class = OrderParser
        parser_options = {"menu_index": self.ticket_index}
        if cfg.ticket_parser == "table":
            from logic.szm.order_parser import OrderParser as TableParser
            parser_class = TableParser
            parser_options = {"vocab_ticket": self.vocab_ticket or None}
        self.order_parser = parser_class(
            menus_ticket=self.menus_ticket,
            menus_with_digits=self.menus_with_digits,
            fuzzy_matcher=self.fuzzy_matcher,
            rec_pred=self.rec_pred,
            **parser_options,
        )
        self.sticker_matcher = StickerMatcher(
            menus_sticker=self.menus_sticker,
            fuzzy_matcher=self.fuzzy_matcher,
            vocab_sticker=self.vocab_sticker or None,
            menu_index=self.sticker_index,
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

        # 訂單防護：超時收尾 + 傳票數量異常重讀。舊 config 物件 / 測試替身沒有
        # 這個屬性時用預設值（超時 600 秒、單品上限 20）。
        self.order_guard = getattr(cfg, "order_guard", None) or OrderGuardCfg()
        self.order_timeout_sec = float(self.order_guard.timeout_sec or 0)
        logger.info(
            f"[order_guard] 訂單超時={self.order_timeout_sec:g}s（0=關閉）, "
            f"傳票單品數量上限={self.order_guard.max_item_qty}（0=關閉）"
        )
        self.still_shift_ratio = float(getattr(cfg.stability, "still_shift_ratio", 0.0) or 0.0)
        if self.still_shift_ratio > 0:
            logger.info(f"[still] 靜止判定啟用：相鄰兩幀位移超過框短邊 {self.still_shift_ratio:.1%} 就重新累積穩定幀數")

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
                ticket_exclude_roi_polygon=cfg.preset_roi.ticket_roi.to_polygon_xyxy(),
                finalize_mode="ticket_leave",
                still_shift_ratio=self.still_shift_ratio,
            )
            logger.info("LogicEngine 啟動於 [single] 模式：單一訂單，不追蹤 tray 盤")
        elif self.mode == "preset_roi":
            self.tracker = PresetRoiTracker(
                bus=bus,
                sticker_missing_frame=self.sticker_missing_frame,
                tray_missing_frame=self.tray_missing_frame,
                frame_width=cfg.runtime.camera.width,
                frame_height=cfg.runtime.camera.height,
                ticket_leave_frame=self.ticket_leave_frame,
                ticket_roi_polygon=cfg.preset_roi.ticket_roi.to_polygon_xyxy(),
                finalize_mode="ticket_mat", # "ticket_leave" | "ticket_mat"
                still_shift_ratio=self.still_shift_ratio,
            )
            logger.info("LogicEngine 啟動於 [preset_roi] 模式：單一訂單，依 ticket 是否在預設 ROI 內判斷生命週期")
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
                still_shift_ratio=self.still_shift_ratio,
            )
            logger.info("LogicEngine 啟動於 [tray] 模式：追蹤多筆 tray 盤")

        self.tracker.STABLE_CONFIRM_CLASSES = set(cfg.object_class_items)
        self.tracker.pending_buffer.item_resolver = cfg.object_class_items.get

        # 除錯總覽圖：一筆訂單收尾時把傳票 / 貼紙 / 醬料包的裁切圖、det 框、rec 文字、
        # 模糊比對與 mapping 結果排成一張圖（見 utils/debug_sheet.py）。只讀不寫，
        # 舊 config / 測試替身沒有 debug_sheet 這一段時等於停用。
        self.debug_sheet = DebugSheet(
            getattr(cfg, "debug_sheet", None) or SimpleNamespace(enabled=False),
            root_dir=getattr(getattr(cfg, "csv_log", None), "root_dir", "/mnt/reserved/record/stream"),
            project_dir=Path(__file__).resolve().parent,
            context=DebugContext(
                menu_variants=self.menu_variants,
                object_class_items=dict(getattr(cfg, "object_class_items", None) or {}),
                profile=(f"{getattr(cfg, 'profile_name', '?')} / parser={getattr(cfg, 'ticket_parser', '?')}"
                         f" / mode={self.mode}"),
            ),
            crop_fn=lambda frame, xywhr: self.mmr.crop_by_angle(frame, *xywhr)[0],
        )

        # tray 狀態機（易變層）：操作 tracker.trays，但不擁有它
        self.state_machine = TrayStateMachine(
            object_class_items=cfg.object_class_items,
            bus=bus,
            order_parser=self.order_parser,
            sticker_matcher=self.sticker_matcher,
            menu_variants=self.menu_variants,
            menu_distinct=getattr(cfg, "menu_distinct", None),
            # 傳票修飾列（換湯…）。舊 config 或測試替身沒有這個屬性時為 None，
            # 狀態機會把所有相關流程當 no-op。
            modifier_knowledge=getattr(cfg, "modifier_knowledge", None),
            n_settle_frame=self.N,
            on_recording_start=self.on_recording_start,
            max_item_qty=self.order_guard.max_item_qty,
            wrong_item_confirm=int(getattr(cfg.stability, "wrong_item_confirm", 0) or 0),
            wrong_item_strong_score=float(getattr(cfg.stability, "wrong_item_strong_score", 0) or 0),
            wrong_item_strong_conf=float(getattr(cfg.stability, "wrong_item_strong_conf", 0.9) or 0),
            debug_sheet=self.debug_sheet,
            save_frame=self.save_frame,
        )

        # CSV log 跟影片放在同一個日期資料夾底下 (<root>/<日期>/csv_logs/)，
        # 以後要對照某一天的影片就不用再跨兩個目錄找。
        self.csv = CsvWriter(
            root_dir=cfg.csv_log.root_dir,
            folder=cfg.csv_log.folder,
            max_video_size=cfg.csv_log.max_video_size,
            video_category=cfg.csv_log.video_category,
        )

    # ------------------------------------------------------------
    # 對外相容 API：trays 字典實際上由 TrayTracker 持有
    # ------------------------------------------------------------
    @property
    def trays(self) -> Dict[str, Tray]:
        return self.tracker.trays

    # ============================================================
    # 訂單生命週期：「結束 / 重置一筆訂單」的入口全部在這一區
    # ============================================================
    # 訂單只有一個開始點（傳票確認 -> TrayStateMachine 發 TICKET_READY、開錄影），
    # 結束 / 重置則有下面幾種來源。新增來源時請接到這裡的方法，不要在別處
    # 自己清 tray 或停錄影。
    #
    # * 正常收尾：護貝板重新露出（preset_roi）、傳票離開（single）、tray 消失（tray）
    #     update() -> _finalize_missing_trays -> _close_order
    #     有訂單內容才寫 CSV -> remove_tray（停錄影 + tracker 重置）
    # * 訂單超時：order_guard.timeout_sec
    #     同上，最後關上接單閘門（ROI 內傳票消失 ticket_leave_frame 幀才重新接單）
    # * App 重置：cmd/reset type a / b / all -> reset / reset_tray_states / reset_all
    #     不寫 CSV（作業員按 reset 就是要重來）-> tracker 重置 + 停錄影，接單閘門打開
    # * App 切換模式：cmd/no_tray_setting -> TaskContext.switch_mode -> shutdown()
    #     不寫 CSV、不發事件，只停錄影；新的 LogicEngine 從空白開始
    # * 錄影 watchdog（app.py，timeout_sec + 30 秒）只是保險，只停影片、不碰訂單
    #
    # tracker 重置一律換全新的 Tray（不逐欄清），對 App 的事件也由 tracker 發：
    #   single / preset_roi：每次重置發 TRAY_RESET + WAITING_TICKET，收尾再加 TRAY_REMOVED
    #   tray：reset a / all 刪掉 tray、reset b 換新 Tray，都不發事件；收尾刪 tray 並發 TRAY_REMOVED
    #
    # log 的訂單邊界也在這一區：tray 第一次看到傳票時印 [訂單開始]（_mark_order_starts），
    # 上面每一種結束 / 重置都印 [訂單結束]（_log_order_end），兩行之間就是這筆訂單的全部 log。
    # ------------------------------------------------------------
    def reset(self, tray_id: str):
        self.debug_sheet.discard(tray_id)     # App 重置 = 這筆訂單作廢，不出總覽圖
        # single 模式只有一筆訂單，a / b / all 結果相同（見 SingleOrderTracker.reset）
        self._end_open_orders(f"App reset a ({tray_id})", None if self.mode != "tray" else [tray_id])
        self.tracker.reset(tray_id)
        self.on_recording_stop(reason=f"reset:{tray_id}")

    def reset_all(self):
        self.debug_sheet.discard()
        self._end_open_orders("App reset all")
        self.tracker.reset_all()
        self.on_recording_stop(reason="reset_all")

    def reset_tray_states(self, tray_id: str):
        self.debug_sheet.discard(tray_id)
        self._end_open_orders(f"App reset b ({tray_id})", None if self.mode != "tray" else [tray_id])
        self.tracker.reset_tray_states(tray_id)
        self.on_recording_stop(reason=f"reset_tray_states:{tray_id}")

    def remove_tray(self, tray_id: str, ts_utc: str) -> bool:
        self.debug_sheet.discard(tray_id)     # 正常收尾時 finalize 已經取走證據，這裡只清空殼 / 異常殘留
        self.on_recording_stop(reason=f"tray_removed:{tray_id}")
        return self.tracker.remove_tray(tray_id, ts_utc)

    def shutdown(self, reason: str) -> None:
        """
        這個 LogicEngine 要被整個丟掉之前呼叫（目前只有切換模式）。

        進行中的訂單不寫 CSV、不發事件（新的 LogicEngine 會從空白開始，App 那邊
        本來就會重新收到狀態），只把事件錄影停掉 —— 不停的話，新的 LogicEngine
        不知道有這支影片，會一路錄到 app.py 的 watchdog 才停。
        """
        self._end_open_orders(f"切換模式 ({reason})")
        self.on_recording_stop(reason=reason)
        self.debug_sheet.discard()
        self.debug_sheet.shutdown()   # 佇列裡已經收尾的訂單要把圖寫完

    def _mark_order_starts(self) -> None:
        """[訂單開始]：tray 第一次出現傳票（ticket_mat 模式是護貝板被遮住且有傳票）。

        刻意不等 TICKET_READY：傳票 OCR、解析、讀失敗重讀都屬於這筆訂單，要落在
        開始與結束之間。沒讀出內容就結束的（手擦過、傳票只是經過）會在結束那行
        標成「無內容」。
        """
        for tray_id, tray in self.trays.items():
            if tray.order_seq is None and (tray.ticket is not None or tray.order_session_active):
                tray.order_seq = _next_order_seq()
                tray.order_begin_mono = time.monotonic()
                tray.perf_start = LOOP_PROFILER.snapshot()
                logger.info(f"{'=' * 12} [訂單開始] 第 {tray.order_seq} 筆 tray={tray_id} {'=' * 12}")

    def _log_order_end(self, tray_id: str, tray, reason: str, result: str) -> None:
        """[訂單結束]，只給印過 [訂單開始] 的 tray（兩行永遠成對）"""
        if tray.order_seq is None:
            return
        elapsed = time.monotonic() - tray.order_begin_mono if tray.order_begin_mono else 0.0
        perf = LOOP_PROFILER.since(tray.perf_start)
        logger.info(
            f"{'=' * 12} [訂單結束] 第 {tray.order_seq} 筆 tray={tray_id} 單號={tray.order_number} "
            f"原因={reason} 結果={result} 歷時 {elapsed:.1f}s" + (f"｜{perf}" if perf else "")
            + f" {'=' * 12}"
        )
        tray.order_seq = None   # 同一個 Tray 物件不會再印第二次結束

    def _end_open_orders(self, reason: str, tray_ids=None) -> None:
        """reset / 切換模式：進行中的訂單不寫 CSV，直接收掉"""
        for tray_id, tray in list(self.trays.items()):
            if tray_ids is None or tray_id in tray_ids:
                self._log_order_end(tray_id, tray, reason, "中止（未寫 CSV）")

    def refresh_preset_roi(self) -> None:
        """
        cfg.preset_roi 被改掉之後(App 的 cmd/preset_roi,或開機的自動校正),
        把新的多邊形推給 tracker。

        tracker 是常駐物件,建構時就把 ticket_roi 攤平成多邊形存了一份,
        之後每幀比對都用那一份、不會回頭重讀 cfg —— 所以這裡一定要主動同步,
        不然設定檔改了、畫面上的框也動了,判斷用的卻還是舊座標。

        只換座標、不重建 LogicEngine,是為了讓現場調 ROI 不會順手把進行中的
        訂單狀態一起清掉(重建就等於 reset)。
        """
        preset_roi = getattr(self.cfg, "preset_roi", None)
        if preset_roi is None:
            return

        polygon = preset_roi.ticket_roi.to_polygon_xyxy()

        # preset_roi 模式看「在 ROI 內」;single 模式反過來看「在 ROI 外」,
        # 兩者存的欄位名稱不同,但都是同一個 ticket_roi。
        if hasattr(self.tracker, "ticket_roi_polygon"):
            self.tracker.ticket_roi_polygon = polygon
        if getattr(self.tracker, "ticket_exclude_roi_polygon", None) is not None:
            self.tracker.ticket_exclude_roi_polygon = polygon

        logger.info(f"[ROI] tracker 的 ticket_roi 已更新為 {polygon}")

    def set_ocr_busy(self, tray_id: str, item_type: str, bbox: PolygonXYXY, target=None):
        """由 app.py 確認已將任務送給 OCR 後呼叫，正式鎖定狀態。target = task["target"]"""
        self.state_machine.set_ocr_busy(self.trays, tray_id, item_type, bbox, target)

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
        self.debug_sheet.set_frame(frame)     # 醬料包裁切圖用；只留參考，update() 結束就放掉

        tray_dets: List[Detection] = []
        ticket_dets: List[Detection] = []
        ticket_mat_dets: List[Detection] = []
        sticker_dets: List[Detection] = []

        for d in detections:
            if d.cls_name == "tray":
                tray_dets.append(d)
            elif d.cls_name == "order_ticket":
                ticket_dets.append(d)
            elif d.cls_name == "order_sticker":
                sticker_dets.append(d)
                # logger.info(f"[logic engine] sticker det: {d}")
            elif d.cls_name in self.cfg.object_class_items:
                sticker_dets.append(d)
                # pass
            elif d.cls_name == "ticket_mat":
                ticket_mat_dets.append(d)

        self.tracker.update_tray_positions(tray_dets, ts_utc)
        self.tracker.update_ticket_and_stickers(ticket_dets, sticker_dets, ticket_mat_dets)
        self._mark_order_starts()
        
        # ========================================================
        # 3. 【核心修改：觸發貼紙消失處理】
        # ========================================================
        # 在位置剛更新完、生成任務前，立刻找出有沒有剛消失的貼紙。
        # 它會幫你發送 -1 給前端，把狀態從 COMPLETED 退回 WAITING_STICKERS，
        # 並將貼紙標記為 has_notified_missing = True，讓 tracker 下一幀可以乾淨移除。
        self.state_machine.handle_missing_items(self.trays)
        self.state_machine.resolve_known_class_items(self.trays)  # 醬料包:直接比對,不進 OCR
        self.state_machine.resolve_wrong_item_removal(self.trays)
        self.state_machine.resolve_missing_extra_tickets(self.trays)
        self.state_machine.heartbeat_wrong_items(self.trays)
        tasks = self.state_machine.generate_tasks(self.trays) # 一般貼紙:才需要送 OCR

        self._finalize_missing_trays(frame, ts_utc)
        self.debug_sheet.set_frame(None)

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
        target=None,
        ocr_meta: Optional[dict] = None,
    ):
        """
        1. 處理 OCR 結果
        2. 檢查 tray盤狀態
        3. 發布 mqtt 訊息
        target = 送出時 task["target"]（被讀的那個追蹤物件），沒有就退回用 task_bbox 找
        ocr_meta = {"seq": log 的 OCR#n, "xywhr": 送 OCR 時的旋轉框}：只給除錯總覽記錄用，不影響判定
        """
        self.debug_sheet.set_ocr_meta(ocr_meta)
        try:
            self.state_machine.apply_ocr_result(
                self.trays, tray_id, item_type, frame_crop, dt_boxes, rec_res, is_flip, task_bbox,
                target=target,
            )
        finally:
            self.debug_sheet.set_ocr_meta(None)

        # === single 模式專屬修正 ===
        # TicketReader.apply_ticket_result 解析失敗時，只會把
        # stable_frames 歸零、state 打回 WAITING_TICKET，並不會把
        # tray.ticket 設回 None（這在 tray 模式下沒問題，因為前端本來
        # 就不是靠「ticket 被鎖定」這件事拿到觸發訊號）。
        # 但 single 模式的 NEW_TRAY_DETECTED 是在 tray.ticket 從 None
        # 變成有值時才會發送（見 SingleOrderTracker），如果失敗後
        # tray.ticket 還留著舊物件，下次重新穩定就不會被當成「第一次
        # 鎖定」，前端就收不到訊號、卡在等待單據。
        # 這裡把它清成 None，讓下一次偵測穩定時視為重新鎖定，
        # 自然重新觸發 NEW_TRAY_DETECTED。
        if self.mode in ("single", "preset_roi") and item_type == "ticket":
            tray = self.trays.get(tray_id)
            if tray is not None and tray.ticket is not None and tray.state == TrayState.WAITING_TICKET:
                tray.ticket = None

    # ------------------------------------------------------------
    # tray 消失後的收尾：截圖 + CSV log + 移除
    # 這段用到 mmr（影像裁切）與 csv（落地紀錄），屬於基礎設施依賴，
    # 不放進 TrayTracker 或 TrayStateMachine，留在 Facade 這一層。
    # ------------------------------------------------------------
    @staticmethod
    def _has_order_content(tray) -> bool:
        """
        這次收尾到底有沒有一筆「真的存在過」的訂單。

        finalize_mode == "ticket_mat" 只要「mat 被遮住 + ROI 內出現 ticket」
        就會把訂單流程打開，所以手擦過、傳票只是經過、或傳票 OCR 一直沒成功
        的情況，一樣會走到收尾。那種空殼收尾如果照樣寫一列進 CSV，就會得到
        單號空白、品項空白、只有時間軸在跑的垃圾列（而且時間軸是接在上一列
        結束的瞬間，因為 start_time_str 是上一次 reset 時刷新的）。

        只要單號、應有品項、已核對品項、錯誤品項四者全空，就認定這次收尾
        沒有任何訂單內容，不落 CSV。
        """
        # confirmed_counts 跟 checked_items 都看：問的是「曾經有內容」，
        # 而 checked_items 在餐點被收走後會清空，單看它會低估。
        return bool(
            tray.order_number
            or tray.expected_items
            or tray.checked_items
            or tray.confirmed_counts
            or tray.wrong_item_history
        )

    def _is_order_timed_out(self, tray, now_mono: float) -> bool:
        """訂單從 TICKET_READY 起超過 order_timeout_sec 還沒收尾"""
        if self.order_timeout_sec <= 0 or tray.order_started_mono is None:
            return False
        return now_mono - tray.order_started_mono >= self.order_timeout_sec

    def _finalize_missing_trays(self, frame: np.ndarray, ts_utc: str):
        """每幀找出該收尾的訂單：tracker 判定結束（missing_count 超過門檻）或超時。"""
        now_mono = time.monotonic()
        ending = []   # (tray_id, timed_out)；先收集再收尾，收尾會改動 trays 字典
        for tray_id, tray in self.trays.items():
            # 超時收尾：preset_roi 模式唯一的正常出口是護貝板重新露出。作業員一直
            # 沒露出時，這筆訂單永遠不會結束，之後放進同一格的傳票會被當成同一張
            # 主票（IoU 對得上）而不送 OCR，那些訂單就整筆消失在 CSV 裡。
            # 超時就照一般收尾寫 CSV、重置、停錄影，下一張傳票才能開新單。
            timed_out = self._is_order_timed_out(tray, now_mono)
            if timed_out:
                logger.error(
                    f"[order_timeout] tray={tray_id} 單號={tray.order_number} 訂單已進行 "
                    f"{now_mono - tray.order_started_mono:.0f}s（上限 {self.order_timeout_sec:g}s），"
                    f"仍未收尾，強制收尾寫入 CSV。state={tray.state.name} start={tray.start_time_str}"
                )

            if timed_out or tray.missing_count > self.tray_missing_frame:
                ending.append((tray_id, timed_out))

        for tray_id, timed_out in ending:
            self._close_order(tray_id, frame, ts_utc, timed_out)

    def _close_order(self, tray_id: str, frame: np.ndarray, ts_utc: str, timed_out: bool) -> None:
        """一筆訂單正常結束：有內容才寫 CSV -> remove_tray（停錄影 + tracker 重置）-> 超時就關接單閘門"""
        tray = self.trays[tray_id]

        if self._has_order_content(tray):
            row = self._build_csv_row(tray_id, tray, frame, ts_utc)
            self.csv.log(row)
            self._finalize_debug_sheet(tray_id, tray, row, timed_out)
            result = row["final_status"]
            for key, label in (("missing_item", "缺"), ("wrong_item", "錯誤")):
                if row[key]:
                    result += f" {label} {row[key]}"
        else:
            # 空殼收尾：照樣重置 tray，但不寫 CSV
            logger.warning(
                f"[CSV] tray={tray_id} 收尾時沒有任何訂單內容"
                f"(單號/應有/已核對/錯誤品項全空)，不寫入 CSV。"
                f"start={tray.start_time_str} end={ts_utc}"
            )
            result = "無內容（未寫 CSV）"
        self._log_order_end(tray_id, tray, self._finish_reason(timed_out), result)

        self.remove_tray(tray_id, ts_utc)

        # 一定要在 remove_tray 之後：remove_tray 會經過 reset_tray_states，那裡會把閘門打開
        if timed_out:
            self.tracker.hold_tickets_until_clear(tray_id)

    def _finalize_debug_sheet(self, tray_id: str, tray, row: dict, timed_out: bool) -> None:
        """把這筆訂單的總覽圖排進背景渲染佇列。用的是剛寫進 CSV 的同一份 row，兩邊結論一定一致。"""
        if not self.debug_sheet.enabled:
            return
        try:
            achieved = Counter(tray.confirmed_counts) | Counter(tray.checked_items)
            started = tray.order_started_mono
            self.debug_sheet.finalize(tray_id, {
                "status": row["final_status"],
                "tray_id": tray_id,
                "order_number": row["order_number"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration_sec": int(time.monotonic() - started) if started is not None else None,
                "expected": row["expected_item"],
                "achieved": [{k: v} for k, v in achieved.items()],
                "missing": row["missing_item"],
                "wrong": row["wrong_item"],
                "timed_out": bool(timed_out),
                "video_file": row["video_file"],
            })
        except Exception as e:   # 除錯功能不能影響收尾
            logger.warning(f"[debug_sheet] 收尾出圖失敗（略過）: {e!r}")

    def _finish_reason(self, timed_out: bool) -> str:
        if timed_out:
            return "超時"
        if self.mode == "tray":
            return "tray 消失"
        return {"ticket_mat": "護貝板露出", "ticket_leave": "傳票離開"}.get(
            getattr(self.tracker, "finalize_mode", None), "收尾"
        )

    def _build_csv_row(self, tray_id: str, tray, frame: np.ndarray, ts_utc: str) -> dict:
        """把一筆結束的訂單整理成 CSV 的一列（欄位見 utils/csv_writer.py）"""
        # final_tray_capture_b64 = ""
        # ticket_capture_b64 = ""

        # last_xywhr = tray.xywhr
        # if last_xywhr is not None:
        #     cx, cy, w, h, r = last_xywhr
        #     warped_img, _, M = self.mmr.crop_by_angle(frame, cx, cy, w, h, r)
        #     final_tray_capture_b64 = encode_image_base64(warped_img)

        # ticket_capture = tray.ticket_crop
        # if ticket_capture is not None:
        #     ticket_capture_b64 = encode_image_base64(ticket_capture)

        # 收尾要回答的是「這筆訂單曾經核對到哪」，不是「現在畫面上還剩什麼」。
        # checked_items 的定義是「當下仍看得到的已確認實體」，餐點被收走時
        # handle_missing_items() 會刻意把它扣掉；而現場習慣驗完就先收餐，
        # 收尾一定發生在扣完之後 —— 用它記帳就會寫出 final_status=OK 卻
        # 同時列出缺少品項的矛盾紀錄。
        # confirmed_counts 是只增不減的歷史最高確認數（見
        # OrderRequirements.update_confirmed_count），也是 TRAY_COMPLETED /
        # resolve_wrong_item_removal 判斷完成用的同一份依據，CSV 跟著用
        # 同一份，報表才不會跟前端看到的結果打架。
        # 再取跟當下 checked 的逐項最大值（Counter 的 | 是逐項 max），
        # 純粹是保險：任何沒走 update_confirmed_count 的路徑（例如
        # confirmed 被 expected 上限夾住）都不會讓收尾少算。
        achieved_counts = Counter(tray.confirmed_counts) | Counter(tray.checked_items)

        # 多變體訂單（同一個傳票名稱對到多種貼紙組合）不能用聯集記帳，
        # 否則「其實不會來的品項」會被記成 missing。取最能解釋現場的那一版。
        # 這裡餵 achieved_counts 而不是當下的 checked，否則餐點收走後
        # 每個變體的 shortfall 都一樣大，挑出來的「最能解釋現場」那一版
        # 會退化成隨便挑一個，連 expected_item 欄都會記錯。
        expected_counts = self.state_machine.resolve_effective_combo(tray, achieved_counts)
        expected_list = [{item: count} for item, count in expected_counts.items()]

        missing_counts = expected_counts - achieved_counts
        missing_list = [{item: count} for item, count in missing_counts.items()]

        wrong_counts = Counter(tray.wrong_item_history)
        wrong_list = [{item: count} for item, count in wrong_counts.items()]

        final_status = (
            "NGtoOK" if tray.state == TrayState.COMPLETED and tray.ever_had_wrong_item
            else "OK" if tray.state == TrayState.COMPLETED
            else "NG"
        )

        video_file = self.get_video_filename(tray.record_session_id) if tray.record_session_id else ""

        return {
            "tray_id": tray_id,
            "order_number": tray.order_number,
            "start_time": tray.start_time_str,
            "end_time": ts_utc,
            "final_status": final_status,
            "expected_item": expected_list,
            "expected_item_count": len(expected_list),
            "missing_item": missing_list,
            "wrong_item": wrong_list,
            # "ticket_capture": ticket_capture_b64,
            # "final_tray_capture": final_tray_capture_b64,
            "video_file": video_file,
        }