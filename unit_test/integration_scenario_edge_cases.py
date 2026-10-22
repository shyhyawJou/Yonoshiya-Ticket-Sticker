"""
Step 3 額外邊界情境測試：涵蓋主場景沒測到的分支
    1. 訂單 OCR 辨識失敗 -> 應退回 WAITING_TICKET，ticket.stable_frames 歸零
    2. 貼紙比對到不在訂單內的品項 -> WRONG_ITEM_DETECTED
    3. reset_tray_states -> 應該保留 tray 位置，但清空狀態
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib
import numpy as np

from fake_dependencies import (
    Detection, make_xyxy, FakeMqttBus, FakeMmr, FakeConfig,
    FakeStability, FakePlacement,
)


def box(x1, y1, x2, y2):
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def run(module_name: str):
    mod = importlib.import_module(module_name)
    LogicEngine = mod.LogicEngine

    cfg = FakeConfig(
        menus_ticket=["味噌ラーメン", "醤油ラーメン"],
        menus_sticker=["味噌ラーメン", "醤油ラーメン"],
        menus_mapping=[],
        stability=FakeStability(n_settle_frame=2, k_container_fail=2, k_container_new=1, tray_missing_frame=99),
        placement=FakePlacement(roi_strink=0.1, iou_assign=0.3, iou_candidate=0.3, drift_iou_thresh=0.5),
    )
    bus = FakeMqttBus()
    mmr = FakeMmr()
    engine = LogicEngine(cfg, bus, mmr, rec_path=None, dict_path=None)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tray_xyxy = make_xyxy(0, 0, 200, 200)
    tray_xywhr = (100.0, 100.0, 200.0, 200.0, 0.0)
    tray_det = Detection("tray", tray_xyxy, tray_xywhr)

    ticket_xyxy = make_xyxy(10, 10, 60, 40)
    ticket_xywhr = (35.0, 25.0, 50.0, 30.0, 0.0)
    ticket_det = Detection("order_ticket", ticket_xyxy, ticket_xywhr)
    ticket_dt_box = box(10, 10, 60, 40)

    sticker_xyxy = make_xyxy(70, 70, 120, 100)
    sticker_xywhr = (95.0, 85.0, 50.0, 30.0, 0.0)
    sticker_det = Detection("order_sticker", sticker_xyxy, sticker_xywhr)
    sticker_dt_box = box(70, 70, 120, 100)

    events = []

    # 建立新盤（候選邏輯本身至少需要 2 幀：第一幀建立候選，第二幀才會判斷門檻）
    engine.update(frame, [tray_det])
    engine.update(frame, [tray_det])
    tray_id = list(engine.trays.keys())[0]
    events.append(("tray_created", tray_id))

    # --- 情境 1：訂單 OCR 辨識失敗 ---
    engine.update(frame, [tray_det, ticket_det])
    engine.update(frame, [tray_det, ticket_det])
    tasks = engine.update(frame, [tray_det, ticket_det])[0]
    assert len(tasks) == 1 and tasks[0]["type"] == "ticket"
    engine.set_ocr_busy(tray_id, "ticket", ticket_xyxy)

    engine.apply_ocr_result(tray_id, "ticket", frame, [ticket_dt_box], [("完全辨識不到的亂碼文字", 0.9)], False, ticket_xyxy)
    tray = engine.trays[tray_id]
    events.append(("after_failed_ocr", tray.state.name, tray.ticket.stable_frames if tray.ticket else None))
    assert tray.state.name == "WAITING_TICKET"
    assert tray.ticket.stable_frames == 0

    # --- 重新來一次，這次訂單成功，才能進到貼紙比對階段 ---
    engine.update(frame, [tray_det, ticket_det])
    engine.update(frame, [tray_det, ticket_det])
    tasks = engine.update(frame, [tray_det, ticket_det])[0]
    assert len(tasks) == 1 and tasks[0]["type"] == "ticket"
    engine.set_ocr_busy(tray_id, "ticket", ticket_xyxy)
    engine.apply_ocr_result(tray_id, "ticket", frame, [ticket_dt_box], [("味噌ラーメン", 0.9)], False, ticket_xyxy)
    tray = engine.trays[tray_id]
    events.append(("after_success_ocr", tray.state.name, sorted(tray.expected_items)))
    assert tray.state.name == "WAITING_STICKERS"

    # --- 情境 2：貼紙比對到不在訂單內的品項 -> WRONG_ITEM ---
    engine.update(frame, [tray_det, sticker_det])
    engine.update(frame, [tray_det, sticker_det])
    tasks = engine.update(frame, [tray_det, sticker_det])[0]
    assert len(tasks) == 1 and tasks[0]["type"] == "sticker"
    engine.set_ocr_busy(tray_id, "sticker", sticker_xyxy)
    engine.apply_ocr_result(tray_id, "sticker", frame, [sticker_dt_box], [("醤油ラーメン", 0.9)], False, sticker_xyxy)
    events.append(("wrong_item_bus_event", bus.det_status_events[-1]))
    assert bus.det_status_events[-1]["status"] == "WRONG_ITEM_DETECTED"

    # --- 情境 3：reset_tray_states ---
    engine.reset_tray_states(tray_id)
    tray = engine.trays[tray_id]
    events.append(("after_reset_tray_states", tray.state.name, tray.expected_items, tray.checked_items, tray.ticket))
    assert tray.state.name == "WAITING_TICKET"
    assert tray.expected_items == []
    assert tray.ticket is None
    assert tray_id in engine.trays  # 位置應該保留（tray 本身還在字典裡）

    return events


if __name__ == "__main__":
    module_name = sys.argv[1] if len(sys.argv) > 1 else "logic_engine_original"
    events = run(module_name)
    for e in events:
        print(e)
    print(f"\n[{module_name}] 全部邊界情境測試通過 ✅")
