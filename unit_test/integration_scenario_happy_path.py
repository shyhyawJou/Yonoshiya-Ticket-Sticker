"""
Step 3 行為快照測試腳本。

跑一段完整的 tray 生命週期（新盤偵測 -> 訂單穩定 -> OCR -> 貼紙穩定 x2
-> OCR 核對 x2 -> 完成 -> 餐盤移除），把每一步的輸出（tasks / bus 事件 /
csv log / tray 狀態）記錄成一份 trace，讓拆分前後可以逐項比對。

用法：
    python3 run_scenario.py logic_engine_original   # 對照原始檔
    python3 run_scenario.py logic_engine_step2       # 對照 Step2 版本
    python3 run_scenario.py logic_engine              # 對照 Step3 拆分後版本
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import importlib
import numpy as np

from fake_dependencies import (
    Detection, make_xyxy, FakeMqttBus, FakeMmr, FakeCsvWriter, FakeConfig,
    FakeStability, FakePlacement,
)


def box(x1, y1, x2, y2):
    """OCR dt_box 格式：4 個 [x, y] 點"""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def summarize_task(t):
    return {"tray_id": t["tray_id"], "type": t["type"], "bbox": tuple(float(v) for v in t["bbox"])}


def summarize_trays(trays):
    out = {}
    for tid, tray in trays.items():
        out[tid] = {
            "state": tray.state.name,
            "expected_items": sorted(tray.expected_items),
            "checked_items": sorted(tray.checked_items),
            "missing_count": tray.missing_count,
            "num_stickers": len(tray.stickers),
        }
    return out


def run(module_name: str):
    mod = importlib.import_module(module_name)
    LogicEngine = mod.LogicEngine

    cfg = FakeConfig(
        menus_ticket=["味噌ラーメン", "醤油ラーメン"],
        menus_sticker=["味噌ラーメン", "醤油ラーメン"],
        menus_mapping=[],
        stability=FakeStability(n_settle_frame=3, k_container_fail=2, k_container_new=2, tray_missing_frame=3),
        placement=FakePlacement(roi_strink=0.1, iou_assign=0.3, iou_candidate=0.3, drift_iou_thresh=0.5),
    )
    bus = FakeMqttBus()
    mmr = FakeMmr()
    engine = LogicEngine(cfg, bus, mmr, rec_path=None, dict_path=None)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    tray_xyxy = make_xyxy(0, 0, 200, 200)
    tray_xywhr = (100.0, 100.0, 200.0, 200.0, 0.0)

    ticket_xyxy = make_xyxy(10, 10, 60, 40)
    ticket_xywhr = (35.0, 25.0, 50.0, 30.0, 0.0)
    ticket_dt_box = box(10, 10, 60, 40)

    stickerA_xyxy = make_xyxy(70, 70, 120, 100)
    stickerA_xywhr = (95.0, 85.0, 50.0, 30.0, 0.0)
    stickerA_dt_box = box(70, 70, 120, 100)

    stickerB_xyxy = make_xyxy(130, 70, 180, 100)
    stickerB_xywhr = (155.0, 85.0, 50.0, 30.0, 0.0)
    stickerB_dt_box = box(130, 70, 180, 100)

    trace = []

    def step(label, detections):
        tasks, _ = engine.update(frame, detections)
        trace.append({
            "label": label,
            "tasks": [summarize_task(t) for t in tasks],
            "trays": summarize_trays(engine.trays),
        })
        return tasks

    tray_det = Detection("tray", tray_xyxy, tray_xywhr)
    ticket_det = Detection("order_ticket", ticket_xyxy, ticket_xywhr)
    stickerA_det = Detection("order_sticker", stickerA_xyxy, stickerA_xywhr)
    stickerB_det = Detection("order_sticker", stickerB_xyxy, stickerB_xywhr)

    # 1-2. 新盤候選累積 -> 建立
    step("f1_candidate", [tray_det])
    step("f2_new_tray", [tray_det])

    # 3-6. ticket 穩定度累積 -> 產生 OCR task
    step("f3_ticket_seen", [tray_det, ticket_det])
    step("f4_ticket_stable1", [tray_det, ticket_det])
    step("f5_ticket_stable2", [tray_det, ticket_det])
    tasks6 = step("f6_ticket_task", [tray_det, ticket_det])
    assert len(tasks6) == 1 and tasks6[0]["type"] == "ticket"

    tray_id = tasks6[0]["tray_id"]
    engine.set_ocr_busy(tray_id, "ticket", ticket_xyxy)

    engine.apply_ocr_result(
        tray_id, "ticket", frame, [ticket_dt_box], [("2味噌ラーメン", 0.9)], False, ticket_xyxy
    )
    trace.append({"label": "ocr_ticket_applied", "trays": summarize_trays(engine.trays)})

    # 7-10. stickerA 穩定度累積 -> 產生 OCR task
    step("f7_stickerA_seen", [tray_det, stickerA_det])
    step("f8_stickerA_stable1", [tray_det, stickerA_det])
    step("f9_stickerA_stable2", [tray_det, stickerA_det])
    tasks10 = step("f10_stickerA_task", [tray_det, stickerA_det])
    assert len(tasks10) == 1 and tasks10[0]["type"] == "sticker"

    engine.set_ocr_busy(tray_id, "sticker", stickerA_xyxy)
    engine.apply_ocr_result(
        tray_id, "sticker", frame, [stickerA_dt_box], [("味噌ラーメン", 0.9)], False, stickerA_xyxy
    )
    trace.append({"label": "ocr_stickerA_applied", "trays": summarize_trays(engine.trays)})

    # 11-14. stickerB 穩定度累積 -> 產生 OCR task
    step("f11_stickerB_seen", [tray_det, stickerA_det, stickerB_det])
    step("f12_stickerB_stable1", [tray_det, stickerA_det, stickerB_det])
    step("f13_stickerB_stable2", [tray_det, stickerA_det, stickerB_det])
    tasks14 = step("f14_stickerB_task", [tray_det, stickerA_det, stickerB_det])
    assert len(tasks14) == 1 and tasks14[0]["type"] == "sticker"

    engine.set_ocr_busy(tray_id, "sticker", stickerB_xyxy)
    engine.apply_ocr_result(
        tray_id, "sticker", frame, [stickerB_dt_box], [("味噌ラーメン", 0.9)], False, stickerB_xyxy
    )
    trace.append({"label": "ocr_stickerB_applied", "trays": summarize_trays(engine.trays)})

    # 15-18. tray 消失 -> 移除 + CSV log
    step("f15_missing1", [])
    step("f16_missing2", [])
    step("f17_missing3", [])
    step("f18_removed", [])

    result = {
        "trace": trace,
        "system_events": bus.system_events,
        "det_status_events": bus.det_status_events,
        "csv_logs": [
            {k: v for k, v in log.items() if k not in ("ticket_capture", "final_tray_capture")}
            for log in engine.csv.logs
        ],
        "final_trays": summarize_trays(engine.trays),
    }
    return result


if __name__ == "__main__":
    module_name = sys.argv[1] if len(sys.argv) > 1 else "logic_engine_original"
    result = run(module_name)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
