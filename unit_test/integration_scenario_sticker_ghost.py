"""
重現並驗證修復：貼紙被拿走後，若消失夠久，追蹤紀錄應該被清除，
之後同一張貼紙放回畫面時，不該被系統誤判成「多一個新餐點」。

情境：
    訂單只點了 1 份「味噌ラーメン」(qty=1)
    1. 貼紙放入 -> 穩定 -> OCR -> MATCHED -> tray 完成
    2. 把貼紙整個拿出畫面（完全不偵測到），超過 sticker_missing_frame 幀
    3. 驗證：tray.stickers 應該只剩 0 筆（幽靈紀錄被清除），而不是繼續掛著
    4. 把貼紙放回同一個位置
    5. 驗證：tray.stickers 只新增 1 筆（不是變成 2 筆）
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


def run(module_name: str):
    mod = importlib.import_module(module_name)
    LogicEngine = mod.LogicEngine

    cfg = FakeConfig(
        menus_ticket=["味噌ラーメン"],
        menus_sticker=["味噌ラーメン"],
        menus_mapping=[],
        stability=FakeStability(
            n_settle_frame=2, k_container_fail=2, k_container_new=1,
            tray_missing_frame=99,       # 這次不測 tray 消失，設大一點避免干擾
            sticker_missing_frame=3,     # 貼紙消失超過 3 幀就視為真的離開
        ),
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
    ticket_dt_box = [[10, 10], [60, 10], [60, 40], [10, 40]]

    sticker_xyxy = make_xyxy(70, 70, 120, 100)
    sticker_xywhr = (95.0, 85.0, 50.0, 30.0, 0.0)
    sticker_det = Detection("order_sticker", sticker_xyxy, sticker_xywhr)
    sticker_dt_box = [[70, 70], [120, 70], [120, 100], [70, 100]]

    # 建立新盤
    engine.update(frame, [tray_det])
    engine.update(frame, [tray_det])
    tray_id = list(engine.trays.keys())[0]

    # 訂單成功
    engine.update(frame, [tray_det, ticket_det])
    engine.update(frame, [tray_det, ticket_det])
    engine.update(frame, [tray_det, ticket_det])
    engine.set_ocr_busy(tray_id, "ticket", ticket_xyxy)
    engine.apply_ocr_result(tray_id, "ticket", frame, [ticket_dt_box], [("味噌ラーメン", 0.9)], False, ticket_xyxy)
    assert engine.trays[tray_id].expected_items == ["味噌ラーメン"]

    # 貼紙放入 -> 穩定 -> OCR -> MATCHED
    engine.update(frame, [tray_det, sticker_det])
    engine.update(frame, [tray_det, sticker_det])
    engine.update(frame, [tray_det, sticker_det])
    engine.set_ocr_busy(tray_id, "sticker", sticker_xyxy)
    engine.apply_ocr_result(tray_id, "sticker", frame, [sticker_dt_box], [("味噌ラーメン", 0.9)], False, sticker_xyxy)

    tray = engine.trays[tray_id]
    assert tray.state.name == "COMPLETED", f"應該完成，實際: {tray.state.name}"
    assert len(tray.stickers) == 1, f"核對完成當下應該只有 1 筆貼紙紀錄，實際: {len(tray.stickers)}"
    print(f"[核對完成] tray.stickers 筆數 = {len(tray.stickers)} (預期 1)")

    # 貼紙整個拿出畫面：連續好幾幀都偵測不到
    for i in range(6):
        engine.update(frame, [tray_det])  # 沒有 sticker_det
        print(f"  貼紙消失第 {i+1} 幀 -> tray.stickers 筆數 = {len(tray.trays[tray_id].stickers) if hasattr(tray, 'trays') else len(engine.trays[tray_id].stickers)}")

    stickers_after_gone = len(engine.trays[tray_id].stickers)
    print(f"[貼紙消失夠久後] tray.stickers 筆數 = {stickers_after_gone} (預期 0，代表幽靈紀錄已被清除)")
    assert stickers_after_gone == 0, "貼紙消失夠久後，幽靈紀錄應該被清除"

    # 把貼紙放回同一個位置
    engine.update(frame, [tray_det, sticker_det])
    stickers_after_return = len(engine.trays[tray_id].stickers)
    print(f"[貼紙放回] tray.stickers 筆數 = {stickers_after_return} (預期 1，不是 2)")
    assert stickers_after_return == 1, f"貼紙放回後應該只有 1 筆追蹤紀錄，實際: {stickers_after_return}（這就是原本的 bug：變成 2 筆）"

    print(f"\n[{module_name}] 貼紙幽靈紀錄測試通過 ✅")
    return stickers_after_return


if __name__ == "__main__":
    module_name = sys.argv[1] if len(sys.argv) > 1 else "logic_engine_step3"
    run(module_name)
