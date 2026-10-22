"""
test_debug_sheet.py
===================
除錯總覽圖（utils/debug_sheet.py）的測試。

用真實的 TrayStateMachine / StickerMatcher / DebugSheet，只把「OCR 引擎」「MQTT」換成假的，
走完：傳票確認 -> 貼紙 OK -> 貼紙連續讀錯成 NG -> 醬料包 -> 收尾出圖。

重點驗證三件事：
  1. 每個環節（傳票 / 貼紙 / 醬料包）的判定、模糊比對、mapping 都有被記下來。
  2. 收尾出的是一張 4024 x 3036 的圖 + 同名 json。
  3. **除錯功能本身出任何錯，都不能改變狀態機的判定結果**（這是它能放在正式流程裡的前提）。

執行：
    py -3.12 -m pytest unit_test/test_debug_sheet.py -q
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logic.geometry import PolygonXYXY  # noqa: E402,F401
from logic.models import Tray, TrackedItem, TrayState  # noqa: E402
from logic.readers.sticker_matcher import StickerMatcher  # noqa: E402
from logic.text_utils import FuzzyMatcher  # noqa: E402
from logic.tray_state_machine import TrayStateMachine  # noqa: E402
from utils.debug_sheet import DebugContext, DebugSheet  # noqa: E402

STICKERS = ["牛丼小", "豚丼並", "豚丼特盛", "ごまドレッシング"]
VARIANTS = {"牛丼･小": [["牛丼小"]], "豚丼･並": [["豚丼並"]], "サラダセット": [["ごまドレッシング"]]}


def poly(x0, y0, x1, y1):
    return (x0, y0, x1, y0, x1, y1, x0, y1)


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class FakeBus:
    def __init__(self):
        self.events = []

    def publish_det_status(self, payload):
        self.events.append(payload)


class FakeParser:
    """不跑 OCR，直接給定解析結果（同時提供 last_debug_rows，跟傳票解析器的介面一致）。"""

    def __init__(self, items, order_number="123"):
        self.items, self.order_number = items, order_number
        self.last_debug_rows = []

    def parse(self, frame_crop, dt_boxes, rec_res, is_flip):
        self.last_debug_rows = [
            {"raw": raw, "name": name, "score": 95.0, "qty": q, "qty_src": "獨立數字框",
             "status": "ok", "box_idx": [i]}
            for i, (raw, (name, q)) in enumerate(zip([t for t, _ in rec_res], self.items))
        ]
        return list(self.items), self.order_number


def make_cfg(tmp: Path, **kw):
    base = dict(enabled=True, mode="all", root_dir=str(tmp), folder="debug_sheet", canvas_w=4024,
                canvas_h=3036, jpeg_quality=80, save_json=True, font_path="", max_entries=80, crop_max_side=1600)
    base.update(kw)
    return SimpleNamespace(**base)


def make_sm(tmp: Path, debug=None, **kw):
    matcher = StickerMatcher(
        menus_sticker=STICKERS,
        fuzzy_matcher=FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2),
    )
    bus = FakeBus()
    sm = TrayStateMachine(
        bus=bus,
        order_parser=FakeParser([("牛丼･小", 2), ("サラダセット", 1)]),
        sticker_matcher=matcher,
        menu_variants=VARIANTS,
        n_settle_frame=2,
        object_class_items={"sesame": "ごまドレッシング"},
        wrong_item_confirm=2,
        debug_sheet=debug,
        **kw,
    )
    return sm, bus


def make_tray():
    tray = Tray(id="t1", rect=poly(0, 0, 500, 500), xywhr=(250, 250, 500, 500, 0.0), ticket_crop=None)
    tray.ticket = TrackedItem(bbox=poly(10, 10, 110, 90), xywhr=(60, 50, 100, 80, 0.0), cls_name="order_ticket")
    tray.state = TrayState.CHECKING_TICKET
    return tray


def sticker_crop(w=300, h=120):
    return np.full((h, w, 3), 240, np.uint8)


def ticket_ocr(sm, trays):
    crop = np.full((600, 400, 3), 245, np.uint8)
    rec = [("牛丼小", 0.97), ("サラダセット", 0.93)]
    dt = [box(20, 100, 200, 140), box(20, 200, 250, 240)]
    sm.apply_ocr_result(trays, "t1", "ticket", crop, dt, rec, False, trays["t1"].ticket.bbox)


def add_sticker(tray, bbox, cls="order_sticker"):
    ts = TrackedItem(bbox=bbox, xywhr=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[5]) / 2, 100, 60, 0.0), cls_name=cls)
    tray.stickers.append(ts)
    return ts


def sticker_ocr(sm, trays, ts, text, conf=0.95):
    trays["t1"].state = TrayState.CHECKING_STICKERS
    ts.is_ocr_busy = True
    sm.apply_ocr_result(trays, "t1", "sticker", sticker_crop(), [box(10, 30, 280, 90)],
                        [(text, conf)], False, ts.bbox)


# ---------------------------------------------------------------------------
def run_full_order(tmp_path):
    """傳票確認 -> 貼紙 A OK -> 貼紙 B 連讀兩次成 NG -> 醬料包。回傳 (ds, sm, tray, trays, a, b)。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT,
                    DebugContext(menu_variants=VARIANTS, object_class_items={"sesame": "ごまドレッシング"}),
                    crop_fn=lambda frame, xywhr: frame[0:80, 0:160])
    sm, _bus = make_sm(tmp_path, ds)
    tray = make_tray()
    trays = {"t1": tray}

    # 1. 傳票
    ticket_ocr(sm, trays)
    assert tray.state == TrayState.WAITING_STICKERS
    ev = ds._evidence["t1"]
    assert [e.kind for e in ev] == ["ticket"]
    assert ev[0].data["accepted"] is True and ev[0].data["order_number"] == "123"
    assert ev[0].data["rows"][0]["name"] == "牛丼･小"
    assert ev[0].data["combos"], "組合假設要被記下來"
    # 新版 OrderRequirements 會記下「實際套用」的傳票 -> 貼紙 mapping，圖上優先用它
    assert any(line.startswith("牛丼･小×") for line in ev[0].data["mapping_lines"]), ev[0].data["mapping_lines"]

    # 2. 貼紙 A：讀對 -> OK
    a = add_sticker(tray, poly(120, 120, 220, 180))
    sticker_ocr(sm, trays, a, "牛丼小")
    last = ds._evidence["t1"][-1]
    assert last.kind == "sticker"
    assert last.data["verdict"] == "OK", last.data
    assert last.data["matched"] == "牛丼小" and last.data["raw"] == "牛丼小"
    assert last.track == ds.track_uid(a)
    assert last.crop is not None and len(last.boxes) == 1

    # 3. 貼紙 B：讀成不在訂單內的品名。wrong_item_confirm=2 -> 第一次確認中、第二次成立
    b = add_sticker(tray, poly(260, 120, 360, 180))
    sticker_ocr(sm, trays, b, "豚丼特盛")
    assert ds._evidence["t1"][-1].data["verdict"] == "SUSPECT"
    assert not b.is_wrong_item
    b.stable_frames = 5
    sticker_ocr(sm, trays, b, "豚丼特盛")
    last = ds._evidence["t1"][-1]
    assert last.data["verdict"] == "NG", last.data
    assert b.is_wrong_item is True
    same_track = [e for e in ds._evidence["t1"] if e.track == ds.track_uid(b)]
    assert len(same_track) == 2, "同一張貼紙的兩次 OCR 要歸在同一個追蹤 id"

    # 4. 醬料包：不走 OCR，用原圖裁切
    ds.set_frame(np.full((600, 800, 3), 180, np.uint8))
    tray.state = TrayState.WAITING_STICKERS
    s = add_sticker(tray, poly(380, 120, 480, 180), cls="sesame")
    s.stable_frames = 99
    sm.resolve_known_class_items(trays)
    known = [e for e in ds._evidence["t1"] if e.kind == "known"]
    assert len(known) == 1 and known[0].crop is not None
    assert known[0].data["item_name"] == "ごまドレッシング"
    assert known[0].data["verdict"] in ("OK", "NG")

    return ds, sm, tray, trays, a, b, same_track


SUMMARY = dict(status="NGtoOK", tray_id="t1", order_number="123", start_time="2026-10-22T12:00:00",
               end_time="2026-10-22T12:01:00", duration_sec=60,
               expected=[{"牛丼小": 2}], achieved=[{"牛丼小": 1}], missing=[{"牛丼小": 1}],
               wrong=[{"豚丼特盛": 1}], timed_out=False, video_file="x.mp4")


def test_full_order_records_every_stage_and_renders_one_sheet(tmp_path):
    ds, sm, tray, trays, a, b, same_track = run_full_order(tmp_path)

    # 5. 事件註記
    sm.debug_hooks.event("t1", b, "錯誤品項 豚丼特盛 已被移走")
    assert any("已被移走" in x for x in ds._evidence["t1"][-1].data.get("events", [])
               + same_track[-1].data.get("events", []))

    # 6. 收尾 -> 一張 4024x3036 + json
    path = ds.finalize("t1", dict(SUMMARY))
    assert path is not None
    ds.shutdown()
    jpg = Path(path)
    assert jpg.is_file() and jpg.parent.name == "debug_sheet"
    img = cv2.imdecode(np.fromfile(str(jpg), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (3036, 4024, 3)
    assert img.std() > 10, "圖不能是空白的"
    doc = json.loads(jpg.with_suffix(".json").read_text(encoding="utf-8"))
    verdicts = [e["data"].get("verdict") for e in doc["entries"] if e["kind"] in ("sticker", "known")]
    assert "OK" in verdicts and "NG" in verdicts
    assert doc["summary"]["status"] == "NGtoOK"
    assert "t1" not in ds._evidence, "finalize 之後證據要被取走"
    assert not list(jpg.parent.glob("*.tmp")), "不能留下寫到一半的 .tmp"


def test_disabled_is_noop(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, enabled=False), tmp_path, ROOT)
    assert ds.record_ocr("t1", "sticker", sticker_crop(), [], [], {}) is None
    assert ds.finalize("t1", {"status": "OK"}) is None
    assert not list(tmp_path.rglob("*.jpg"))


def test_ng_only_mode_skips_clean_orders(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, mode="ng_only"), tmp_path, ROOT)
    ds.record_ocr("t1", "sticker", sticker_crop(), [], [], {"verdict": "OK"})
    assert ds.finalize("t1", {"status": "OK", "missing": [], "wrong": []}) is None
    ds.record_ocr("t1", "sticker", sticker_crop(), [], [], {"verdict": "NG"})
    path = ds.finalize("t1", {"status": "NG", "missing": [{"x": 1}], "wrong": []})
    assert path is not None
    ds.shutdown()
    assert Path(path).is_file()


def test_discard_drops_evidence(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    ds.record_ocr("t1", "sticker", sticker_crop(), [], [], {})
    ds.discard("t1")
    assert "t1" not in ds._evidence
    ds.record_ocr("t2", "sticker", sticker_crop(), [], [], {})
    ds.discard()
    assert not ds._evidence


def test_main_loop_only_copies_and_background_downscales(tmp_path):
    """主迴圈不縮圖（INTER_AREA 縮一張傳票要 25~50ms，會卡辨識）；縮圖在背景，框座標跟著縮。"""
    ds = DebugSheet(make_cfg(tmp_path, crop_max_side=500), tmp_path, ROOT)
    big = np.zeros((2000, 1000, 3), np.uint8)
    ev = ds.record_ocr("t1", "ticket", big, [box(100, 200, 500, 400)], [("文字", 0.9)], {})
    assert ev.crop.shape == big.shape and ev.crop is not big, "記錄時只複製、不縮圖"
    big[:] = 255
    assert ev.crop.max() == 0, "必須是複製，不能跟呼叫端共用記憶體"

    ds._shrink_entries([ev])                      # 這是背景 thread 渲染前做的事
    assert max(ev.crop.shape[:2]) == 500
    p = np.asarray(ev.boxes[0]["poly"])
    assert abs(p[:, 1].max() - 100.0) < 1.0       # 縮了 4 倍：400→100


def test_only_latest_ticket_crops_are_kept_in_memory(tmp_path):
    """傳票解析失敗會一直重試；只留最新 4 張的圖，更舊的只留文字紀錄。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    for i in range(10):
        ds.record_ocr("t1", "ticket", np.zeros((50, 50, 3), np.uint8), [], [], {"n": i})
    ev = ds._evidence["t1"]
    assert len(ev) == 10, "文字紀錄一筆都不能少"
    assert [e.crop is not None for e in ev] == [False] * 6 + [True] * 4


def test_malformed_ocr_input_does_not_raise(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    ds.record_ocr("t1", "sticker", None, [[1, 2]], [("x",)], {})        # 框不是 4 點、rec 缺信心
    ds.record_ocr("t1", "sticker", sticker_crop(), None, None, {})
    ds.record_known("t1", None, {})
    assert ds.finalize("t1", {"status": "NG"}) is not None
    ds.shutdown()


class ExplodingDebug:
    """模擬除錯功能整個壞掉：每個方法都丟例外。"""
    enabled = True

    def __getattr__(self, name):
        def boom(*a, **k):
            raise RuntimeError(f"debug {name} exploded")
        return boom


def test_debug_failure_never_changes_state_machine_result(tmp_path):
    """同一段劇本，除錯功能關閉 / 正常 / 全部爆炸，狀態機的結論必須完全一樣。"""
    def run(debug):
        sm, bus = make_sm(tmp_path, debug)
        tray = make_tray()
        trays = {"t1": tray}
        ticket_ocr(sm, trays)
        a = add_sticker(tray, poly(120, 120, 220, 180))
        sticker_ocr(sm, trays, a, "牛丼小")
        b = add_sticker(tray, poly(260, 120, 360, 180))
        sticker_ocr(sm, trays, b, "豚丼特盛")
        sticker_ocr(sm, trays, b, "豚丼特盛")
        return (tray.state.name, list(tray.expected_items), list(tray.checked_items),
                dict(tray.confirmed_counts), tray.ever_had_wrong_item, list(tray.wrong_item_history),
                [(e.get("status"), str(e.get("items"))) for e in bus.events])

    baseline = run(None)
    normal = run(DebugSheet(make_cfg(tmp_path), tmp_path, ROOT))
    exploding = run(ExplodingDebug())
    assert baseline == normal == exploding


def test_every_rendered_character_exists_in_the_font(tmp_path):
    """圖上實際畫出去的每個字都要在 NotoSansJP 裡，缺的字會變成方塊（例如中文的「查」日文字型沒有）。"""
    from PIL import ImageFont
    import utils.debug_sheet as dbg

    ds, sm, tray, trays, a, b, _ = run_full_order(tmp_path)
    sm.debug_hooks.event("t1", b, "錯誤品項 豚丼特盛 已被移走")

    drawn = []
    orig = dbg._TextLayer.flush

    def spy(self):
        drawn.extend(s for _, _, s, _, _ in self._items)
        return orig(self)

    dbg._TextLayer.flush = spy
    try:
        ds.render_to_array(ds._evidence["t1"], dict(SUMMARY))
    finally:
        dbg._TextLayer.flush = orig

    from PIL import Image, ImageDraw
    font = ImageFont.truetype(ds.font_path, 40)

    def bitmap(ch):
        im = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(im).text((6, 6), ch, font=font, fill=255)
        return im.tobytes()

    tofu = bitmap("\uffff")      # 字型沒有的字會畫成 .notdef 方塊
    missing = sorted({ch for text in drawn for ch in text
                      if not ch.isspace() and bitmap(ch) == tofu})
    assert drawn, "沒有攔到任何文字"
    assert not missing, f"字型缺這些字，圖上會變方塊: {missing}"


def test_timing_is_collected_for_main_loop_and_background(tmp_path):
    """主迴圈（會拖慢辨識）與背景（獨立 thread）的耗時都要分開量到，並在出圖時有一行 timing log。"""
    ds, sm, tray, trays, a, b, _ = run_full_order(tmp_path)
    sm.debug_hooks.event("t1", b, "x")

    cost = ds._cost["t1"]
    assert cost["n"] >= 5                              # 傳票 1 + 貼紙 3 + 醬料包 1 + 事件
    for label in ("ticket", "sticker", "known", "event"):
        assert cost["by"].get(label, 0) > 0, f"{label} 沒被計時"
    assert cost["total"] >= cost["max"] > 0

    assert ds.finalize("t1", dict(SUMMARY)) is not None
    ds.shutdown()
    t = ds.last_timing
    assert t["main_n"] >= 5 and t["main_total_s"] > 0 and t["main_finalize_s"] >= 0
    assert t["render_s"] > 0 and t["encode_s"] > 0 and t["bg_total_s"] >= t["render_s"]
    assert t["queue_wait_s"] >= 0
    assert "t1" not in ds._cost, "finalize 之後這筆訂單的計時要被取走"
    assert ds._totals["n"] == 1


def test_timing_costs_nothing_when_disabled(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, enabled=False), tmp_path, ROOT)
    ds.add_cost("t1", 1.0, "sticker")
    assert not ds._cost


def test_slow_record_warns_but_does_not_raise(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, slow_record_ms=0.0), tmp_path, ROOT)
    ds.add_cost("t1", 0.5, "sticker")                  # 超過門檻 -> 只記 warning
    assert ds._cost["t1"]["max_label"] == "sticker"


# ---------------------------------------------------------------------------
# config.yaml -> cfg.debug_sheet（走專案真正的 load_config）
# ---------------------------------------------------------------------------
def _profile_copy(tmp_path, profile="yoshinoya", edit=None):
    """複製 tasks/ocr 到暫存資料夾（includes 用相對路徑），可選擇改寫 config.yaml 的內容。"""
    import shutil
    import yaml
    dst = tmp_path / "tasks" / "ocr"
    shutil.copytree(ROOT / "tasks" / "ocr", dst)
    path = dst / profile / "config.yaml"
    if edit is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, indent=4), encoding="utf-8")
    return path


def test_both_shipped_profiles_load_debug_sheet_section(tmp_path):
    import config
    from config import DebugSheetCfg
    for profile in ("yoshinoya", "suzumo"):
        cfg = config.load_config(str(ROOT / "tasks" / "ocr" / profile / "config.yaml"))
        assert cfg.debug_sheet == DebugSheetCfg(), f"{profile}: 隨附的 config.yaml 應該等於預設值"


def test_missing_section_falls_back_to_defaults(tmp_path):
    """機器上執行期會被改寫的舊 config.yaml 沒有這一段，也要能載入。"""
    import config
    from config import DebugSheetCfg
    path = _profile_copy(tmp_path, edit=lambda d: d.pop("debug_sheet"))
    assert "debug_sheet" not in path.read_text(encoding="utf-8")
    assert config.load_config(str(path)).debug_sheet == DebugSheetCfg()


def test_custom_values_and_bad_mode(tmp_path):
    import config

    def edit(d):
        d["debug_sheet"].update(mode="ng_only", jpeg_quality=70, folder="dbg", slow_render_s=2.5,
                                enabled=False, root_dir="/tmp/x")

    ds = config.load_config(str(_profile_copy(tmp_path, edit=edit))).debug_sheet
    assert (ds.mode, ds.jpeg_quality, ds.folder, ds.slow_render_s, ds.enabled, ds.root_dir) == \
        ("ng_only", 70, "dbg", 2.5, False, "/tmp/x")

    bad = tmp_path / "bad"
    bad.mkdir()
    ds = config.load_config(str(_profile_copy(bad, edit=lambda d: d["debug_sheet"].update(mode="whatever")))).debug_sheet
    assert ds.mode == "all", "mode 寫錯要退回 all，不能讓整個 app 起不來"


def test_suzumo_table_parser_rows_are_converted(tmp_path):
    """鈴茂的表格解析器沒有 last_debug_rows，要從它的 last_rows（TicketRow）轉成同一種逐列格式。"""
    from logic.readers.debug_hooks import ReaderDebugHooks
    from logic.szm.ticket_layout import TicketRow

    def row(**kw):
        base = dict(row_index=0, cy=0.0, height=10.0, name=None, name_raw="", name_score=0.0, name_margin=0.0,
                    name_reason="", quantity=None, quantity_raw=None, quantity_source="")
        base.update(kw)
        return TicketRow(**base)

    class FakeTableParser:
        last_rows = [
            row(name="牛丼･小", name_raw="牛丼小", name_score=93.0, quantity=2, quantity_source="qty_column"),
            row(name=None, name_raw="ほげ", name_score=12.0),
            row(name="豚丼･並", name_raw="豚丼並", name_score=88.0, quantity=None),
        ]

    rows = ReaderDebugHooks(None, order_parser=FakeTableParser()).rows()
    assert [r["status"] for r in rows] == ["ok", "no_match", "qty_unconfirmed"]
    assert rows[0]["name"] == "牛丼･小" and rows[0]["qty"] == 2 and rows[0]["qty_src"] == "qty_column"
    assert rows[1]["name"] is None and rows[2]["qty"] is None
