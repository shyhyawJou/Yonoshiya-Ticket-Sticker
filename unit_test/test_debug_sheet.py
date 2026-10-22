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
    assert jpg.name.endswith("_p01.jpg"), "單頁訂單也帶 _p01，每張檔名格式一致"
    doc = json.loads(DebugSheet.json_path_for(jpg).read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# NG 之後被 OK 貼紙蓋住：NG 的證據不能從圖上消失
# ---------------------------------------------------------------------------
def _renderer(ds):
    from utils.debug_sheet import SheetRenderer
    return SheetRenderer(ds.ctx, ds.font_path)


def _stickers(ds):
    return [e for e in ds._evidence["t1"] if e.kind in ("sticker", "known")]


def _order_with_ng(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    sm, _ = make_sm(tmp_path, ds)
    tray = make_tray()
    trays = {"t1": tray}
    ticket_ocr(sm, trays)
    b = add_sticker(tray, poly(260, 120, 360, 180))
    sticker_ocr(sm, trays, b, "豚丼特盛")
    b.stable_frames = 5
    sticker_ocr(sm, trays, b, "豚丼特盛")
    assert b.is_wrong_item and tray.ever_had_wrong_item
    return ds, sm, tray, trays, b


def test_ng_covered_by_ok_sticker_keeps_both_tiles(tmp_path):
    """復現：NG 出現後用 OK 貼紙蓋住（追蹤器把它當同一個物件重讀，判定 NG -> OK）。
    以前併成一格、只顯示最後的 OK，NG 那張的裁切圖與 det 框就消失了。"""
    ds, sm, tray, trays, b = _order_with_ng(tmp_path)
    b.stable_frames = 5
    sticker_ocr(sm, trays, b, "牛丼小")
    assert not b.is_wrong_item and tray.ever_had_wrong_item, "狀態機本來就會解除警示，但 CSV 仍記錯誤"

    segs = _renderer(ds)._group_tracks(_stickers(ds))
    assert len(segs) == 2, "NG 與 OK 必須是兩格"
    ng, ok = segs[0], segs[1]
    assert ng[-1].data["verdict"] == "NG" and ng[-1].crop is not None and ng[-1].boxes
    assert ok[-1].data["verdict"] == "OK"
    assert ng.next is ok and ok.prev is ng, "兩格要互相連結，圖上才能互相註明"
    lines = [t for t, _, _ in _renderer(ds)._item_lines(ng)]
    assert any("之後" in x and "OK" in x for x in lines), lines
    assert any("先前" in x and "NG" in x for x in [t for t, _, _ in _renderer(ds)._item_lines(ok)])

    # 整張圖要畫得出來，而且 json 裡兩筆判定都在
    ds.render_to_array(ds._evidence["t1"], dict(SUMMARY))
    assert ds.finalize("t1", dict(SUMMARY)) is not None
    ds.shutdown()
    doc = json.loads(next(tmp_path.rglob("*.json")).read_text(encoding="utf-8"))
    assert [e["data"]["verdict"] for e in doc["entries"] if e["kind"] == "sticker"][-3:] == ["SUSPECT", "NG", "OK"]


def test_single_misread_then_ok_keeps_the_suspect_tile(tmp_path):
    """只讀錯一次（確認中、還沒成立）就換成對的：狀態機不記錯誤，但除錯圖要留下那一次疑似。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    sm, _ = make_sm(tmp_path, ds)
    tray = make_tray()
    trays = {"t1": tray}
    ticket_ocr(sm, trays)
    b = add_sticker(tray, poly(260, 120, 360, 180))
    sticker_ocr(sm, trays, b, "豚丼特盛")
    assert not b.is_wrong_item and not tray.ever_had_wrong_item
    b.stable_frames = 5
    sticker_ocr(sm, trays, b, "牛丼小")
    segs = _renderer(ds)._group_tracks(_stickers(ds))
    assert [s[-1].data["verdict"] for s in segs] == ["SUSPECT", "OK"]


def test_rereading_a_normal_sticker_stays_one_tile(tmp_path):
    """同一張正常貼紙讀好幾次（OK / 沒讀到字）不能拆開，否則圖面會被塞滿。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    track = ds.track_uid(SimpleNamespace())
    for verdict in ("OK", "UNREAD", "OK", "UNTRACKED", "OK"):
        ds.record_ocr("t1", "sticker", sticker_crop(), [], [], {"verdict": verdict}, track=track)
    assert len(_renderer(ds)._group_tracks(_stickers(ds))) == 1


def test_ng_sticker_removed_then_new_sticker_are_separate_tracks(tmp_path):
    """NG 的貼紙被拿走、另放一張 OK（追蹤器當成不同物件）：兩格都在，NG 那格註記「已被移走」。"""
    ds, sm, tray, trays, b = _order_with_ng(tmp_path)
    b.is_missing = True
    sm.resolve_wrong_item_removal(trays)
    c = add_sticker(tray, poly(380, 120, 480, 180))
    sticker_ocr(sm, trays, c, "牛丼小")
    segs = _renderer(ds)._group_tracks(_stickers(ds))
    assert [s[-1].data["verdict"] for s in segs] == ["NG", "OK"]
    assert segs[0].next is None and segs[1].prev is None, "不同追蹤物件不能互相連結"
    assert any("已被移走" in x for x in segs[0][-1].data.get("events", []))


def test_trimming_never_frees_the_ng_crop(tmp_path):
    """長訂單超過 max_entries 時會釋放最舊的貼紙裁切圖；NG（之後被同一位置的 OK 蓋過、不再是最新一筆）不能先被釋放。"""
    ds = DebugSheet(make_cfg(tmp_path, max_entries=20), tmp_path, ROOT)
    t = ds.track_uid(SimpleNamespace())
    crop = lambda: np.zeros((40, 40, 3), np.uint8)   # noqa: E731
    ng = ds.record_ocr("t1", "sticker", crop(), [], [], {"verdict": "NG"}, track=t)
    sus = ds.record_ocr("t1", "sticker", crop(), [], [], {"verdict": "SUSPECT"}, track=ds.track_uid(SimpleNamespace()))
    ds.record_ocr("t1", "sticker", crop(), [], [], {"verdict": "OK"}, track=t)
    other = ds.track_uid(SimpleNamespace())
    for _ in range(60):
        ds.record_ocr("t1", "sticker", crop(), [], [], {"verdict": "OK"}, track=other)
    assert ng.crop is not None and sus.crop is not None, "有問題的判定裁切圖不能被釋放"
    assert sum(1 for e in ds._evidence["t1"] if e.crop is not None) < 30, "正常的重讀仍要被釋放，記憶體才有上限"


# ---------------------------------------------------------------------------
# 分頁：塞不下的格子往下一張圖放
# ---------------------------------------------------------------------------
def _fill(ds, n_ng=2, n_ok=28, tray="t1"):
    """記下 n_ng 張 NG + n_ok 張 OK 貼紙（每張各自一個追蹤物件，各佔一格）。"""
    def add(verdict, raw):
        ds.record_ocr(tray, "sticker", sticker_crop(), [box(10, 30, 280, 90)], [(raw, 0.95)],
                      {"verdict": verdict, "raw": raw, "matched": raw, "note": verdict},
                      track=ds.track_uid(SimpleNamespace()))
    for _ in range(n_ok):          # 先放 OK，NG 在時間上排最後，驗證「排序」而不是「先來後到」
        add("OK", "牛丼小")
    for _ in range(n_ng):
        add("NG", "豚丼特盛")


def test_overflow_tiles_move_to_next_page_with_problems_first(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, tiles_per_page=24), tmp_path, ROOT)
    _fill(ds, n_ng=2, n_ok=28)
    pages, info = _renderer_for(ds).plan(ds._evidence["t1"])
    assert len(pages) == 2 and info["hidden_tiles"] == 0
    assert [len(p["segs"]) for p in pages] == [24, 6], "30 格：第 1 頁 24、第 2 頁 6，一格都不能丟"
    assert [s[-1].data["verdict"] for s in pages[0]["segs"][:2]] == ["NG", "NG"], "NG 一定在第 1 頁最前面"
    assert all(s[-1].data["verdict"] == "OK" for s in pages[1]["segs"])
    assert pages[1]["offset"] == 24, "格子編號要跨頁連續（#25 起）"


def _renderer_for(ds):
    return ds._get_renderer()


def test_every_page_has_the_same_name_format_and_one_shared_json(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, tiles_per_page=24), tmp_path, ROOT)
    _fill(ds, n_ng=2, n_ok=28)
    p1 = Path(ds.finalize("t1", dict(SUMMARY)))
    ds.shutdown()
    folder = p1.parent
    base = p1.name[: -len("_p01.jpg")]
    assert p1.name == f"{base}_p01.jpg", "finalize 回傳第 1 頁的路徑"
    names = sorted(p.name for p in folder.iterdir())
    assert names == sorted([f"{base}_p01.jpg", f"{base}_p02.jpg", f"{base}.json"]), names
    for name in (f"{base}_p01.jpg", f"{base}_p02.jpg"):
        img = cv2.imdecode(np.fromfile(str(folder / name), np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (3036, 4024, 3)
    # 任何一頁都能找到同一份 json；json 不帶頁碼
    assert DebugSheet.json_path_for(p1) == DebugSheet.json_path_for(folder / f"{base}_p02.jpg") == folder / f"{base}.json"
    doc = json.loads((folder / f"{base}.json").read_text(encoding="utf-8"))
    assert doc["pages"] == 2 and doc["files"] == [f"{base}_p01.jpg", f"{base}_p02.jpg"]
    assert len([e for e in doc["entries"] if e["kind"] == "sticker"]) == 30, "json 要有全部 30 筆"
    assert ds.last_timing["pages"] == 2 and ds.last_timing["path"] == str(p1)
    assert not list(folder.glob("*.tmp"))


def test_single_page_order_is_also_named_p01(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    _fill(ds, n_ng=1, n_ok=3)
    path = Path(ds.finalize("t1", dict(SUMMARY)))
    ds.shutdown()
    assert path.name.endswith("_p01.jpg")
    assert sorted(p.suffix for p in path.parent.iterdir()) == [".jpg", ".json"]
    assert ds.last_timing["pages"] == 1


def test_page_number_is_fixed_two_digits(tmp_path):
    base = tmp_path / "20261022_120000_t1_No1_OK.jpg"
    names = [DebugSheet.page_path(base, n).name for n in (1, 2, 9, 10, 99)]
    assert names == [f"20261022_120000_t1_No1_OK_p{n:02d}.jpg" for n in (1, 2, 9, 10, 99)], "固定 2 碼"
    assert names == sorted(names), "固定 2 碼 → 字典序 = 頁序"
    assert DebugSheet.json_path_for(DebugSheet.page_path(base, 99)).name == "20261022_120000_t1_No1_OK.json"

    assert DebugSheet.json_path_for(DebugSheet.page_path(base, 1)) == DebugSheet.json_path_for(DebugSheet.page_path(base, 99))


def _expect_value_error(fn, must_contain: str):
    """不依賴 pytest 的 raises：必須丟 ValueError，而且訊息要提到上限。"""
    try:
        fn()
    except ValueError as e:
        assert must_contain in str(e), f"錯誤訊息要說明上限: {e}"
        return
    raise AssertionError("應該要丟 ValueError")


def test_max_pages_caps_output_and_json_keeps_everything(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path, tiles_per_page=5, max_pages=2), tmp_path, ROOT)
    _fill(ds, n_ng=1, n_ok=13)                       # 14 格，只放得下 10 格
    pages, info = _renderer_for(ds).plan(ds._evidence["t1"])
    assert len(pages) == 2 and info["hidden_tiles"] == 4
    path = Path(ds.finalize("t1", dict(SUMMARY)))
    ds.shutdown()
    assert len(list(path.parent.glob("*.jpg"))) == 2
    doc = json.loads(DebugSheet.json_path_for(path).read_text(encoding="utf-8"))
    assert len([e for e in doc["entries"] if e["kind"] == "sticker"]) == 14


def test_older_tickets_with_crops_go_to_following_pages(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    for i in range(6):                               # 傳票解析失敗反覆重試，留了 6 筆；只有最新 4 筆還有圖
        ds.record_ocr("t1", "ticket", np.full((300, 200, 3), 240, np.uint8), [box(10, 10, 100, 40)],
                      [("牛丼小", 0.9)], {"order_number": "1", "rows": [], "accepted": False, "outcome": f"第{i}次"})
    pages, info = _renderer_for(ds).plan(ds._evidence["t1"])
    assert [len(p["tickets"]) for p in pages] == [3, 1], "第 1 頁最新 3 張，第 4 新的（還有圖）放第 2 頁"
    assert info["hidden_tickets"] == 2, "沒有圖的 2 筆舊傳票只留在 json"
    assert pages[0]["tickets"][-1].data["outcome"] == "第5次"


def test_page_labels_only_appear_on_multi_page_and_glyphs_exist(tmp_path):
    from PIL import Image, ImageDraw, ImageFont
    import utils.debug_sheet as dbg

    def drawn_texts(ds):
        drawn, orig = [], dbg._TextLayer.flush

        def spy(self):
            drawn.append([s for _, _, s, _, _ in self._items])
            return orig(self)
        dbg._TextLayer.flush = spy
        try:
            ds.render_pages(ds._evidence["t1"], dict(SUMMARY))
        finally:
            dbg._TextLayer.flush = orig
        return drawn

    ds = DebugSheet(make_cfg(tmp_path, tiles_per_page=24), tmp_path, ROOT)
    _fill(ds, n_ng=2, n_ok=28)
    pages = drawn_texts(ds)
    assert len(pages) == 2
    assert "第 1 / 2 頁" in pages[0] and "第 2 / 2 頁" in pages[1]
    assert any("續：第 2 頁" in t for t in pages[0]), "第 1 頁要提示續頁"
    assert not any("續" in t for t in pages[1]), "最後一頁不提示續頁"

    font = ImageFont.truetype(ds.font_path, 40)

    def bitmap(ch):
        im = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(im).text((6, 6), ch, font=font, fill=255)
        return im.tobytes()
    tofu = bitmap("\uffff")
    missing = sorted({c for page in pages for t in page for c in t if not c.isspace() and bitmap(c) == tofu})
    assert not missing, f"字型缺這些字: {missing}"

    small = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    _fill(small, n_ng=1, n_ok=2)
    only = drawn_texts(small)
    assert len(only) == 1 and not any("頁" in t for t in only[0]), "單頁不顯示頁碼"


def test_config_max_pages_over_99_is_an_error_at_load(tmp_path):
    import config

    def edit(n):
        return lambda d: d["debug_sheet"].update(max_pages=n)

    assert config.load_config(str(_profile_copy(tmp_path / "a", edit=edit(99)))).debug_sheet.max_pages == 99
    _expect_value_error(lambda: config.load_config(str(_profile_copy(tmp_path / "b", edit=edit(100)))), "99")


# ---------------------------------------------------------------------------
# 傳票：圖上的框只標編號（文字太多會擋住傳票），貼紙維持標文字
# ---------------------------------------------------------------------------
def _drawn_texts(ds, summary=None):
    import utils.debug_sheet as dbg
    drawn, orig = [], dbg._TextLayer.flush

    def spy(self):
        drawn.extend(s for _, _, s, _, _ in self._items)
        return orig(self)
    dbg._TextLayer.flush = spy
    try:
        ds.render_pages(ds._evidence["t1"], summary or dict(SUMMARY))
    finally:
        dbg._TextLayer.flush = orig
    return drawn


def _ticket_with_many_boxes(ds, n=12, linked=(2, 3)):
    boxes = [box(20, 20 + i * 40, 300, 50 + i * 40) for i in range(n)]
    rec = [(f"品名文字{i}", 0.9) for i in range(n)]
    rows = [{"raw": "牛丼小2", "name": "牛丼･小", "score": 95.0, "qty": 2, "qty_src": "獨立數字框",
             "status": "ok", "box_idx": list(linked)}]
    ds.record_ocr("t1", "ticket", np.full((600, 400, 3), 245, np.uint8), boxes, rec,
                  dict(order_number="1", rows=rows, accepted=True, outcome="主票確認"))


def test_ticket_boxes_are_labeled_with_index_only(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    _ticket_with_many_boxes(ds)
    texts = _drawn_texts(ds)
    import re
    assert not [t for t in texts if re.match(r"^\d+:.+ \d\.\d\d$", t)], "傳票的框不可標 `編號:文字 信心`"
    assert {"0", "1", "5", "11"} <= set(texts), "框上只有編號"
    assert "框 2、3" in texts, "右側每一列要標出它用到哪幾個框"


def test_every_ticket_box_text_and_conf_is_listed_under_the_crop(tmp_path):
    """圖上的框只標編號；每個框（不論有沒有歸到某一列）的 rec 文字與信心都要在傳票圖下方對照，資訊不能不見。
    歸了列的框，列裡只有合併後的文字，看不到各框自己的信心。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    _ticket_with_many_boxes(ds, n=12, linked=(2, 3))
    texts = _drawn_texts(ds)
    for i in range(12):
        assert f"{i}:品名文字{i}(0.90)" in texts, f"框 {i} 的文字與信心要列出來（含已歸列的 2、3）"
    assert any(t.startswith("框編號對照") for t in texts)


def test_ticket_box_legend_says_how_many_did_not_fit(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    _ticket_with_many_boxes(ds, n=1500, linked=())           # 多到放不下（400 個在整張面板上還放得下）
    texts = _drawn_texts(ds)
    note = [t for t in texts if t.startswith("…另 ") and t.endswith("個框見 .json")]
    assert len(note) == 1, "放不下時要寫明還有幾個框、文字在 .json"
    shown = [t for t in texts if t[:1].isdigit() and ":品名文字" in t]
    assert 0 < len(shown) < 1500
    assert len(shown) + int(note[0].split()[1]) == 1500, "顯示的 + 沒顯示的 = 全部，不能悄悄少"


def test_ticket_without_row_linkage_lists_every_box(tmp_path):
    """鈴茂的表格解析器沒有『列 -> 框』對應：全部的框都要在右側對照，否則圖上的編號沒有意義。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    _ticket_with_many_boxes(ds, n=5, linked=())
    texts = _drawn_texts(ds)
    assert any(t.startswith("框編號對照") for t in texts)
    assert any("4:品名文字4(0.90)" in t for t in texts)


def test_sticker_labels_keep_text_and_confidence(tmp_path):
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    ds.record_ocr("t1", "sticker", sticker_crop(), [box(10, 30, 280, 90)], [("豚丼特盛", 0.95)],
                  {"verdict": "NG", "raw": "豚丼特盛", "matched": "豚丼特盛", "note": "x"}, track=1)
    assert "0:豚丼特盛 0.95" in _drawn_texts(ds), "貼紙不更改：仍標 編號:文字 信心"


# ---------------------------------------------------------------------------
# 版面：標題不重疊、文字不超出畫布、標頭不截斷、畫布尺寸可以改
# ---------------------------------------------------------------------------
def _tl(ds):
    import utils.debug_sheet as dbg
    return dbg._TextLayer(np.zeros((10, 10, 3), np.uint8), ds.font_path)


def test_sticker_title_never_overlaps_its_ocr_tag(tmp_path):
    """標題左邊（判定 #編號 種類 時間）與右邊（OCR×n）在任何格寬都不能撞在一起。
    實際現場：NG 常被重讀（OCR×51）、UNREAD / CHECKING 的名稱又長，窄格子就疊在一起。"""
    from utils.debug_sheet import SheetRenderer, VERDICT_STYLE
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    tl = _tl(ds)
    for w in (180, 240, 330, 432, 544, 730, 1102, 2218):
        for label in [v[1] for v in VERDICT_STYLE.values()]:
            for idx in (1, 24, 96):
                for tag in ("", "OCR×2", "OCR×51", "OCR×999"):
                    text, fs, tag_fs = SheetRenderer._title_plan(tl, label, idx, "醬料包/物件", "15:50:29", tag, w)
                    used = tl.width(text, fs) + ((16 + tl.width(tag, tag_fs)) if tag else 0)
                    assert used <= w - 28 or fs == 18, f"w={w} {label} #{idx} {tag!r}: 左右兩邊會重疊 ({used} > {w - 28})"


def _all_text_items(ds, summary=None, canvas=None):
    """渲染並攔下每一頁所有文字 (x, y, 文字, 字級, 寬度, 畫布寬)。"""
    import utils.debug_sheet as dbg
    seen, orig = [], dbg._TextLayer.flush

    def spy(self):
        seen.extend((x, y, t, sz, self.width(t, sz), self.canvas.shape[1]) for x, y, t, sz, _ in self._items)
        return orig(self)
    dbg._TextLayer.flush = spy
    try:
        ds.render_pages(ds._evidence["t1"], summary or dict(SUMMARY))
    finally:
        dbg._TextLayer.flush = orig
    return seen


def test_no_text_runs_off_the_canvas(tmp_path):
    """任何一段文字都不能超出畫布右緣（以前右上角的 profile 與標頭「應有」會被切掉）。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS, profile=(
        "yoshinoya / parser=legacy / mode=single / 一段很長很長很長很長的描述文字用來確認會換行而不是被畫布邊緣切掉")))
    _ticket_with_many_boxes(ds, n=28, linked=(2, 3))
    _fill(ds, n_ng=3, n_ok=10)
    long_sum = dict(SUMMARY, video_file="20261022_155102_131546_event_with_a_really_long_name.mp4",
                    expected=[{f"品項名稱{i}": 1} for i in range(40)])
    for x, y, t, sz, w, cw in _all_text_items(ds, long_sum):
        assert x + w <= cw, f"{t!r} 超出畫布右緣: x={x} w={w} canvas={cw}"


def test_header_wraps_items_instead_of_truncating(tmp_path):
    from utils.debug_sheet import SheetRenderer
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT)
    r = SheetRenderer(ds.ctx, ds.font_path)
    base = np.zeros((r.H, r.W, 3), np.uint8)

    def header(summary):
        tl = _tl(ds)
        tl.canvas = base.copy()
        hh = r._draw_header(tl.canvas, tl, summary, 1, 5, 5, 1, 1)
        return hh, [t for _, _, t, _, _ in tl._items]

    hh_short, _ = header(dict(SUMMARY))
    assert hh_short == r.HEADER_H, "項目少時標頭高度不變"
    exp = [{f"品項{i:02d}": 2} for i in range(12)]
    hh, texts = header(dict(SUMMARY, expected=exp))
    joined = "".join(texts)
    assert all(f"品項{i:02d}×2" in joined for i in range(12)), "12 個項目一個都不能少（以前第 11 個起被截成「…」）"
    assert "共" not in joined
    hh_huge, texts = header(dict(SUMMARY, expected=[{f"品項{i}": 1} for i in range(300)]))
    assert any("共 300 項" in t for t in texts), "太多放不下時要寫明總數，不是只有「…」"
    assert hh < hh_huge <= 700, "標頭依內容加高，但有上限"


def test_canvas_size_only_scales_the_layout(tmp_path):
    """改 canvas_w / canvas_h 不能讓版面擠壞：先照 4024 寬排好再縮放，所以結果 = 預設尺寸縮小。"""
    def render(w, h):
        ds = DebugSheet(make_cfg(tmp_path, canvas_w=w, canvas_h=h), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
        _ticket_with_many_boxes(ds, n=12, linked=(2, 3))
        _fill(ds, n_ng=3, n_ok=10)
        return ds.render_pages(ds._evidence["t1"], dict(SUMMARY))

    big = render(4024, 3036)[0]
    small = render(2576, 1944)[0]
    wide = render(1920, 1080)[0]
    assert big.shape == (3036, 4024, 3) and small.shape == (1944, 2576, 3) and wide.shape == (1080, 1920, 3)
    ref = cv2.resize(big, (2576, 1944), interpolation=cv2.INTER_AREA).astype(np.int16)
    assert np.abs(ref - small.astype(np.int16)).mean() < 3.0, "縮小的畫布應該只是預設版面縮小，而不是另一種擠壓過的版面"


def test_text_stays_inside_canvas_at_small_and_odd_sizes(tmp_path):
    for w, h in ((2576, 1944), (1920, 1080), (1280, 720)):
        ds = DebugSheet(make_cfg(tmp_path, canvas_w=w, canvas_h=h), tmp_path, ROOT)
        _ticket_with_many_boxes(ds, n=28, linked=(2, 3))
        _fill(ds, n_ng=4, n_ok=20)
        for x, y, t, sz, wd, cw in _all_text_items(ds):
            assert x + wd <= cw, f"{w}x{h}: {t!r} 超出畫布"


def test_no_text_is_drawn_twice(tmp_path):
    """同一個位置同一段文字不能畫兩遍（疊在一起肉眼看不出來，但表示排版跑了兩次）。"""
    ds = DebugSheet(make_cfg(tmp_path), tmp_path, ROOT, DebugContext(menu_variants=VARIANTS))
    _ticket_with_many_boxes(ds, n=400, linked=())
    _fill(ds, n_ng=2, n_ok=6)
    items = [(x, y, t) for x, y, t, *_ in _all_text_items(ds)]
    dup = {k for k in items if items.count(k) > 1}
    assert not dup, f"重複繪製: {sorted(dup)[:5]}"
