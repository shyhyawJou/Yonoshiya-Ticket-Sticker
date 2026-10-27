"""
test_full_screen_frames.py
==========================
full_screen 全幀：檔名標籤（為什麼存）+ 醬料包下結論時存全幀。

重點：
  1. 每張圖的檔名都帶「為什麼存」：ticket / sticker / ticket-extra（送 OCR 那一幀）、sauce（醬料包下結論那一幀）。
  2. 丟幀的邏輯**沒有改**：佇列滿了照樣丟、照樣計數；只是 save() 把結果（檔名或 None）回報給呼叫端。
  3. 醬料包 OK / NG 都存；存檔出錯不影響判定；沒設 save_frame 時照常運作。

執行：
    py -3.12 -m pytest unit_test/test_full_screen_frames.py -q
"""
from __future__ import annotations

import inspect
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unit_test"))

from logic.readers.sticker_matcher import StickerMatcher  # noqa: E402
from logic.text_utils import FuzzyMatcher  # noqa: E402
from logic.tray_state_machine import TrayStateMachine  # noqa: E402
from utils.frame_writer import AsyncFrameWriter, frame_filename, full_screen_tag  # noqa: E402

import test_debug_sheet as base  # noqa: E402  共用假的 bus / parser / tray 建構

UPLOADER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,200}$")      # aibox_upload.py 的 MEDIA_NAME_RE


# ---------------------------------------------------------------------------
# 檔名
# ---------------------------------------------------------------------------
def test_filename_carries_the_reason_tag():
    t = 1761148429.123
    plain = frame_filename(t)
    tagged = frame_filename(t, "sticker")
    assert re.fullmatch(r"\d{8}_\d{6}_123\.jpg", plain), "不帶標籤 = 舊格式，沒變"
    assert re.fullmatch(r"\d{8}_\d{6}_123_sticker\.jpg", tagged)
    assert tagged == plain[:-4] + "_sticker.jpg"


def test_tags_for_the_three_ocr_task_types_and_sauce():
    assert [full_screen_tag(x) for x in ("ticket", "sticker", "ticket_extra")] == ["ticket", "sticker", "ticket-extra"]
    for tag in ("ticket", "sticker", "ticket-extra", "sauce"):
        assert UPLOADER_NAME_RE.match(frame_filename(time.time(), tag)), "要符合 aibox_uploader 的檔名規則"


def test_odd_tags_are_sanitized_for_the_uploader():
    name = frame_filename(time.time(), "醬料包 x/y")
    assert UPLOADER_NAME_RE.match(name) and "/" not in name and " " not in name


def test_names_sort_chronologically_even_with_different_tags():
    """離線重放 / 人工翻看時，依檔名排序仍是時間順序（時間戳在最前面，標籤只是後綴）。"""
    ts = [1761148429.100, 1761148429.250, 1761148430.001, 1761148431.999]
    tags = ["ticket", "sauce", "sticker", "ticket-extra"]
    names = [frame_filename(t, g) for t, g in zip(ts, tags)]
    assert names == sorted(names)


# ---------------------------------------------------------------------------
# 寫檔器：丟幀邏輯沒變
# ---------------------------------------------------------------------------
def _writer_without_worker(max_size: int) -> AsyncFrameWriter:
    """不啟動背景 thread 的寫檔器：佇列不會被消化，才能確定性地測「滿了會怎樣」。"""
    import queue
    w = AsyncFrameWriter.__new__(AsyncFrameWriter)
    w._queue = queue.Queue(maxsize=max_size)
    w._running = True
    w._dropped_count = 0
    w._known_dirs = set()
    return w


def test_dropping_behaviour_is_unchanged_but_now_reported():
    w = _writer_without_worker(2)
    img = np.zeros((4, 4, 3), np.uint8)
    out = [w.save(img, Path("/tmp"), "f", tag="sticker") for _ in range(4)]
    assert out[0] and out[1], "沒滿：回傳預定檔名"
    assert out[2] is None and out[3] is None, "滿了：照舊丟棄，回傳 None"
    assert w._dropped_count == 2, "丟棄計數照舊"
    assert w._queue.qsize() == 2, "佇列大小（上限）照舊"


def test_save_without_tag_still_works_for_old_callers():
    w = _writer_without_worker(2)
    name = w.save(np.zeros((4, 4, 3), np.uint8), Path("/tmp"), "snapshot")     # 舊的呼叫方式（snapshot 指令）
    assert re.fullmatch(r"\d{8}_\d{6}_\d{3}\.jpg", name)


def test_real_write_uses_the_same_name_that_save_returned(tmp_path):
    w = AsyncFrameWriter(max_queue_size=5)
    try:
        name = w.save(np.full((20, 30, 3), 200, np.uint8), tmp_path, "full_screen", tag="sauce")
        w._queue.join()
    finally:
        w.stop(wait=True)
    assert (tmp_path / "full_screen" / name).is_file(), "save() 回傳的檔名 = 實際寫出的檔名"
    assert name.endswith("_sauce.jpg")


# ---------------------------------------------------------------------------
# 醬料包：下 OK / NG 結論時存全幀
# ---------------------------------------------------------------------------
def _sm_with_sauces(save_frame):
    matcher = StickerMatcher(menus_sticker=base.STICKERS, fuzzy_matcher=FuzzyMatcher(special_prefixes=["ねぎ"], prefix_len=2))
    sm = TrayStateMachine(
        bus=base.FakeBus(), order_parser=base.FakeParser([("牛丼･小", 2), ("サラダセット", 1)]),
        sticker_matcher=matcher, menu_variants=base.VARIANTS, n_settle_frame=2,
        object_class_items={"sesame": "ごまドレッシング", "wafu": "和風ドレッング"},
        wrong_item_confirm=2, save_frame=save_frame,
    )
    tray = base.make_tray()
    trays = {"t1": tray}
    base.ticket_ocr(sm, trays)                 # 傳票 -> 待核對：牛丼小×2、ごまドレッシング×1
    return sm, tray, trays


def _add_sauce(tray, cls, x0):
    s = base.add_sticker(tray, base.poly(x0, 120, x0 + 100, 180), cls=cls)
    s.stable_frames = 99
    return s


def test_sauce_ok_saves_a_full_frame_tagged_sauce():
    calls = []
    sm, tray, trays = _sm_with_sauces(calls.append)
    s = _add_sauce(tray, "sesame", 380)
    sm.resolve_known_class_items(trays)
    assert s.is_checked, "判定本身照常"
    assert calls == ["sauce"]


def test_sauce_ng_also_saves_a_full_frame():
    """醬料包 OK / NG 都要存：不在訂單內的醬料包（NG）一樣有那一幀。"""
    calls = []
    sm, tray, trays = _sm_with_sauces(calls.append)
    s = _add_sauce(tray, "wafu", 380)           # 和風ドレッング 不在傳票的待核對清單內
    sm.resolve_known_class_items(trays)
    assert s.is_wrong_item and calls == ["sauce"]


def test_one_frame_per_verdict_not_per_video_frame():
    """已經判定成功的醬料包不會每幀都存；沒到穩定門檻也不存。"""
    calls = []
    sm, tray, trays = _sm_with_sauces(calls.append)
    s = _add_sauce(tray, "sesame", 380)
    for _ in range(5):
        sm.resolve_known_class_items(trays)
    assert calls == ["sauce"], "OK 之後 is_checked，不再判定、不再存"
    unstable = base.add_sticker(tray, base.poly(500, 120, 600, 180), cls="sesame")
    unstable.stable_frames = 0
    sm.resolve_known_class_items(trays)
    assert calls == ["sauce"], "還沒穩定到門檻，不存"


def test_normal_ocr_stickers_do_not_trigger_the_sauce_save():
    calls = []
    sm, tray, trays = _sm_with_sauces(calls.append)
    a = base.add_sticker(tray, base.poly(120, 120, 220, 180))
    base.sticker_ocr(sm, trays, a, "牛丼小")
    sm.resolve_known_class_items(trays)
    assert calls == [], "一般貼紙的全幀是『送 OCR 時』由 app 存，不是這裡"


def test_a_failing_save_never_changes_the_verdict():
    def boom(tag):
        raise RuntimeError("磁碟滿了")
    sm, tray, trays = _sm_with_sauces(boom)
    s = _add_sauce(tray, "sesame", 380)
    sm.resolve_known_class_items(trays)
    assert s.is_checked and "ごまドレッシング" in tray.checked_items


def test_without_save_frame_nothing_changes():
    sm, tray, trays = _sm_with_sauces(None)
    s = _add_sauce(tray, "sesame", 380)
    sm.resolve_known_class_items(trays)
    assert s.is_checked


# ---------------------------------------------------------------------------
# 接線（LogicEngine / TaskContext / app.py）。這幾處沒辦法在沒有 NPU 的環境整個跑起來，
# 所以檢查簽名與原始碼，避免只改一半在機器上才炸。
# ---------------------------------------------------------------------------
def test_logic_engine_accepts_and_forwards_save_frame():
    le = base._import_logic_engine()
    assert "save_frame" in inspect.signature(le.LogicEngine.__init__).parameters
    src = inspect.getsource(le.LogicEngine.__init__)
    assert "self.save_frame = save_frame" in src and "save_frame=self.save_frame" in src


def test_task_context_and_app_wire_the_callback_everywhere():
    tc = (ROOT / "manager" / "task_context.py").read_text(encoding="utf-8")
    assert "save_frame=None" in tc.split("def load_task", 1)[1].split(")", 1)[0]
    assert tc.count("save_frame=self._save_frame_cb") == 2, "初始化與切換模式（switch_mode）兩處建 LogicEngine 都要帶"
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "save_frame=self._save_full_screen" in app
    assert 'self._save_full_screen(full_screen_tag(task["type"]))' in app, "送 OCR 的全幀要帶任務類型標籤"
    assert "full_screen_tag" in app.split("class ", 1)[0], "app.py 要 import full_screen_tag"
    assert app.count("/full_screen\"") == 1, "full_screen 只有一個寫入點（_save_full_screen）"
