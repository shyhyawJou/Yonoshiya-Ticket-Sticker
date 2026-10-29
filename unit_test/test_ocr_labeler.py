"""
test_ocr_labeler.py
===================
完整幀的標註檔（utils/ocr_labeler.py）：每張 full_screen 全幀旁邊的同名 .json，裡面有這一幀的
mmrotate 偵測框 + OCR 文字框（座標都是全幀）。

最重要的兩個：
  * test_output_is_identical_to_the_real_autolabeling
      標準答案 fixtures/ocr_label_golden.json 是用**真的** PaddleOCR_tools/autolabeling.py 產生的
      （make_ocr_label_golden.py）；同樣的輸入給 ocr_labeler，要求文字框的結果逐位元組相同。
  * test_boxes_land_on_the_real_text_in_the_full_frame
      用真的 cv2.warpAffine 切一張旋轉的裁切圖、再經過 app 的轉正旋轉，在裁切圖裡找到已知的東西，
      換算回全幀，必須落在它在全幀真正的位置。座標換算錯了，標註整個都是歪的。

執行：
    py -3.12 -m pytest unit_test/test_ocr_labeler.py -q
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "unit_test" / "fixtures"))

import utils.ocr_labeler as ol  # noqa: E402
from fake_fuzz import FakeProcess  # noqa: E402
from utils.frame_writer import AsyncFrameWriter  # noqa: E402
from utils.skew_corrector import (  # noqa: E402
    affine_to_homogeneous, rotate90_ccw_matrix, rotate90_cw_matrix, rotate180_matrix,
)

GOLDEN = json.loads((ROOT / "unit_test" / "fixtures" / "ocr_label_golden.json").read_text(encoding="utf-8"))


class fake_rapidfuzz:
    """測試期間把 ocr_labeler 的 process 換成固定分數（見 fake_fuzz.py），結束一定換回來。"""

    def __enter__(self):
        self._orig = ol.process
        ol.process = FakeProcess
        return self

    def __exit__(self, *exc):
        ol.process = self._orig


def _menus(tmp_path):
    p = tmp_path / "menus_mapping.yaml"
    p.write_text(GOLDEN["menu_yaml"], encoding="utf-8")
    return p


def _label_for(case, menus):
    shapes = ol.build_shapes([np.array(b, dtype=np.float32) for b in case["boxes"]], case["texts"], case["confs"],
                             menus, case["category"], GOLDEN["match_threshold"])
    w, h = case["wh"]
    return json.loads(json.dumps(ol.build_label(case["file"], h, w, shapes)))      # 經過 JSON 一圈，跟寫到檔案的一樣


# ---------------------------------------------------------------------------
# 與真的 autolabeling.py 一致
# ---------------------------------------------------------------------------
def test_output_is_identical_to_the_real_autolabeling(tmp_path):
    with fake_rapidfuzz():
        menus = ol.load_menus(str(_menus(tmp_path)))
        for case in GOLDEN["cases"]:
            got = _label_for(case, menus)
            assert got == case["expected_label"], f"{case['name']} 與真的 autolabeling.py 的輸出不同"
            # dict 相等不看 key 順序；X-AnyLabeling 與人看 JSON 都在意順序，所以另外比
            assert list(got) == list(case["expected_label"])
            for a, b in zip(got["shapes"], case["expected_label"]["shapes"]):
                assert list(a) == list(b) and list(a["attributes"]) == list(b["attributes"])


def test_golden_actually_exercises_the_interesting_branches():
    """標準答案本身要涵蓋這些情況，否則『一致』沒有意義。"""
    shapes = [s for c in GOLDEN["cases"] for s in c["expected_label"]["shapes"]]
    assert any(s["attributes"]["auto_matched"] for s in shapes) and any(not s["attributes"]["auto_matched"] for s in shapes)
    assert {"rectangle", "polygon"} <= {s["shape_type"] for s in shapes}
    assert any(s["attributes"]["menu_context"] for s in shapes), "有帶 menu_context 的"
    assert any(c["expected_label"]["shapes"] == [] for c in GOLDEN["cases"]), "沒有文字的案例"
    assert any(s["attributes"]["menu_match_score"] == GOLDEN["match_threshold"] and s["attributes"]["auto_matched"]
               for s in shapes), "分數剛好等於門檻（要算比對成功）"


def test_normalize_text_replacements_match_autolabeling():
    n = lambda t: ol.normalize_text(t, lower=False, ignore_space=True)  # noqa: E731
    assert n("牛丼ﾂﾕﾀﾞｸ小") == "牛丼ツユダク小"            # NFKC：半形片假名 -> 全形
    assert n("牛丼･並") == "牛丼・並" and n("a·b") == "a・b"
    assert n("※変更しじみ") == "変更しじみ"                # 刪 ※
    assert n("★チーズ") == "☆チーズ" and n("≠") == "キ"
    assert n("增变黑燒烧醬") == "増変黒焼焼醤"
    assert n("ＡＢＣ１２３") == "ABC123", "不轉小寫（logic/text_utils 那一版會轉）"
    assert n(" 1 セット ") == "1セット"


def test_the_label_normalizer_is_not_the_projects_normalizer():
    """兩個 normalize_text 不一樣是刻意的。如果哪天有人想『合併』它們，這個測試要他先想清楚。"""
    from logic.text_utils import normalize_text as project_norm
    assert ol.normalize_text("※★≠ＡＢＣ", ignore_space=True) != project_norm("※★≠ＡＢＣ", ignore_space=True)


GOOD_EXAMPLE_TOP = ["version", "flags", "checked", "shapes", "imagePath", "imageData", "imageHeight", "imageWidth", "description"]
GOOD_EXAMPLE_SHAPE = ["label", "score", "points", "group_id", "description", "difficult", "shape_type", "flags", "attributes", "kie_linking"]
GOOD_EXAMPLE_ATTRS = ["category", "auto_matched", "menu_match_score", "raw_ocr_text", "menu_context"]


def test_format_matches_good_example():
    """欄位與順序取自 good_example.zip 裡的 20260919_102943_892__order_ticket__1.json。"""
    with fake_rapidfuzz():
        label = _label_for(GOLDEN["cases"][0], ol.EMPTY_MENUS)
    assert list(label) == GOOD_EXAMPLE_TOP and label["version"] == "4.0.0-beta.9" and label["imageData"] is None
    s = label["shapes"][0]
    assert list(s) == GOOD_EXAMPLE_SHAPE and list(s["attributes"]) == GOOD_EXAMPLE_ATTRS
    assert s["label"] == "text" and s["group_id"] == 0 and s["kie_linking"] == [] and s["flags"] is None
    assert len(s["points"]) == 4 and all(len(p) == 2 for p in s["points"])


def test_category_and_route_rules():
    assert [ol.category_for_task(t) for t in ("ticket", "ticket_extra", "sticker")] == ["ticket", "ticket", "sticker"]
    routes = {"order_ticket": "det", "order_sticker": "det"}
    assert ol.should_label(routes, "order_ticket") and ol.should_label(routes, "order_sticker")
    assert not ol.should_label({"order_ticket": "cv_table", "order_sticker": "det"}, "order_ticket"), "鈴茂傳票：框是儲存格"
    assert not ol.should_label(routes, None) and ol.should_label({}, "order_sticker"), "沒設路由 = 預設 det"


def test_matching_without_menus_marks_everything_difficult():
    with fake_rapidfuzz():
        shapes = ol.build_shapes([np.zeros((4, 2))], ["牛丼並"], [0.9], ol.EMPTY_MENUS, "ticket", 85.0)
    assert shapes[0]["difficult"] is True and shapes[0]["attributes"]["menu_match_score"] == 0.0


# ---------------------------------------------------------------------------
# 座標換算：OCR 的框（轉正後裁切圖座標）-> 全幀座標
# ---------------------------------------------------------------------------
def _crop_matrix(cx, cy, w, h, angle_deg):
    """複製 mmr_engine.Rotated_RTMDET.crop_by_angle 的 M（那個檔案要 NPU 才能 import）。w < h，不需要 90 度正規化。"""
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    M[0, 2] += (w / 2) - cx
    M[1, 2] += (h / 2) - cy
    return M


def _full_frame_scene():
    """一張 1200x900 的全幀，上面有一塊旋轉 25 度的『傳票』，傳票上有一個已知位置的黑方塊。"""
    W, H = 1200, 900
    frame = np.full((H, W, 3), 255, np.uint8)
    cx, cy, w, h, ang = 620.0, 430.0, 300, 560, 25.0
    M = _crop_matrix(cx, cy, w, h, ang)
    Minv = cv2.invertAffineTransform(M)
    # 在「裁切圖座標」的 (60,100)-(160,150) 畫一個黑方塊，用反矩陣放進全幀 -> 全幀真正的位置
    crop_rect = np.array([[60, 100], [160, 100], [160, 150], [60, 150]], np.float32)
    frame_rect = cv2.transform(crop_rect.reshape(-1, 1, 2), Minv).reshape(-1, 2)
    cv2.fillPoly(frame, [frame_rect.astype(np.int32)], (0, 0, 0))
    return frame, M, (w, h), frame_rect


def _find_black_box(img):
    mask = (img.max(axis=2) < 60).astype(np.uint8)
    pts = cv2.findNonZero(mask)
    box = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
    return box


def test_boxes_land_on_the_real_text_in_the_full_frame():
    """對 app 會遇到的四種情況（不轉 / 轉 90 度順時針 / 逆時針 / 180 度），用真的影像驗證換算。"""
    frame, M, (w, h), frame_rect = _full_frame_scene()
    crop = cv2.warpAffine(frame, M, (w, h))
    for name, code, mat_fn in (("不轉", None, None),
                               ("180 度", cv2.ROTATE_180, rotate180_matrix),
                               ("順時針 90 度", cv2.ROTATE_90_CLOCKWISE, rotate90_cw_matrix),
                               ("逆時針 90 度", cv2.ROTATE_90_COUNTERCLOCKWISE, rotate90_ccw_matrix)):
        # 跟 ocr_engine 的 det 路由一模一樣：T_total = 旋轉矩陣 @ M，M_inv_full = inv(T_total)
        t_total = affine_to_homogeneous(M)
        upright = crop
        if code is not None:
            t_total = mat_fn(w, h) @ t_total
            upright = cv2.rotate(crop, code)
        m_inv_full = np.linalg.inv(t_total)

        found = _find_black_box(upright)                  # OCR 會給的框：在轉正後裁切圖上看到的方塊
        (mapped,) = ol.boxes_to_frame([found], m_inv_full)

        # 兩個四邊形的角點順序可能不同：比「每個真實角點到最近的換算角點」的距離
        d = max(np.min(np.linalg.norm(mapped - p, axis=1)) for p in frame_rect.astype(np.float64))
        assert d < 2.5, f"{name}: 換算後的框離全幀裡真正的方塊最遠差 {d:.1f}px"


def test_boxes_to_frame_matches_the_overlay_way_and_rounds():
    """跟 manager/overlay_renderer.py 畫框用的同一個 cv2.transform（取 3x3 的前兩列），再四捨五入到 0.1px。"""
    m = np.array([[0.9, -0.4, 120.3], [0.4, 0.9, 80.7], [0, 0, 1]], np.float64)
    box = np.array([[10, 20], [110, 20], [110, 60], [10, 60]], np.float32)
    (got,) = ol.boxes_to_frame([box], m)
    want = cv2.transform(box.reshape(-1, 1, 2), m[:2, :]).reshape(4, 2)
    assert np.allclose(got, want, atol=0.051) and np.allclose(got * 10, np.round(got * 10)), "最多一位小數"
    assert json.dumps(got.tolist()).count(".") == 8 and all(len(str(v)) < 9 for v in got.ravel()), "不能有 float32 的長尾巴"
    assert ol.boxes_to_frame(None, m) == [] and ol.boxes_to_frame([], m) == []
    # 2x3 也接受（舊呼叫端）
    (got23,) = ol.boxes_to_frame([box], m[:2, :])
    assert np.array_equal(got, got23)


# ---------------------------------------------------------------------------
# mmrotate 偵測框
# ---------------------------------------------------------------------------
from utils.schema import Detection  # noqa: E402

DETS = [
    ("order_ticket", 0.93, (110.0, 40.0, 300.0, 55.2, 290.0, 400.0, 100.0, 385.0), (200.0, 220.0, 350.0, 190.0, 0.0796)),
    ("order_sticker", 0.88, (500.0, 100.0, 620.0, 100.0, 620.0, 160.0, 500.0, 160.0), (560.0, 130.0, 120.0, 60.0, 0.0)),
    ("sesame_front", 0.71, (30.0, 30.0, 80.0, 30.0, 80.0, 70.0, 30.0, 70.0), (55.0, 50.0, 50.0, 40.0, 0.0)),
]


def test_detection_shapes_use_the_same_fields_as_text_shapes_plus_direction():
    shapes = ol.detection_shapes(DETS)
    assert [s["label"] for s in shapes] == ["order_ticket", "order_sticker", "sesame_front"], "label = 偵測的類別名"
    with fake_rapidfuzz():
        text_shape = ol.build_shapes([np.zeros((4, 2))], ["x"], [0.9], ol.EMPTY_MENUS, "ticket", 85.0)[0]
    assert list(shapes[0]) == list(text_shape) + ["direction"], "欄位與順序跟文字框一樣；rotation 多一個 direction 放最後"
    s = shapes[0]
    assert s["score"] == 0.93 and s["shape_type"] == "rotation" and len(s["points"]) == 4
    assert s["direction"] == 0.0796, "direction = 偵測器的旋轉角 r（弧度）"
    assert s["points"][0] == [110.0, 40.0] and s["points"][2] == [290.0, 400.0], "角點就是偵測的 xyxy，沒有被換算"
    assert s["attributes"] == {"source": "mmrotate", "xywhr": [200.0, 220.0, 350.0, 190.0, 0.0796]}, "原始 xywhr 保留"
    assert s["difficult"] is False and s["group_id"] is None and s["description"] == ""


def _mmr_corners(x, y, w, h, r):
    """複製 mmr_engine.Rotated_RTMDET.xywhr2xyxyxyxy（那個檔案要 NPU 才能 import）：偵測器的 xyxy 就是這樣算的。"""
    ctr = np.array([x, y], np.float64)
    vec1 = np.array([w / 2 * np.cos(r), w / 2 * np.sin(r)])
    vec2 = np.array([-h / 2 * np.sin(r), h / 2 * np.cos(r)])
    return np.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2])


def test_rotation_direction_is_the_detector_angle_and_agrees_with_the_corners():
    """direction 直接取偵測器的 r（不是從角點反推）。用偵測器自己的公式造角點，確認 r 與角點方向一致。"""
    for x, y, w, h, r in [(500.0, 400.0, 300.0, 120.0, 0.35), (500.0, 400.0, 120.0, 300.0, -0.7),
                          (100.0, 100.0, 50.0, 50.0, 0.0), (2000.0, 1500.0, 700.0, 1500.0, 1.2)]:
        xyxy = tuple(_mmr_corners(x, y, w, h, r).flatten())
        (s,) = ol.detection_shapes([("order_ticket", 0.9, xyxy, (x, y, w, h, r))])
        assert s["shape_type"] == "rotation" and s["direction"] == round(r, 6)
        pts = np.array(s["points"])
        assert np.allclose(pts.mean(axis=0), [x, y], atol=0.06), "4 個角點的中心 = 偵測框中心"
        assert abs(np.linalg.norm(pts[1] - pts[0]) - h) < 0.2 and abs(np.linalg.norm(pts[2] - pts[1]) - w) < 0.2
        edge = pts[2] - pts[1]                                 # pt2 -> pt3 = -w * (cos r, sin r)
        assert np.allclose(edge / np.linalg.norm(edge), [-np.cos(r), -np.sin(r)], atol=0.01), "角點方向與 direction 一致"
        assert s["attributes"]["xywhr"] == [round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(r, 4)]


def test_detection_shapes_ignore_malformed_entries_and_empty_input():
    assert ol.detection_shapes([]) == [] and ol.detection_shapes(None) == []
    assert ol.detection_shapes([("x", 0.9, (1, 2, 3), (0, 0, 1, 1, 0))]) == [], "角點不是 8 個數字就略過"


# ---------------------------------------------------------------------------
# 背景輸出：JSON 放在全幀旁邊、同檔名
# ---------------------------------------------------------------------------
def _cfg(**kw):
    base = dict(enabled=True, match_threshold=85.0, image_wait_sec=3.0, max_queue=50)
    base.update(kw)
    return SimpleNamespace(**base)


def _frame_jpg(folder: Path, name="20261028_052109_217_sticker.jpg", w=800, h=600):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    cv2.imwrite(str(p), np.full((h, w, 3), 180, np.uint8))
    return p


FRAME_BOXES = [np.array([[100.5, 50.0], [300.0, 50.0], [300.0, 90.0], [100.5, 90.0]]),     # 正的 -> rectangle
               np.round(cv2.boxPoints(((150.0, 140.0), (100.0, 30.0), 10.0)).astype(np.float64), 1)]   # 傾斜 10 度 -> polygon
REC = [("牛丼･並", 0.97), ("690", 0.99)]


def test_json_is_written_next_to_the_frame_with_the_same_name(tmp_path):
    img = _frame_jpg(tmp_path / "20261028" / "full_screen")
    with fake_rapidfuzz():
        w = ol.AsyncLabelWriter(_cfg(), str(_menus(tmp_path)))
        assert w.submit(img, (800, 600), FRAME_BOXES, REC, "ticket")
        w.stop(wait=True)
    names = sorted(p.name for p in img.parent.iterdir())
    assert names == ["20261028_052109_217_sticker.jpg", "20261028_052109_217_sticker.json"], "同名、同資料夾，X-AnyLabeling 直接配對"
    label = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    assert label["imagePath"] == img.name and (label["imageWidth"], label["imageHeight"]) == (800, 600), "尺寸是全幀的"
    assert label["shapes"][0]["points"][0] == [100.5, 50.0], "框就是給進去的全幀座標，沒有被動過"
    assert label["shapes"][0]["description"] == "牛丼･並" and label["shapes"][0]["attributes"]["auto_matched"] is True
    assert label["shapes"][0]["attributes"]["category"] == "ticket"
    assert label["shapes"][1]["shape_type"] == "polygon" and label["shapes"][1]["difficult"] is True
    assert not list(img.parent.glob("*.tmp"))


def test_detections_are_written_even_without_any_ocr_text(tmp_path):
    """醬料包沒有 OCR，但有偵測：標註檔裡只有偵測框。"""
    img = _frame_jpg(tmp_path / "full_screen", "20261028_052109_217_sauce.jpg")
    w = ol.AsyncLabelWriter(_cfg(), "")
    assert w.submit(img, (800, 600), [], [], "ticket", detections=DETS)
    w.stop(wait=True)
    label = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    assert [s["label"] for s in label["shapes"]] == ["order_ticket", "order_sticker", "sesame_front"]
    assert all(s["attributes"]["source"] == "mmrotate" and s["shape_type"] == "rotation" and "direction" in s
               for s in label["shapes"])


def test_detections_come_first_then_the_ocr_text(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    with fake_rapidfuzz():
        w = ol.AsyncLabelWriter(_cfg(), str(_menus(tmp_path)))
        w.submit(img, (800, 600), FRAME_BOXES, REC, "ticket", detections=DETS)
        w.stop(wait=True)
    shapes = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))["shapes"]
    assert [s["label"] for s in shapes] == ["order_ticket", "order_sticker", "sesame_front", "text", "text"]
    assert shapes[3]["description"] == "牛丼･並" and set(shapes[3]["attributes"]) == set(GOOD_EXAMPLE_ATTRS), "文字框維持原格式"


def test_no_ai_result_at_all_writes_nothing(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    w = ol.AsyncLabelWriter(_cfg(), "")
    assert w.submit(img, (800, 600), [], [], "ticket", detections=[]) is False
    assert w.submit(img, (800, 600), [], [], "ticket") is False
    w.stop(wait=True)
    assert not list(tmp_path.rglob("*.json")), "沒有偵測、沒有文字，就沒有東西可以標"


def test_waits_for_the_frame_to_be_written_by_the_other_thread(tmp_path):
    folder = tmp_path / "full_screen"
    folder.mkdir()
    img = folder / "20261028_052109_217_ticket.jpg"
    w = ol.AsyncLabelWriter(_cfg(image_wait_sec=5.0), "")
    assert w.submit(img, (800, 600), FRAME_BOXES, REC, "ticket")          # 標註先到，圖還沒寫好
    time.sleep(0.4)
    assert not img.with_suffix(".json").exists(), "圖還沒出現，不能先寫 JSON"
    _frame_jpg(folder, img.name)                                            # 寫檔執行緒把圖寫好了
    w.stop(wait=True)
    assert img.with_suffix(".json").is_file()


def test_no_json_when_the_frame_never_appears(tmp_path):
    msgs = []
    saved, ol.logger = ol.logger, SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None,
                                                  warning=lambda m, *a, **k: msgs.append(m))
    try:
        w = ol.AsyncLabelWriter(_cfg(image_wait_sec=0.3), "")
        w.submit(tmp_path / "full_screen" / "20261028_052109_217_sticker.jpg", (800, 600), FRAME_BOXES, REC, "sticker")
        w.stop(wait=True)
    finally:
        ol.logger = saved
    assert not list(tmp_path.rglob("*.json")), "沒有圖就不留 JSON（沒有圖的標註沒用，還會讓人以為有圖）"
    assert any("沒有出現" in m for m in msgs)


def test_the_writer_never_blocks_the_caller(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    gate = threading.Event()
    w = ol.AsyncLabelWriter(_cfg(max_queue=2), "")
    real = w._write_one
    w._write_one = lambda job: (gate.wait(5), real(job))      # 卡住 worker，讓佇列塞滿
    t0 = time.perf_counter()
    results = [w.submit(img, (800, 600), FRAME_BOXES, REC, "sticker") for _ in range(6)]
    assert time.perf_counter() - t0 < 1.0, "submit 不能阻塞主迴圈"
    assert results[0] is True and results.count(False) >= 2 and w._dropped >= 2
    gate.set()
    w.stop(wait=True)


def test_disabled_does_nothing(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    w = ol.AsyncLabelWriter(_cfg(enabled=False), "")
    assert w.submit(img, (800, 600), FRAME_BOXES, REC, "sticker") is False and w._thread is None
    w.stop(wait=True)
    assert not list(tmp_path.rglob("*.json"))


def test_bad_input_is_handled(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    w = ol.AsyncLabelWriter(_cfg(), "")
    assert w.submit(None, (800, 600), FRAME_BOXES, REC, "sticker") is False, "沒有圖路徑"
    assert w.submit(img, (800, 600), FRAME_BOXES, [("a", 0.9)], "sticker")          # 框比文字多：只留兩邊都有的
    w.stop(wait=True)
    assert len(json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))["shapes"]) == 1


def test_a_frame_with_detections_but_no_text_still_gets_a_label_file(tmp_path):
    img = _frame_jpg(tmp_path / "full_screen")
    w = ol.AsyncLabelWriter(_cfg(), "")
    w.submit(img, (800, 600), [], [], "ticket", detections=DETS[:1])      # OCR 沒讀到字，但偵測有東西
    w.stop(wait=True)
    assert [s["label"] for s in json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))["shapes"]] == ["order_ticket"]


# ---------------------------------------------------------------------------
# 整合：app.py 真正的方法 + 真的寫檔器
# ---------------------------------------------------------------------------
APP_METHODS = {"_save_frame", "_frame_detections", "_save_full_screen", "_save_sauce_frame",
               "_submit_frame_label", "_submit_ocr_label"}


def _app_methods(tmp_path, ocr_routes=None, frame_queue=10, labeler_cfg=None, start_frame_worker=True):
    """app.py 沒辦法在沒有相機 / NPU 的環境 import，所以用 ast 把這幾個方法的原始碼取出來、綁到假的 self 上。
    跑的是 app.py 裡真正的程式碼，不是我另寫一份。"""
    import ast
    import queue
    import types
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    nodes = [n for c in ast.parse(src).body if isinstance(c, ast.ClassDef)
             for n in c.body if isinstance(n, ast.FunctionDef) and n.name in APP_METHODS]
    assert {n.name for n in nodes} == APP_METHODS
    errors, warns, infos = [], [], []
    ns = {"Path": Path, "should_label": ol.should_label, "category_for_task": ol.category_for_task,
          "boxes_to_frame": ol.boxes_to_frame,
          "logger": types.SimpleNamespace(info=lambda m, *a, **k: infos.append(m), warning=lambda m, *a, **k: warns.append(m),
                                          error=lambda m, *a, **k: errors.append(m))}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "app.py(methods)", "exec"), ns)

    if start_frame_worker:
        frame_writer = AsyncFrameWriter(max_queue_size=frame_queue)
    else:    # 不啟動背景執行緒：佇列塞滿後 save() 會丟棄，用來測「這一幀沒存成」
        frame_writer = AsyncFrameWriter.__new__(AsyncFrameWriter)
        frame_writer._queue = queue.Queue(maxsize=frame_queue)
        frame_writer._running, frame_writer._dropped_count, frame_writer._known_dirs = True, 0, set()
    labeler = ol.AsyncLabelWriter(labeler_cfg or _cfg(), "")
    me = types.SimpleNamespace(
        save_dir=tmp_path, event_recorder=types.SimpleNamespace(date="20261028"),
        tmp_frame=np.full((600, 800, 3), 120, np.uint8), tmp_detections=_detections(),
        frame_writer=frame_writer, ocr_labeler=labeler,
        task=types.SimpleNamespace(cfg=types.SimpleNamespace(ocr_routes=ocr_routes or {"order_ticket": "det", "order_sticker": "det"})),
        _label_skipped=set())
    for name in APP_METHODS:
        setattr(me, name, types.MethodType(ns[name], me))
    me._logs = types.SimpleNamespace(errors=errors, warns=warns, infos=infos)
    return me, errors, warns


def _detections(*which):
    """跟 mmr_engine.detect() 一樣的真 Detection 物件（utils.schema.Detection）。"""
    chosen = [DETS[i] for i in (which or range(len(DETS)))]
    return [Detection(xyxy=x, xywhr=r, cls_id=i, cls_name=c, conf=np.float32(cf)) for i, (c, cf, x, r) in enumerate(chosen)]


def _finish(me):
    if hasattr(me.frame_writer, "_thread"):
        me.frame_writer._queue.join()
    me.ocr_labeler.stop(wait=True)
    if hasattr(me.frame_writer, "_thread"):
        me.frame_writer.stop(wait=True)


def _ocr_result():
    # crop 座標的框；M_INV_FULL 把「crop 座標 -> 全幀座標」平移 (+100, +200)
    boxes = [np.array([[5, 5], [50, 5], [50, 25], [5, 25]], np.float32)]
    return {"frame_crop": np.zeros((90, 60, 3), np.uint8), "dt_boxes": boxes, "rec_res": [("690", 0.99)]}


M_INV_FULL = np.array([[1, 0, 100], [0, 1, 200], [0, 0, 1]], np.float64)


def _metadata(ref, task_type="ticket", cls="order_ticket", **extra):
    return {"type": task_type, "cls_name": cls, "frame_ref": ref, "M_inv_full": M_INV_FULL, **extra}


def _labels(tmp_path):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "20261028" / "full_screen").glob("*.json")}


def test_integration_ocr_frame_has_the_detections_and_the_text_in_full_frame_coordinates(tmp_path):
    me, errors, _ = _app_methods(tmp_path)
    ref = me._save_full_screen("ticket")                               # 送 OCR 前：存全幀（同時記下這一幀的偵測結果）
    me._submit_ocr_label(_ocr_result(), _metadata(ref))                # OCR 結果回來：存標註
    _finish(me)
    assert not errors
    folder = tmp_path / "20261028" / "full_screen"
    jpgs = sorted(folder.glob("*.jpg"))
    assert len(jpgs) == 1 and jpgs[0].name.endswith("_ticket.jpg") and ref["path"] == jpgs[0]
    assert sorted(p.name for p in folder.iterdir()) == [jpgs[0].name, jpgs[0].stem + ".json"], "全幀旁邊同名 .json，沒有別的檔案"
    label = json.loads((folder / (jpgs[0].stem + ".json")).read_text(encoding="utf-8"))
    assert (label["imageWidth"], label["imageHeight"]) == (800, 600) == ref["wh"]
    assert [s["label"] for s in label["shapes"]] == ["order_ticket", "order_sticker", "sesame_front", "text"]
    assert label["shapes"][0]["points"][0] == [110.0, 40.0], "偵測框 = 偵測的角點，不經換算"
    assert label["shapes"][3]["points"][0] == [105.0, 205.0], "文字框：crop 的 (5,5) 經 M_inv_full 平移 (+100,+200) -> 全幀 (105,205)"


def test_integration_sauce_frame_gets_a_detections_only_label(tmp_path):
    """醬料包不走 OCR，沒有文字結果；但這一幀有 mmrotate 偵測，標註檔要有。"""
    me, errors, _ = _app_methods(tmp_path)
    me._save_sauce_frame("sauce")                                      # 狀態機下 OK/NG 結論時的 callback
    _finish(me)
    assert not errors
    (stem, label), = _labels(tmp_path).items()
    assert stem.endswith("_sauce") and (tmp_path / "20261028" / "full_screen" / f"{stem}.jpg").is_file()
    assert [s["label"] for s in label["shapes"]] == ["order_ticket", "order_sticker", "sesame_front"], "只有偵測框，沒有 text"


def test_integration_suzumo_cv_table_ticket_has_detections_but_no_text(tmp_path):
    """鈴茂傳票（cv_table）的 OCR 文字框不寫，但這一幀的 mmrotate 偵測結果照寫。"""
    me, errors, _ = _app_methods(tmp_path, ocr_routes={"order_ticket": "cv_table", "order_sticker": "det"})
    ref = me._save_full_screen("ticket")
    for _ in range(3):
        me._submit_ocr_label(_ocr_result(), _metadata(ref))
    _finish(me)
    assert not errors and me._label_skipped == {"order_ticket"}, "只講一次"
    assert sum("cv_table" in m for m in me._logs.infos) == 1
    (_, label), = _labels(tmp_path).items()
    assert "text" not in [s["label"] for s in label["shapes"]] and len(label["shapes"]) == 3


def test_integration_missing_matrix_still_writes_the_detections(tmp_path):
    """沒有 M_inv_full 就不能把文字框換成全幀座標：文字框不寫（硬寫會整個錯位），偵測框照寫。"""
    me, errors, warns = _app_methods(tmp_path)
    ref = me._save_full_screen("ticket")
    md = _metadata(ref)
    md.pop("M_inv_full")
    me._submit_ocr_label(_ocr_result(), md)
    _finish(me)
    assert not errors and sum("M_inv_full" in w for w in warns) == 1
    (_, label), = _labels(tmp_path).items()
    assert "text" not in [s["label"] for s in label["shapes"]] and len(label["shapes"]) == 3


def test_integration_each_frame_gets_its_own_detections(tmp_path):
    """傳票那一幀與貼紙那一幀偵測到的東西不同；標註不能串到別幀（用的是存圖當下的快照）。"""
    me, errors, _ = _app_methods(tmp_path)
    me.tmp_detections = _detections(0)                                  # 這一幀只偵測到傳票
    ref1 = me._save_full_screen("ticket")
    me.tmp_detections = _detections(1, 2)                               # 主迴圈已經進到下一幀：偵測到貼紙與醬料包
    ref2 = me._save_full_screen("sticker")
    assert [d[0] for d in ref1["dets"]] == ["order_ticket"], "快照：之後主迴圈換掉 tmp_detections 不影響第一幀"
    me._submit_ocr_label(_ocr_result(), _metadata(ref1))                # OCR 結果比較晚才回來
    me._submit_ocr_label(_ocr_result(), _metadata(ref2, "sticker", "order_sticker"))
    _finish(me)
    assert not errors
    by_tag = {stem.rsplit("_", 1)[1]: [s["label"] for s in lab["shapes"]] for stem, lab in _labels(tmp_path).items()}
    assert by_tag["ticket"] == ["order_ticket", "text"]
    assert by_tag["sticker"] == ["order_sticker", "sesame_front", "text"]


def test_integration_each_task_type_gets_the_right_text_category(tmp_path):
    me, errors, _ = _app_methods(tmp_path)
    for task_type, cls, tag in (("ticket", "order_ticket", "ticket"), ("sticker", "order_sticker", "sticker"),
                                ("ticket_extra", "order_ticket", "ticket-extra")):
        ref = me._save_full_screen(tag)
        me._submit_ocr_label(_ocr_result(), _metadata(ref, task_type, cls))
        time.sleep(0.003)
    _finish(me)
    assert not errors
    cats = sorted(s["attributes"]["category"] for lab in _labels(tmp_path).values() for s in lab["shapes"] if s["label"] == "text")
    assert cats == ["sticker", "ticket", "ticket"]


def test_integration_a_frame_dropped_by_the_writer_gets_no_label_and_is_counted(tmp_path):
    ol_warns = []
    saved, ol.logger = ol.logger, SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None,
                                                  warning=lambda m, *a, **k: ol_warns.append(m))
    try:
        me, errors, warns = _app_methods(tmp_path, frame_queue=1, start_frame_worker=False)
        first = me._save_full_screen("ticket")           # 佇列裝得下
        second = me._save_full_screen("sticker")         # 佇列滿了 -> 被丟棄
        assert first["path"] is not None and second["path"] is None
        me._submit_ocr_label(_ocr_result(), _metadata(second, "sticker", "order_sticker"))
        me._save_sauce_frame("sauce")                    # 醬料包那幀也被丟（佇列還是滿的）
        _finish(me)
    finally:
        ol.logger = saved
    assert me.ocr_labeler._no_frame == 2 and not errors
    assert any("全幀被寫檔佇列丟棄" in w for w in ol_warns)
    assert not list(tmp_path.rglob("*.json")), "沒有圖就沒有標註"


def test_integration_nothing_detected_and_no_text_writes_no_file(tmp_path):
    me, errors, _ = _app_methods(tmp_path)
    me.tmp_detections = []
    ref = me._save_full_screen("ticket")
    me._submit_ocr_label({"frame_crop": None, "dt_boxes": [], "rec_res": []}, _metadata(ref))
    _finish(me)
    assert not errors and not list(tmp_path.rglob("*.json"))


def test_integration_disabled_labeler_costs_nothing(tmp_path):
    me, errors, _ = _app_methods(tmp_path, labeler_cfg=_cfg(enabled=False))
    ref = me._save_full_screen("ticket")
    assert ref["dets"] == [], "停用時不蒐集偵測結果"
    me._submit_ocr_label(_ocr_result(), _metadata(ref))
    me._save_sauce_frame("sauce")
    _finish(me)
    assert not errors and not list(tmp_path.rglob("*.json"))


def test_integration_broken_input_never_raises(tmp_path):
    me, errors, _ = _app_methods(tmp_path)
    ref = me._save_full_screen("ticket")
    me._submit_ocr_label({}, _metadata(ref))                                         # 少 key
    me._submit_ocr_label(_ocr_result(), {"type": "ticket", "cls_name": "order_ticket"})   # 沒 frame_ref -> 靜靜略過
    me.tmp_detections = [SimpleNamespace(cls_name="x")]                              # 壞掉的偵測物件
    assert me._frame_detections() == [], "壞掉的偵測物件略過，不丟例外"
    _finish(me)
    assert len(errors) == 1 and "提交失敗" in errors[0]


# ---------------------------------------------------------------------------
# 設定與接線
# ---------------------------------------------------------------------------
def test_shipped_profiles_load_ocr_label_and_the_menus_path():
    import config
    for prof in ("yoshinoya", "suzumo"):
        cfg = config.load_config(str(ROOT / "tasks" / "ocr" / prof / "config.yaml"))
        assert cfg.ocr_label == config.OcrLabelCfg(), f"{prof}: 隨附的 config.yaml 應等於預設值"
        assert cfg.menus_mapping_path.endswith("menus_mapping.yaml") and Path(cfg.menus_mapping_path).is_file()


def test_app_wires_dispatch_result_sauce_and_shutdown():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "frame_ref = self._save_full_screen(full_screen_tag(task[\"type\"]))" in app
    assert "\"frame_ref\": frame_ref" in app, "全幀的路徑、大小與偵測結果要跟著 metadata 從送出帶到結果"
    assert app.index("logic.apply_ocr_result(") < app.index("self._submit_ocr_label(ocr_result, metadata)"), "結果套用之後才輸出標註"
    assert "self.ocr_labeler.stop(wait=True)" in app, "關機要把標註佇列寫完"
    body = app.split("def _submit_ocr_label", 1)[1].split("\n    def ", 1)[0]
    assert 'metadata["M_inv_full"]' in body and "boxes_to_frame(" in body, "文字框要用 M_inv_full 換成全幀座標"
    assert "M_inv\"]" not in body, "不能用只還原 crop 那一步的 M_inv"
    # 偵測結果：要在 mmr.detect 之後、logic.update 之前記下來（醬料包的 callback 在 logic.update 裡面被呼叫）
    assert app.index("out, detections = mmr.detect(") < app.index("self.tmp_detections = detections") \
        < app.index("logic.update(frame, detections)"), "tmp_detections 要在 detect 之後、logic.update 之前設定"
    assert "save_frame=self._save_sauce_frame" in app, "醬料包判定 -> 存全幀 + 輸出標註"
    assert "self.tmp_detections = []" in app.split("def _save_frame", 1)[0], "__init__ 要先宣告"


def test_frame_writer_is_unchanged_by_labeling():
    """標註檔不需要動寫檔器（丟幀邏輯與檔名都維持原樣）。"""
    src = (ROOT / "utils" / "frame_writer.py").read_text(encoding="utf-8")
    assert "current_time" in src.split("def save", 1)[1].split("def _worker", 1)[0]
    assert "frame_stem" not in src
