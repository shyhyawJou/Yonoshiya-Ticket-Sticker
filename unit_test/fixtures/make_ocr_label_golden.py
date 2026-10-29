"""
make_ocr_label_golden.py
========================
用「真的 autolabeling.py」產生 test_ocr_labeler.py 的標準答案（ocr_label_golden.json）。

  python unit_test/fixtures/make_ocr_label_golden.py <PaddleOCR_tools 資料夾>

什麼時候要重跑：autolabeling.py / metrics.py 的標註邏輯有改（例如 normalize_text 增減取代規則、門檻判斷、
shape 欄位），先重跑這支，再看 test_ocr_labeler.py 哪裡不一致，就知道 ocr_labeler 要跟著改什麼。

做法：
  * 真的 autolabeling.load_menus / process_image / build_shapes / match_menu_text
  * 只把「跑 det + rec 的 PipelineEval」換成假的（直接回傳我們給的框與文字）
  * 只把 rapidfuzz 的 process 換成 fake_fuzz.FakeProcess（固定分數，見該檔說明）
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fake_fuzz import FakeProcess  # noqa: E402

MENU_YAML = """\
mapping:
  - name: 牛丼･並
    sub_items:
      - 牛丼並
  - name: 生野菜ｻﾗﾀﾞ･みそ汁ｾｯﾄ
    sub_items:
      - サラダ
      - みそ汁
  - name: 牛丼並ﾂﾕﾀﾞｸ
    sub_items:
      - 牛丼並ﾂﾕﾀﾞｸ
  - name: 牛丼･並
    sub_items:
      - 重複の貼紙は最初の1件が勝つ
  - name: 変更ご飯軽め
    sub_items:
      - ご飯少なめ
  - name: あああああああああああああああああいうえ
    sub_items:
      - 境界テスト貼紙
"""


def rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


CASES = [
    {
        "name": "ticket_mixed", "category": "ticket", "file": "20260919_102943_892__order_ticket__1.jpg", "wh": [351, 739],
        "boxes": [rect(217, 93, 279, 133), rect(30, 150, 80, 190), rect(20, 200, 200, 240),
                  [[40.5, 300.2], [210.7, 280.1], [214.0, 310.9], [44.0, 331.0]],       # 傾斜 -> polygon
                  rect(10, 400, 150, 430), rect(10, 440, 150, 470)],
        "texts": ["690", "数", "3牛丼･並", "※生野菜ｻﾗﾀﾞ･みそ汁ｾｯﾄ", "ＡＢＣ 增 变 黑 燒 烧 醬 ★ ≠ ·", ""],
        "confs": [0.99, 0.5, 0.97654321, 0.9, 0.123456, 0.0],
    },
    {
        "name": "sticker_matched", "category": "sticker", "file": "20260919_103000_100__order_sticker__1.jpg", "wh": [300, 120],
        "boxes": [rect(10, 20, 280, 90)],
        "texts": ["牛丼並ﾂﾕﾀﾞｸ"],
        "confs": [0.95],
    },
    {
        "name": "sticker_unmatched_and_context", "category": "sticker", "file": "20260919_103001_200__order_sticker__1.jpg", "wh": [300, 120],
        "boxes": [rect(10, 20, 150, 60), rect(10, 62, 150, 100)],
        "texts": ["サラタ", "ご飯軽め"],
        "confs": [0.8, 0.99],
    },
    {
        # 假的比對分數 = 100 * difflib ratio；兩個 20 字的字串共同 17 字 -> 2*17/40 = 0.85 -> 剛好 85.0。
        # 分數「等於」門檻要算比對成功（>=）；> 門檻的寫法會在這裡露餡。
        "name": "score_exactly_at_threshold", "category": "ticket", "file": "20260919_103003_400__order_ticket__1.jpg", "wh": [200, 100],
        "boxes": [rect(10, 10, 190, 40), rect(10, 50, 190, 80)],
        "texts": ["あああああああああああああああああかきく", "あああああああああああああああああいうえ"],
        "confs": [0.9, 0.9],
    },
    {
        "name": "no_text_found", "category": "ticket", "file": "20260919_103002_300__order_ticket__1.jpg", "wh": [200, 300],
        "boxes": [], "texts": [], "confs": [],
    },
]


def main(tool_dir: str):
    tool_dir = str(Path(tool_dir).resolve())
    sys.path.insert(0, tool_dir)

    # autolabeling.py 要 import evaluator.PipelineEval（跑模型）與 tqdm；這裡只需要它的標註邏輯
    class _Pipeline:
        def __init__(self, case):
            self.case = case

        def infer_single(self, img):
            return {"boxes": [np.array(b, dtype=np.float32) for b in self.case["boxes"]],
                    "texts": list(self.case["texts"]), "confs": list(self.case["confs"])}

    sys.modules["evaluator"] = types.SimpleNamespace(PipelineEval=object)
    if "tqdm" not in sys.modules:
        sys.modules["tqdm"] = types.SimpleNamespace(tqdm=lambda x, **k: x)
    import autolabeling as al

    al.process = FakeProcess

    out = {"_note": "由 make_ocr_label_golden.py 用真的 autolabeling.py 產生（假的 OCR 管線 + 固定比對分數）。"
                    "autolabeling 的標註邏輯有改時，重跑它。",
           "menu_yaml": MENU_YAML, "match_threshold": 85.0, "cases": []}

    with tempfile.TemporaryDirectory() as d:
        menu_path = os.path.join(d, "menus_mapping.yaml")
        Path(menu_path).write_text(MENU_YAML, encoding="utf-8")
        menus = al.load_menus(menu_path)
        for case in CASES:
            # detect_category 看路徑裡有沒有 sticker / ticket -> 放進對應名字的資料夾
            folder = os.path.join(d, f"imgs_{case['category']}")
            os.makedirs(folder, exist_ok=True)
            import cv2
            img_path = os.path.join(folder, case["file"])
            cv2.imwrite(img_path, np.full((case["wh"][1], case["wh"][0], 3), 200, np.uint8))
            label = al.process_image(img_path, _Pipeline(case), menus, 85.0)
            label["description"] = ""          # X-AnyLabeling 存檔會有；good_example 也有（見 ocr_labeler.build_label）
            out["cases"].append({**case, "expected_label": label})

    path = HERE / "ocr_label_golden.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"寫入 {path}（{len(out['cases'])} 個案例）")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
