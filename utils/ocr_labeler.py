"""
ocr_labeler.py
==============
每存一張 full_screen 全幀，就在**同一個資料夾、同一個檔名**旁邊輸出一份「完整幀的標註檔」
（X-AnyLabeling JSON，文字框的格式與 PaddleOCR_tools/autolabeling.py 產生的一樣，見 good_example.zip）：

    full_screen/20261028_052109_217_sticker.jpg      全幀（AsyncFrameWriter 存的）
    full_screen/20261028_052109_217_sticker.json     這一幀的標註（這個模組存的，檔名跟圖相同，X-AnyLabeling 直接配對）

這一幀有哪些 AI 結果，就寫哪些（兩種都在同一個 shapes 清單裡，座標都是整張全幀的座標）：

    * mmrotate 偵測：這一幀偵測到的每個物件（傳票、貼紙、盤子、醬料包…）。X-AnyLabeling 的 rotation 框：
      label = 類別名，4 個角點，direction = 偵測器的旋轉角 r（弧度）。醬料包沒有 OCR，只有這一種。
    * OCR：每個文字框（label = "text"）。**description = AI（rec 模型）的原始預測文字**，原樣、不正規化、
      不做 fuzzy、不換成菜單文字。菜單比對（fuzzy + mapping，做法同 autolabeling.py）的結果只當複查的參考，放在
      attributes 與 difficult：auto_matched、menu_match_score、menu_context、raw_ocr_text（正規化後的預測）。
      只有走 det 路由的類別才寫（鈴茂傳票是 cv_table，框是儲存格，不寫）。

兩種都沒有就不寫檔。imageWidth / imageHeight 是全幀的大小。

跟 autolabeling.py 的差別（刻意的）
----------------------------------
autolabeling.py 對一個資料夾的裁切圖「重新跑一次」det + cls + rec。這裡的 OCR 結果 app 已經算好了
（框、文字、信心），所以**直接拿來用，不再跑模型**（不占 NPU、不拖慢辨識），再把 autolabeling.py 之後
的那一半逐字搬過來：

    normalize_text  ->  菜單比對（rapidfuzz token_set_ratio >= 門檻就換成菜單原文，否則保留並標 difficult）
                    ->  組成 shape / attributes  ->  X-AnyLabeling JSON

這些函式的輸出由 unit_test/test_ocr_labeler.py 用「真的 autolabeling.py 產生的對照檔」逐字驗證
（description 除外：這裡放 AI 的原始預測，autolabeling.py 放菜單比對後的文字）。
框的座標換算（crop -> 全幀）用的是 OCR 完成時 metadata["M_inv_full"]，跟 overlay 畫框同一個矩陣。

注意：一幀只送一個 OCR 目標，所以 OCR 文字框只有「這次被讀的那個傳票 / 貼紙」的；但偵測框是這一幀所有的物件。
標註用的 normalize_text 跟 logic/text_utils.py 的**不是同一個**，見該函式的說明。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
from loguru import logger
from rapidfuzz import fuzz, process

XANY_VERSION = "4.0.0-beta.9"


# ---------------------------------------------------------------------------
# 文字正規化
# ---------------------------------------------------------------------------
def normalize_text(text: str, lower: bool = False, ignore_space: bool = False) -> str:
    """逐字複製自 PaddleOCR_tools/metrics.py::normalize_text（autolabeling.py 用的那一版）。

    **不要改**：標籤文字必須跟離線的 autolabeling.py 產生的一致。注意它跟 logic/text_utils.py 的
    normalize_text **不同**（這一版會刪 ※、把 ★ 換成 ☆、≠ 換成 キ、簡體字換成日文字）。
    """
    # text = unicodedata.normalize("NFC", text).strip()
    text = unicodedata.normalize("NFKC", text).strip()
    if lower:
        text = text.lower()
    if ignore_space:
        text = text.replace(" ", "")
    
    text = text.replace("･", "・")
    text = text.replace("·", "・")
    text = text.replace("增", "増")
    text = text.replace("变", "変")
    text = text.replace("★", "☆")
    text = text.replace("黑", "黒")
    text = text.replace("燒", "焼")
    text = text.replace("烧", "焼")
    text = text.replace("醬", "醤")
    text = text.replace("※", "")
    text = text.replace("≠", "キ")
    return text


# ---------------------------------------------------------------------------
# 菜單載入 / 比對（逐字照 autolabeling.py，只差在 print -> loguru、關鍵字清單預先算好）
# ---------------------------------------------------------------------------
def load_menus(menu_yaml: str) -> dict:
    """讀 menus_mapping.yaml 的 `mapping:`，建兩張查表：

      - `name`      -> 印在**傳票**上
      - `sub_items` -> 印在**貼紙**上（一行傳票可以產生好幾張貼紙）

    回傳 {"ticket": {"norm_map", "context", "keys"}, "sticker": {...}}。
    keys 是 norm_map 的鍵清單（每次比對都重建太浪費），不影響結果。
    """
    import yaml
    with open(menu_yaml, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entries = raw.get("mapping", []) or []
    ticket_norm, ticket_ctx = {}, {}
    sticker_norm, sticker_ctx = {}, {}

    for entry in entries:
        name = str(entry.get("name", "")).strip()
        subs = [str(s).strip() for s in (entry.get("sub_items", []) or []) if str(s).strip()]

        if name:
            norm_name = normalize_text(name, lower=False, ignore_space=True)
            # first occurrence wins if the same ticket text repeats in the yaml
            ticket_norm.setdefault(norm_name, name)
            ticket_ctx.setdefault(norm_name, subs)

        for s in subs:
            norm_s = normalize_text(s, lower=False, ignore_space=True)
            sticker_norm.setdefault(norm_s, s)
            sticker_ctx.setdefault(norm_s, name)

    logger.info(f"[ocr_label] menus: {len(ticket_norm)} 種傳票寫法、{len(sticker_norm)} 種貼紙寫法 ({menu_yaml})")
    return {
        "ticket": {"norm_map": ticket_norm, "context": ticket_ctx, "keys": list(ticket_norm.keys())},
        "sticker": {"norm_map": sticker_norm, "context": sticker_ctx, "keys": list(sticker_norm.keys())},
    }


EMPTY_MENUS = {
    "ticket": {"norm_map": {}, "context": {}, "keys": []},
    "sticker": {"norm_map": {}, "context": {}, "keys": []},
}


def match_menu_text(pred_norm: str, menus: dict, category: str, threshold: float):
    """把正規化後的 OCR 文字貼到菜單上最像的那一條。

    回傳 (final_text, score, matched, context)：
      - matched=True  -> final_text 是菜單原文（修掉 OCR 錯字）；context 是「另一邊」供人工檢查
                         （傳票 -> 它對應的貼紙 sub_items；貼紙 -> 它所屬的傳票品名）
      - matched=False -> final_text 就是正規化後的 OCR 文字（數字、編號之類不在菜單上的），留給人工看
    """
    table = menus.get(category, {})
    norm_map = table.get("norm_map", {})
    if not pred_norm or not norm_map:
        return pred_norm, 0.0, False, None

    keys = table.get("keys") or list(norm_map.keys())
    best = process.extractOne(pred_norm, keys, scorer=fuzz.token_set_ratio)
    if best is None:
        return pred_norm, 0.0, False, None
    best_key, score, _ = best
    if score >= threshold:
        return norm_map[best_key], float(score), True, table.get("context", {}).get(best_key)
    return pred_norm, float(score), False, None


# ---------------------------------------------------------------------------
# shape / label 組裝
# ---------------------------------------------------------------------------
def shape_type_for_box(box: np.ndarray, tol_deg: float = 3.0) -> str:
    """det 輸出的是任意四邊形（跟著文字傾斜）。幾乎正的才標 rectangle，否則要標 polygon，
    不然 X-AnyLabeling 畫出來會歪。"""
    rect = cv2.minAreaRect(box.astype(np.float32))
    angle = rect[-1] % 90.0
    is_aligned = angle <= tol_deg or angle >= (90.0 - tol_deg)
    return "rectangle" if is_aligned else "polygon"


def build_shapes(boxes: Sequence, texts: Sequence[str], confs: Sequence[float], menus: dict,
                 category: str, match_threshold: float) -> List[dict]:
    shapes = []
    for box, text, conf in zip(boxes, texts, confs):
        box = np.asarray(box)
        norm_text = normalize_text(text, lower=False, ignore_space=True)
        final_text, match_score, matched, context = match_menu_text(norm_text, menus, category, match_threshold)

        # description 放 AI 的原始預測（跟 autolabeling.py 不同：它放 final_text，也就是比對成功時換成菜單原文）。
        # 這樣複查時改的文字就是「模型錯在哪」，不會被菜單比對蓋掉；比對結果（final_text 的來源）仍算出來，
        # 只放在 attributes / difficult 當參考。
        shapes.append({
            "label": "text",
            "score": round(float(conf), 4),
            "points": [[float(x), float(y)] for x, y in box.tolist()],
            "group_id": 0,
            "description": str(text),
            "difficult": not matched,   # 沒比對上的標起來，人工複查時容易篩
            "shape_type": shape_type_for_box(box),
            "flags": None,
            "attributes": {
                "category": category,
                "auto_matched": matched,
                "menu_match_score": round(match_score, 1),
                "raw_ocr_text": norm_text,
                # 傳票這行 -> 它的貼紙 sub_items；貼紙這行 -> 它所屬的傳票品名
                "menu_context": context,
            },
            "kie_linking": [],
        })
    return shapes


def build_label(image_name: str, height: int, width: int, shapes: List[dict]) -> dict:
    return {
        "version": XANY_VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",   # autolabeling.py 沒寫，但 X-AnyLabeling 存檔一定有（good_example 就有）
    }


def detection_shapes(dets: Sequence) -> List[dict]:
    """mmrotate 的偵測結果 -> X-AnyLabeling 的 rotation shape。
    dets 的每一項是 (類別名, 信心, xyxy 8 個數字, xywhr 5 個數字)。

    * points    = xyxy：旋轉框的 4 個角點（mmr_engine.xywhr2xyxyxyxy 算的，順序 pt1~pt4 原樣保留；
                  已經是全幀座標，overlay 畫偵測框也用它），不需要換算。
    * direction = 偵測器的旋轉角 r（弧度），原樣放進去，不取負、不換算。
                  **單位與正負號都已確認**（使用者確認）：X-AnyLabeling 的 direction 單位是弧度、順時針為正；
                  偵測器的 r 也是弧度、順時針為正（影像座標 y 向下，r > 0 時寬軸往畫面下方轉，用 xywhr2xyxyxyxy
                  實際算過），兩邊單位與正負號都一樣，所以直接用。
                  **還沒確認的**：0 度時是不是「寬邊沿 +x」（跟偵測器 r = 0 的定義一樣）、角點順序要不要重排。
                  開發環境沒有 X-AnyLabeling 可以開來看；要改只改這一行（上機先開一張標註檔確認框的角度對不對）。
    * 其餘欄位與順序跟文字框的 shape 一樣（見 build_shapes），rotation 多一個 direction 放在最後
      （依記憶，X-AnyLabeling 存 rotation 時是在標準欄位之後多一個 direction；沒有對照過實際檔案）。
    * attributes.xywhr 保留偵測器原始的（中心、寬、高、弧度），要轉回 mmrotate 格式不用從角點反推。
    """
    shapes = []
    for cls_name, conf, xyxy, xywhr in dets or []:
        pts = np.asarray(xyxy, dtype=np.float64)
        if pts.size != 8:
            continue
        x, y, w, h, r = (float(v) for v in xywhr)
        shapes.append({
            "label": str(cls_name),
            "score": round(float(conf), 4),
            "points": np.round(pts.reshape(4, 2), 1).tolist(),
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rotation",
            "flags": None,
            "attributes": {"source": "mmrotate", "xywhr": [round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(r, 4)]},
            "kie_linking": [],
            "direction": round(r, 6),
        })
    return shapes


# ---------------------------------------------------------------------------
# 跟 app 銜接的小工具
# ---------------------------------------------------------------------------
def category_for_task(task_type: str) -> str:
    """OCR 任務類型 -> 菜單比對的類別。sticker 比對貼紙寫法；ticket / ticket_extra 都是傳票。"""
    return "sticker" if task_type == "sticker" else "ticket"


def should_label(ocr_routes: Dict[str, str], cls_name: Optional[str]) -> bool:
    """只有走 `det` 路由的類別才輸出標註。

    標註檔是給 PaddleOCR 訓練 / 複查用的「文字行框 + 文字」，跟 autolabeling.py 一樣。
    `cv_table`（鈴茂傳票）的框是表格儲存格、不是 det 的文字行框，混進去會汙染資料集。
    """
    if not cls_name:
        return False
    return (ocr_routes or {}).get(cls_name, "det") == "det"


def boxes_to_frame(boxes: Sequence, m_inv_full) -> List[np.ndarray]:
    """OCR 的框（轉正後裁切圖的座標）-> 整張全幀的座標。

    m_inv_full 是 OCR 完成時 metadata["M_inv_full"]（3x3）：ocr_engine 把「crop_by_angle 的 M」再串上
    OCR 為了轉正做的 90 / 180 度旋轉、取反矩陣得到的，**不是**只還原 crop 那一步的 metadata["M_inv"]。
    manager/overlay_renderer.py 畫 OCR 框用的就是它，這裡用同一個、同一個 cv2.transform 呼叫方式，
    標註才會跟畫面上看到的框重疊。座標四捨五入到 0.1 px。
    """
    m = np.asarray(m_inv_full, dtype=np.float64)
    affine = m[:2, :] if m.shape == (3, 3) else m
    out = []
    for b in boxes if boxes is not None else []:
        pts = np.asarray(b, dtype=np.float32).reshape(-1, 1, 2)
        out.append(np.round(cv2.transform(pts, affine).reshape(-1, 2).astype(np.float64), 1))
    return out


# ---------------------------------------------------------------------------
# 背景輸出
# ---------------------------------------------------------------------------
class AsyncLabelWriter:
    """背景執行緒：組標註（菜單比對）、寫 JSON。主迴圈只付出「丟進佇列」。

    - JSON 放在全幀旁邊、檔名同圖（<圖檔名去掉副檔名>.json）。
    - 全幀是 AsyncFrameWriter 另一條執行緒寫的，可能比標註晚幾百毫秒才出現：這裡最多等 image_wait_sec 秒，
      等不到（寫檔佇列丟棄 / 寫檔失敗）就不寫這份標註 —— 沒有圖的 JSON 沒有用，還會讓人以為有圖。
    - 佇列有上限；滿了丟掉並用 loguru 警告（每 50 次一則），不拖慢辨識。
    - stop() 會把佇列裡的寫完再結束。JSON 用「先寫 .tmp 再改名」。
    """

    def __init__(self, cfg, menus_mapping_path: str = ""):
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.threshold = float(getattr(cfg, "match_threshold", 85.0))
        self.image_wait_sec = float(getattr(cfg, "image_wait_sec", 10.0))
        self.menus = EMPTY_MENUS
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max(1, int(getattr(cfg, "max_queue", 200))))
        self._dropped = 0
        self._no_frame = 0
        self._thread: Optional[threading.Thread] = None
        if not self.enabled:
            return
        if menus_mapping_path and os.path.isfile(menus_mapping_path):
            try:
                self.menus = load_menus(menus_mapping_path)
            except Exception as e:
                logger.error(f"[ocr_label] 讀不了 menus_mapping（{menus_mapping_path}）：{e!r}；所有文字都會標 difficult")
        else:
            logger.warning(f"[ocr_label] 找不到 menus_mapping（{menus_mapping_path!r}）；所有文字都不做菜單比對、標 difficult")
        self._thread = threading.Thread(target=self._worker, name="ocr-label", daemon=True)
        self._thread.start()
        logger.info(f"[ocr_label] 啟用：在 full_screen 的圖旁邊輸出同名 .json，比對門檻 {self.threshold:g}，"
                    f"佇列上限 {self._queue.maxsize}")

    # -- 主迴圈呼叫 ------------------------------------------------------
    def note_missing_frame(self) -> None:
        """這次 OCR 的全幀沒存成（寫檔佇列丟棄），所以沒有可以配對的圖，標註跳過。"""
        self._no_frame += 1
        if self._no_frame % 50 == 1:
            logger.warning(f"[ocr_label] 全幀被寫檔佇列丟棄，這次 OCR 的標註跳過（累積 {self._no_frame} 次）")

    def submit(self, image_path, frame_wh, boxes: Sequence, rec_res, category: str, detections=None) -> bool:
        """丟一筆進佇列。boxes（OCR 文字框）必須已經是全幀座標（見 boxes_to_frame）；沒有 OCR 就給空的。
        detections 是這一幀的 mmrotate 結果，每項 (類別名, 信心, xyxy, xywhr)（見 detection_shapes）。
        偵測與文字都沒有就不寫。回傳 False = 沒收（停用、沒有任何 AI 結果、或佇列滿了被丟棄）。"""
        if not self.enabled or self._thread is None or image_path is None:
            return False
        boxes = list(boxes) if boxes is not None else []
        texts = [str(r[0]) for r in (rec_res or [])]
        confs = [float(r[1]) for r in (rec_res or [])]
        n = min(len(boxes), len(texts))
        dets = list(detections or [])
        if n == 0 and not dets:
            return False
        job = {
            "image_path": Path(image_path), "wh": (int(frame_wh[0]), int(frame_wh[1])),
            "boxes": [np.asarray(b, dtype=np.float64) for b in boxes[:n]],
            "texts": texts[:n], "confs": confs[:n], "category": category, "dets": dets,
        }
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning(f"[ocr_label] 佇列已滿，累積丟棄 {self._dropped} 筆標註")
            return False
        return True

    # -- 背景 ------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._write_one(job)
            except Exception as e:
                logger.error(f"[ocr_label] 輸出失敗（略過，不影響辨識）: {e!r}")
            finally:
                self._queue.task_done()

    def _write_one(self, job: dict) -> None:
        img: Path = job["image_path"]
        deadline = time.monotonic() + self.image_wait_sec
        while not img.is_file():
            if time.monotonic() >= deadline:
                logger.warning(f"[ocr_label] {img.name} 在 {self.image_wait_sec:g} 秒內沒有出現"
                               f"（寫檔佇列丟棄或寫檔失敗），這份標註不寫")
                return
            time.sleep(0.05)

        w, h = job["wh"]
        det_shapes = detection_shapes(job["dets"])
        text_shapes = build_shapes(job["boxes"], job["texts"], job["confs"], self.menus, job["category"], self.threshold)
        shapes = det_shapes + text_shapes
        label = build_label(img.name, h, w, shapes)

        path = img.with_suffix(".json")
        tmp = img.with_name(img.stem + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

        n = len(text_shapes)
        matched = sum(1 for s in text_shapes if s["attributes"]["auto_matched"])
        logger.info(f"[ocr_label] {path.name}  偵測框 {len(det_shapes)}／文字框 {n}"
                    f"（菜單比對成功 {matched}／標 difficult {n - matched}）")

    def stop(self, wait: bool = True, timeout: float = 10.0) -> None:
        """把佇列裡的寫完再結束。"""
        t = self._thread
        if t is None or not t.is_alive():
            return
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            return
        if wait:
            t.join(timeout=timeout)
