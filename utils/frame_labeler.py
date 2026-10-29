"""
frame_labeler.py
================
每存一張 full_screen 全幀，就在**同一個資料夾、同一個檔名**旁邊輸出一份標籤檔（X-AnyLabeling JSON，
格式見 good_example.zip）：

    full_screen/20261028_052109_217_sticker.jpg      全幀（AsyncFrameWriter 存的）
    full_screen/20261028_052109_217_sticker.json     這一幀的標籤（這個模組存的，檔名跟圖相同，X-AnyLabeling 直接配對）

標籤只有 **mmrotate 的偵測結果**：這一幀偵測到的每個物件（傳票、貼紙、盤子、醬料包…），`rotation` shape，
label = 類別名，4 個角點（已經是全幀座標）。**不含 OCR 的結果**：OCR 是另一條流程，標籤不需要它，
所以也不用等 OCR 結果回來 —— 存全幀的當下就能把標籤交給背景輸出。

四種全幀（傳票 / 貼紙 / 第二張傳票 / 醬料包）都有標籤；偵測不到任何東西就不寫檔。
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from loguru import logger

XANY_VERSION = "4.0.0-beta.9"


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
    * 其餘欄位與順序沿用 PaddleOCR_tools/autolabeling.py 的 shape 格式（label、score、points、group_id、description、
      difficult、shape_type、flags、attributes、kie_linking），rotation 多一個 direction 放在最後
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


class AsyncFrameLabelWriter:
    """背景執行緒：組標籤、寫 JSON。主迴圈只付出「丟進佇列」。

    - JSON 放在全幀旁邊、檔名同圖（<圖檔名去掉副檔名>.json）。
    - 全幀是 AsyncFrameWriter 另一條執行緒寫的，可能比這邊晚幾百毫秒才出現：這裡最多等 image_wait_sec 秒，
      等不到（寫檔佇列丟棄 / 寫檔失敗）就不寫這份標籤 —— 沒有圖的 JSON 沒有用，還會讓人以為有圖。
    - 佇列有上限；滿了丟掉並用 loguru 警告（每 50 次一則），不拖慢辨識。
    - stop() 會把佇列裡的寫完再結束。JSON 用「先寫 .tmp 再改名」。
    """

    def __init__(self, cfg):
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.image_wait_sec = float(getattr(cfg, "image_wait_sec", 10.0))
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max(1, int(getattr(cfg, "max_queue", 200))))
        self._dropped = 0
        self._no_frame = 0
        self._thread: Optional[threading.Thread] = None
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._worker, name="frame-label", daemon=True)
        self._thread.start()
        logger.info(f"[frame_label] 啟用：在 full_screen 的圖旁邊輸出同名 .json（mmrotate 偵測框），佇列上限 {self._queue.maxsize}")

    # -- 主迴圈呼叫 ------------------------------------------------------
    def note_missing_frame(self) -> None:
        """這一幀的全幀沒存成（寫檔佇列丟棄），所以沒有可以配對的圖，標籤跳過。"""
        self._no_frame += 1
        if self._no_frame % 50 == 1:
            logger.warning(f"[frame_label] 全幀被寫檔佇列丟棄，這一幀的標籤跳過（累積 {self._no_frame} 次）")

    def submit(self, image_path, frame_wh, detections: Sequence) -> bool:
        """丟一筆進佇列。detections 是這一幀的 mmrotate 結果，每項 (類別名, 信心, xyxy, xywhr)（見 detection_shapes）。
        沒有任何偵測就不寫。回傳 False = 沒收（停用、沒有偵測、或佇列滿了被丟棄）。"""
        if not self.enabled or self._thread is None or image_path is None:
            return False
        dets = list(detections or [])
        if not dets:
            return False
        job = {"image_path": Path(image_path), "wh": (int(frame_wh[0]), int(frame_wh[1])), "dets": dets}
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning(f"[frame_label] 佇列已滿，累積丟棄 {self._dropped} 筆標籤")
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
                logger.error(f"[frame_label] 輸出失敗（略過，不影響辨識）: {e!r}")
            finally:
                self._queue.task_done()

    def _write_one(self, job: dict) -> None:
        img: Path = job["image_path"]
        deadline = time.monotonic() + self.image_wait_sec
        while not img.is_file():
            if time.monotonic() >= deadline:
                logger.warning(f"[frame_label] {img.name} 在 {self.image_wait_sec:g} 秒內沒有出現"
                               f"（寫檔佇列丟棄或寫檔失敗），這份標籤不寫")
                return
            time.sleep(0.05)

        w, h = job["wh"]
        shapes = detection_shapes(job["dets"])
        if not shapes:
            return
        label = build_label(img.name, h, w, shapes)

        path = img.with_suffix(".json")
        tmp = img.with_name(img.stem + ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(label, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        logger.info(f"[frame_label] {path.name}  偵測框 {len(shapes)}")

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
