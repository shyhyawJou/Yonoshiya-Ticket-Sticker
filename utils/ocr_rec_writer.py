"""
ocr_rec_writer.py
=================
每筆訂單，把每次 OCR 裡「rec 實際讀的每個框小圖」另存下來，最終的辨識結果直接畫在小圖上。

用途：debug sheet 上畫的是整張轉正後的圖，但 rec 讀的不是那張整圖，而是每個偵測框各自裁出來（再經過 cls 單獨判 180 度、
必要時只翻那一個框）的小圖。整張圖看起來是正的、某個數字卻辨識錯誤時，要看的是 rec 當下吃進去的那張小圖。

存哪裡、存什麼
--------------
    <save_dir>/<日期>/ocr_rec/<訂單>/<時分秒>_<類型><序號>_<框編號>.jpg

  * <訂單>：`No<單號>`（跟訂單照片組的標籤同寫法）；這筆 OCR 套用完還不知道單號就用 `tray<id>`
  * <類型>：ticket / ticket_extra / sticker …（OCR 的 type）
  * <序號>：log 上 `[OCR#n]` 的 n（跟 debug sheet JSON 的 ocr_seq 是同一個數字，可以對照）
  * <框編號>：這次讀取最後採用的結果裡，第幾個框（跟 debug sheet 畫的框同順序）
  * 圖：原解析度的小圖（不縮放）+ 底下一條白底：最終辨識文字、信心、類型 / 序號 / 框編號。
    文字就是 debug sheet 上呈現的那份（OCR 引擎最後採用的 rec_res），不含過程（翻轉比對、重讀等）

背景執行緒負責畫文字與寫檔，主迴圈只付出「丟進佇列」。佇列滿了整筆丟棄並警告（不拖慢辨識）；
每筆訂單最多存 max_per_order 張，超過就不再存並警告一次。
"""
from __future__ import annotations

import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except Exception:                                   # 沒裝 Pillow：文字退回 ASCII（日文會變 ?）
    _HAS_PIL = False

DEFAULT_FONT_REL = "tasks/ocr/NotoSansJP-Bold.otf"   # 跟 utils/debug_sheet.py 同一個字型

PAD = 8                  # 底部說明列的內距
MIN_W = 300              # 小圖很窄時，畫布至少這麼寬，文字才放得下
MAX_W = 1800
FS_TEXT, FS_META = 28, 18
C_BG, C_TEXT, C_META, C_RULE = (255, 255, 255), (20, 20, 20), (105, 105, 105), (190, 190, 190)   # RGB

_SAFE = re.compile(r"[^0-9A-Za-z_-]+")


def safe_name(s) -> str:
    """檔名 / 資料夾名只留 ASCII 字母數字與 _ -（單號可能含全形、空白）。"""
    return _SAFE.sub("_", str(s)).strip("_") or "x"


class _Fonts:
    def __init__(self, font_path: Optional[str]):
        self.path = font_path if (font_path and os.path.isfile(font_path)) else None
        self.ok = bool(_HAS_PIL and self.path)
        self._cache: Dict[int, object] = {}
        if not self.ok:
            logger.warning(f"[ocr_rec] 找不到可用的日文字型（PIL={_HAS_PIL}, font={font_path}），文字退回 ASCII，日文會顯示成 '?'")

    def get(self, size: int):
        f = self._cache.get(size)
        if f is None:
            f = ImageFont.truetype(self.path, size)
            self._cache[size] = f
        return f

    def width(self, s: str, size: int) -> int:
        if self.ok:
            return int(self.get(size).getlength(s))
        return int(len(s) * size * 0.6)

    def clip(self, s: str, size: int, max_w: int) -> str:
        if self.width(s, size) <= max_w:
            return s
        while s and self.width(s + "…", size) > max_w:
            s = s[:-1]
        return s + "…"


def render_rec_crop(crop: np.ndarray, text: str, conf: float, meta: str, fonts: _Fonts) -> np.ndarray:
    """小圖原樣放在上面，底下接一條白底說明列：最終辨識文字 + 信心 / meta。回傳 BGR。"""
    img = crop
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    line1 = text if text != "" else "（空）"
    line2 = f"conf {conf:.3f}   {meta}"
    w1, w2 = fonts.width(line1, FS_TEXT), fonts.width(line2, FS_META)
    W = int(min(MAX_W, max(MIN_W, w, w1 + 2 * PAD, w2 + 2 * PAD)))
    lh1, lh2 = int(FS_TEXT * 1.25), int(FS_META * 1.25)
    band_h = PAD + lh1 + 4 + lh2 + PAD
    canvas = np.full((h + band_h, W, 3), 255, np.uint8)
    canvas[0:h, 0:w] = img[:, :min(w, W)]
    canvas[h:h + 1, :] = C_RULE[::-1]                                  # 小圖與說明列之間一條細線
    line1 = fonts.clip(line1, FS_TEXT, W - 2 * PAD)
    line2 = fonts.clip(line2, FS_META, W - 2 * PAD)
    y1, y2 = h + PAD, h + PAD + lh1 + 4
    if fonts.ok:
        band = Image.fromarray(cv2.cvtColor(canvas[h:], cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(band)
        draw.text((PAD, y1 - h), line1, font=fonts.get(FS_TEXT), fill=C_TEXT)
        draw.text((PAD, y2 - h), line2, font=fonts.get(FS_META), fill=C_META)
        canvas[h:] = cv2.cvtColor(np.asarray(band), cv2.COLOR_RGB2BGR)
    else:
        ascii1 = line1.encode("ascii", "replace").decode("ascii")
        ascii2 = line2.encode("ascii", "replace").decode("ascii")
        cv2.putText(canvas, ascii1, (PAD, y1 + int(FS_TEXT * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, FS_TEXT / 36.0, C_TEXT[::-1], 1, cv2.LINE_AA)
        cv2.putText(canvas, ascii2, (PAD, y2 + int(FS_META * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, FS_META / 36.0, C_META[::-1], 1, cv2.LINE_AA)
    return canvas


class AsyncRecCropWriter:
    def __init__(self, cfg, project_dir):
        self.enabled = bool(getattr(cfg, "enabled", False))
        self.max_per_order = max(1, int(getattr(cfg, "max_per_order", 3000)))
        self._queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=max(1, int(getattr(cfg, "max_queue", 400))))
        self._counts: Dict[str, int] = {}                # 訂單資料夾 -> 已收的張數（只在主迴圈 submit 時動）
        self._full_warned: set = set()
        self._dropped = 0
        self._fail = 0
        self._thread: Optional[threading.Thread] = None
        if not self.enabled:
            return
        fp = str(getattr(cfg, "font_path", "") or "") or str(Path(project_dir) / DEFAULT_FONT_REL)
        self._fonts = _Fonts(fp)
        self._thread = threading.Thread(target=self._worker, name="ocr-rec-writer", daemon=True)
        self._thread.start()
        logger.info(f"[ocr_rec] 啟用：每筆訂單另存 rec 實際讀的每個框小圖（每筆訂單最多 {self.max_per_order} 張），"
                    f"佇列上限 {self._queue.maxsize}")

    # -- 主迴圈呼叫 ------------------------------------------------------
    def submit(self, folder, order_key: str, kind: str, seq: Optional[int],
               crops: Sequence[np.ndarray], rec_res: Sequence) -> int:
        """丟一次 OCR 的所有框小圖進佇列。回傳收了幾張（0 = 沒收）。

        crops / rec_res 同順序（OCR 引擎保證，見 TextSystemDLA.rec_crops_for）。文字與信心在這裡就取出來，
        之後 rec_res 被別人改了也不影響。
        """
        if not self.enabled or self._thread is None or not crops or rec_res is None:
            return 0
        order_dir = Path(folder) / safe_name(order_key)
        used = self._counts.get(str(order_dir), 0)
        room = self.max_per_order - used
        if room <= 0:
            if str(order_dir) not in self._full_warned:
                self._full_warned.add(str(order_dir))
                logger.warning(f"[ocr_rec] {order_dir.name} 已經存了 {used} 張，到達 max_per_order，之後這筆訂單不再存")
            return 0
        items: List[Tuple[np.ndarray, str, float]] = []
        for crop, rec in zip(crops, rec_res):
            if crop is None or getattr(crop, "size", 0) == 0:
                continue
            try:
                items.append((crop, str(rec[0]), float(rec[1])))
            except Exception:
                continue
        items = items[:room]
        if not items:
            return 0
        job = {"dir": order_dir, "stamp": time.strftime("%H%M%S"), "kind": safe_name(kind or "ocr"),
               "seq": seq, "items": items}
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 20 == 1:
                logger.warning(f"[ocr_rec] 佇列已滿，累積丟棄 {self._dropped} 次 OCR 的小圖")
            return 0
        self._counts[str(order_dir)] = used + len(items)
        return len(items)

    # -- 背景 ------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self._write_one(job)
            except Exception as e:
                logger.error(f"[ocr_rec] 寫圖失敗（略過，不影響辨識）: {e!r}")
            finally:
                self._queue.task_done()

    def _write_one(self, job: dict) -> None:
        order_dir: Path = job["dir"]
        os.makedirs(order_dir, exist_ok=True)
        seq = job["seq"]
        tag = f"{job['kind']}{seq if seq is not None else ''}"
        written = 0
        for i, (crop, text, conf) in enumerate(job["items"]):
            name = f"{job['stamp']}_{tag}_{i:02d}.jpg"
            img = render_rec_crop(crop, text, conf, f"{job['kind']} #{seq if seq is not None else '-'} box {i}", self._fonts)
            if cv2.imwrite(str(order_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                written += 1
            else:
                self._fail += 1
                if self._fail % 20 == 1:
                    logger.warning(f"[ocr_rec] imwrite 失敗（累積 {self._fail} 次）: {order_dir / name}")
        logger.info(f"[ocr_rec] {order_dir.parent.parent.name}/ocr_rec/{order_dir.name}/{job['stamp']}_{tag}_*.jpg  {written}/{len(job['items'])} 張")

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
