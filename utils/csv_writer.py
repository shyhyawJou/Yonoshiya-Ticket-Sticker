import csv
import io
import threading
import queue
import socket
import os
import json
import re
from datetime import datetime
from loguru import logger


_SIZE_MULTIPLIER = {
    "": 1,
    "K": 1024,
    "M": 1024 ** 2,
    "G": 1024 ** 3,
    "T": 1024 ** 4,
}


def parse_size(value) -> int | None:
    """
    把設定檔的大小欄位轉成位元組。

    接受純數字 (視為位元組) 或 "1GB" / "900MB" / "1.5G" / "500KiB" 這類字串。
    None / 空值 / 0 / 無法解析都回傳 None,代表「不限制」。
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        size = int(value)
        return size if size > 0 else None

    text = str(value).strip().upper().replace(" ", "")
    if not text or text in ("NULL", "NONE", "FALSE", "0"):
        return None

    m = re.fullmatch(r"([0-9]*\.?[0-9]+)([KMGT]?I?B?)", text)
    if not m:
        logger.warning(f"[CsvWriter] 無法解析大小設定 {value!r},視為不限制")
        return None

    unit = m.group(2).rstrip("B").rstrip("I")   # "KIB"/"KB"/"K" -> "K"
    size = int(float(m.group(1)) * _SIZE_MULTIPLIER[unit])
    return size if size > 0 else None


class CsvWriter:
    """
    一筆完成的訂單寫一列 CSV。

    檔案跟影片放在同一個日期資料夾底下:
        <root_dir>/<YYYYMMDD>/<folder>/<YYYYMMDD>_<機器ID後4碼>.csv

    max_video_size 有設定時,會累計「這一份 CSV 裡所有 video_file 的檔案大小」,
    超過上限就換下一份,檔名多帶份數與該份第一列的時間:
        20260911_7d2f_1_120315.csv / 20260911_7d2f_2_140052.csv
    不設上限 (None) 時檔名維持原本的 <日期>_<ID>.csv,行為跟舊版一致。

    影片大小是「事後才算得出來」的: 訂單收尾寫 CSV 的那一刻 ffmpeg 往往還在
    背景 encode,檔案還不存在。所以這裡不在寫入當下記住大小,而是每次要寫新列
    之前,重新 sum 一次這一份 CSV 已經寫過的那些影片檔 —— 還沒 encode 完的這次
    算 0,下一列就會被算進去,會自我補齊。代價是切點可能超出上限一支影片的量,
    這是可以接受的近似。
    """

    FIELDNAMES = [
        "tray_id", "order_number", "start_time", "end_time", "final_status",
        "expected_item", "expected_item_count", "missing_item", "wrong_item",
        "video_file"
    ]

    def __init__(self, root_dir: str, folder: str = "csv_logs",
                 max_video_size=None, video_category: str = "event"):

        self.root_dir = root_dir
        self.folder = folder
        self.video_category = video_category
        self.max_video_size = parse_size(max_video_size)

        full_hostname = socket.gethostname()
        self.host_suffix = full_hostname[-4:]

        # --- 目前這一份 CSV 的狀態 ---
        self._part_date: str | None = None     # 這一份屬於哪一天 (YYYYMMDD)
        self._part_index: int = 1              # 第幾份
        self._part_stamp: str = ""             # 這一份第一列的時間 (HHMMSS)
        self._part_videos: list[str] = []      # 這一份已寫過哪些影片檔名

        # 這個 process 已檢查過斷電殘留的檔案。斷電殘骸只會在本 process 啟動前
        # 產生，所以每份檔案只需檢查一次；寫入失敗時會移出這個集合重新檢查。
        self._checked_files: set[str] = set()

        if self.max_video_size:
            logger.info(
                f"[CsvWriter] 單份 CSV 對應影片大小上限: "
                f"{self.max_video_size / 1024 ** 2:.1f} MB"
            )
        else:
            logger.info("[CsvWriter] 單份 CSV 不限制對應影片大小")

        self.queue = queue.Queue()
        self.running = True

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ---------- 檔名 / 切檔 ----------
    def _video_size_total(self) -> int:
        """重新 sum 這一份 CSV 已寫過、且已經 encode 出來的影片大小"""
        total = 0
        video_dir = os.path.join(self.root_dir, self._part_date, self.video_category)
        for name in self._part_videos:
            try:
                total += os.path.getsize(os.path.join(video_dir, name))
            except OSError:
                continue    # 還在 encode、encode 失敗或已被搬走,這次算 0
        return total

    def _start_new_part(self, date_str: str, index: int) -> None:
        self._part_date = date_str
        self._part_index = index
        self._part_stamp = datetime.now().strftime("%H%M%S")
        self._part_videos = []

    def _get_current_filename(self) -> str:
        """
        決定這一列要寫進哪個檔案,必要時換日 / 換份,並確保資料夾存在。
        例如: /mnt/reserved/record/stream/20260911/csv_logs/20260911_7d2f.csv
        """
        date_str = datetime.now().strftime("%Y%m%d")

        if date_str != self._part_date:
            # 跨日 (或第一次寫入): 重新從第 1 份開始
            self._start_new_part(date_str, 1)
        elif self.max_video_size and self._part_videos:
            total = self._video_size_total()
            if total >= self.max_video_size:
                logger.info(
                    f"[CsvWriter] 第 {self._part_index} 份對應影片累計 "
                    f"{total / 1024 ** 2:.1f} MB 已達上限,切換到下一份"
                )
                self._start_new_part(date_str, self._part_index + 1)

        if self.max_video_size:
            filename = f"{date_str}_{self.host_suffix}_{self._part_index}_{self._part_stamp}.csv"
        else:
            filename = f"{date_str}_{self.host_suffix}.csv"

        log_dir = os.path.join(self.root_dir, date_str, self.folder)
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, filename)

    # ---------- 斷電防護 ----------
    @staticmethod
    def _fsync_dir(path: str) -> None:
        """新建的檔案要連資料夾一起 fsync，斷電後目錄裡才確定看得到它。"""
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _salvage(data: bytes, fieldnames: list[str]) -> tuple[bytes, list[bytes], dict]:
        """從可能損壞的 CSV 內容中，只留下完整、可解析的 record。

        純函式（不碰檔案系統），回傳 (修復後內容, 被丟掉的損壞片段, 統計)。

        判斷依據：這個 writer 寫出的每一筆 record 都是「單獨一行、以 \\r\\n
        結尾」—— dict/list 欄位經 json.dumps 後換行會被跳脫成 \\n，其餘欄位是
        單號、時間、狀態與檔名，都不含換行。所以可以逐行判斷：
          1. 最後一段沒有 \\n 結尾 = 寫到一半就斷了，整段丟掉。不能只補換行：
             若截在 JSON 的引號裡，引號不會被關上，後面追加的每一筆都會被
             CSV 解析器吞進同一個欄位，直到換檔為止。
          2. 某行含 NUL：跳電時 ext4 記下了檔案長度、資料卻沒落盤，那段變成
             NUL；重開機後追加的新 record 從 NUL 之後開始。所以最後一個 NUL
             之前是被截斷的舊 record（丟掉），之後才是完整的新 record。
             不能只刪 NUL：一列跨頁時可能前半落盤、後半是 NUL，刪掉 NUL 會把
             半列和下一列黏成一行。
          3. 每一行都要能以 UTF-8 解碼、解析成剛好一筆、欄位數與標題相同。
             截在日文多位元組字元中間的半列若留著，整份檔案都無法以 UTF-8 讀取。
          4. 第一行必須是完整標題；只剩 BOM、標題被截斷或含 NUL 時重建標題。
             標題完整但欄位與目前 FIELDNAMES 不同（舊版檔案）時保留原標題，
             並以它的欄位數檢查 record，避免把整份舊資料當成損壞。
        """
        bom = b"\xef\xbb\xbf"
        expected_header = (",".join(fieldnames) + "\r\n").encode("utf-8")
        stats = {"nul": data.count(b"\x00"), "dropped": 0,
                 "header_rebuilt": False, "kept": 0}

        has_bom = data.startswith(bom)
        if has_bom:
            body = data[len(bom):]
        elif len(data) < len(bom) and bom.startswith(data):
            body = b""          # BOM 本身就被截斷了
        else:
            body = data

        def parse(line: bytes):
            try:
                text = line.decode("utf-8")
            except UnicodeDecodeError:
                return None
            rows = list(csv.reader(io.StringIO(text, newline="")))
            return rows[0] if len(rows) == 1 else None

        dropped: list[bytes] = []

        def drop(fragment: bytes) -> None:
            fragment = fragment.replace(b"\x00", b"")
            if fragment.strip(b"\r\n"):
                dropped.append(fragment)

        # 1. 拆成完整的行；沒有 \n 結尾的最後一段一律丟掉
        *complete, tail = body.split(b"\n")
        drop(tail)

        # 2. 含 NUL 的行，只留最後一個 NUL 之後的部分
        lines: list[bytes] = []
        for raw in complete:
            line = raw + b"\n"
            if b"\x00" in line:
                cut = line.rfind(b"\x00") + 1
                drop(line[:cut])
                line = line[cut:]
                if not line.strip(b"\r\n"):
                    continue
            lines.append(line)

        # 4. 標題：完整且以第一個欄位名開頭才算數
        header = expected_header
        n_fields = len(fieldnames)
        first = lines[0] if lines else b""
        first_row = parse(first) if first else None
        if first_row and first_row[0] == fieldnames[0]:
            header = first
            n_fields = len(first_row)
            lines = lines[1:]
        else:
            stats["header_rebuilt"] = True

        # 3. 其餘每一行都必須是一筆完整 record
        kept: list[bytes] = []
        for line in lines:
            row = parse(line)
            if row is not None and len(row) == n_fields:
                kept.append(line)
            elif line.strip(b"\r\n"):
                drop(line)

        stats["dropped"] = len(dropped)
        stats["kept"] = len(kept)
        prefix = bom if (has_bom or stats["header_rebuilt"]) else b""
        return prefix + header + b"".join(kept), dropped, stats

    def _repair_after_power_loss(self, path: str) -> None:
        """清掉上次斷電（或寫入失敗）留下的殘骸。

        2026-10-02 吉野家現場：寫完一列後還沒落盤就跳電，ext4 已經記下檔案
        長度、資料區塊卻沒寫進去，重開機後那一列變成同樣長度的 NUL（\\x00）。
        app 接著在後面追加，NUL 就夾在兩列中間 —— VS Code 顯示成空白，
        同仁用程式讀 CSV 卻讀到一串 NULL。損壞的那一列已經找不回來，這裡只
        保留完整的 record（判斷規則見 _salvage），被丟掉的片段另存到
        <檔名>.damaged.txt 並記 WARNING，讓遺失有跡可循。

        改寫走「暫存檔 → fsync → rename」，修到一半斷電也不會比原本更糟。
        檢查成功（或確認不需修復）才記為已檢查；讀不到或修復失敗時，下一筆
        寫入前會再試一次。寫入失敗時 _worker 也會清掉標記（例如磁碟滿留下半列）。
        """
        if path in self._checked_files:
            return

        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            self._checked_files.add(path)   # 新檔，由這個 process 從頭寫
            return
        except OSError as e:
            logger.warning(f"[CsvWriter] 無法檢查 {path} 是否有斷電殘留，下一筆再試: {e}")
            return

        if not data:
            self._checked_files.add(path)   # 0 byte：_worker 會補寫標題
            return

        fixed, dropped, stats = self._salvage(data, self.FIELDNAMES)
        if fixed == data:
            self._checked_files.add(path)
            return

        damaged_path = path + ".damaged.txt"
        tmp = path + ".repair.tmp"
        try:
            if dropped:
                # 先把要丟掉的片段落盤，再改寫本體：修到一半斷電也不會兩邊都沒有
                with open(damaged_path, "ab") as f:
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"# {stamp} 從 {os.path.basename(path)} 移除的損壞片段\n".encode("utf-8"))
                    for fragment in dropped:
                        f.write(fragment if fragment.endswith(b"\n") else fragment + b"\n")
                    f.flush()
                    os.fsync(f.fileno())
            with open(tmp, "wb") as f:
                f.write(fixed)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            self._fsync_dir(os.path.dirname(path))
        except OSError as e:
            logger.error(f"[CsvWriter] 修復 {path} 失敗，下一筆再試: {e}")
            return

        self._checked_files.add(path)
        done = []
        if stats["nul"]:
            done.append(f"刪除 {stats['nul']} 個 NUL 位元組")
        if stats["dropped"]:
            done.append(f"移除 {stats['dropped']} 段不完整的資料（另存於 {os.path.basename(damaged_path)}）")
        if stats["header_rebuilt"]:
            done.append("重建標題列")
        logger.warning(
            f"[CsvWriter] {os.path.basename(path)} 有斷電或寫入失敗留下的殘骸，已修復："
            f"{'、'.join(done) or '整理換行'}；保留 {stats['kept']} 筆完整訂單。"
            f"損壞那幾筆的內容已無法還原"
        )

    def _worker(self):
        # "ticket_capture", "final_tray_capture",
        fieldnames = self.FIELDNAMES

        while self.running:
            try:
                data = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_filename = None
            try:
                current_filename = self._get_current_filename()
                self._repair_after_power_loss(current_filename)
                # 空檔也要寫標題：剛建檔就斷電的話，修復後會剩一個 0 byte 的檔案
                write_header = (not os.path.exists(current_filename)
                                or os.path.getsize(current_filename) == 0)

                with open(current_filename, mode='a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()

                    row_to_write = {}
                    for k in fieldnames:
                        val = data.get(k)
                        if val is None:
                            row_to_write[k] = ""
                        elif isinstance(val, (dict, list)):
                            row_to_write[k] = json.dumps(val, ensure_ascii=False)
                        elif isinstance(val, float):
                            row_to_write[k] = round(val, 2)
                        else:
                            row_to_write[k] = val

                    writer.writerow(row_to_write)
                    # 落盤才算寫完：只 close 的話資料可能還在 page cache，
                    # 這時跳電就會變成上面那種 NUL 列。一筆訂單才寫一次，成本可忽略。
                    f.flush()
                    os.fsync(f.fileno())
                if write_header:
                    self._fsync_dir(os.path.dirname(current_filename))
                logger.success(f'寫入一筆資料到 {current_filename}')

                # 記住這一列對應的影片,下一列才算得出這一份累積了多大
                video_file = row_to_write.get("video_file")
                if video_file and video_file not in self._part_videos:
                    self._part_videos.append(video_file)

                self.queue.task_done()
            except Exception as e:
                logger.error(f"CSV Write Error: {e}")
                # 寫到一半失敗（例如磁碟滿）可能留下半列；下一筆寫入前要重新檢查
                if current_filename:
                    self._checked_files.discard(current_filename)

    # ---------- 空資料防線 ----------
    _EMPTY_VALUES = (None, "", "-", "[]", "{}")

    @classmethod
    def _is_empty_row(cls, data: dict) -> bool:
        """
        判斷這一列是不是「沒有任何訂單內容」的空殼。

        會出現空殼是因為收尾條件被誤觸發（例如 ticket_mat 模式下手擦過、
        傳票只是經過而從沒 OCR 成功），結果寫出單號空白、品項空白、只有
        時間軸在跑的列。呼叫端（LogicEngine）已經會擋掉，這裡是第二道
        防線：不管是誰呼叫 log()，空殼都不落地。

        判定條件刻意保守 —— 單號、應有品項、缺少品項、錯誤品項全空，
        且 expected_item_count 不大於 0，才算空殼。只要有任何一項有值
        （例如單號辨識失敗但品項讀到了）就照常寫入。
        """
        if data.get("order_number"):
            return False

        for key in ("expected_item", "missing_item", "wrong_item"):
            val = data.get(key)
            if isinstance(val, (list, dict, tuple)):
                if len(val) > 0:
                    return False
            elif val not in cls._EMPTY_VALUES:
                return False

        try:
            if int(data.get("expected_item_count") or 0) > 0:
                return False
        except (TypeError, ValueError):
            return False    # 值怪異時寧可寫進去，留著給人看

        return True

    def log(self, data: dict):
        if self._is_empty_row(data):
            logger.warning(
                f"[CsvWriter] 丟棄一列空資料 (單號與所有品項皆空): "
                f"tray_id={data.get('tray_id')} "
                f"start={data.get('start_time')} end={data.get('end_time')} "
                f"video={data.get('video_file')}"
            )
            return
        self.queue.put(data)

    def stop(self):
        self.running = False
        self.thread.join()
        logger.info("Csv Writer stopped.")
