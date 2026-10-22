import cv2
from pathlib import Path as p
import numpy as np
from queue import Queue, Full, Empty
from threading import Thread, Lock
from collections import deque
import subprocess
import os
from time import sleep, time
from datetime import datetime, timezone
from loguru import logger
from typing import Any, Optional
import csv
import traceback
import shutil
import uuid
import re
from contextlib import nullcontext


def get_now_str(timestamp: float, utc: bool = False) -> str:
    """
    將 epoch timestamp 轉成 "YYYYMMDD HHMMSS.ffffff" 格式的字串。

    前 8 碼 (YYYYMMDD) 被呼叫端拿去當日期資料夾名稱使用，
    例如: "20260821 143022.123456"

    Args:
        timestamp: time.time() 回傳的 epoch seconds (float)
        utc: True 則輸出 UTC 時間；False (預設) 輸出本地時間
    """
    if utc:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(timestamp)  # 系統本地時區

    return dt.strftime("%Y%m%d_%H%M%S_%f")

def get_now_iso_str(timestamp: float, utc: bool = False) -> str:
    """
    跟 get_now_str 用同一套轉換邏輯，只是輸出成 ISO 格式
    "YYYY-MM-DDTHH:MM:SS"，給 CSV / tray.start_time_str 這類
    需要可讀時間戳的欄位使用，確保跟影片檔名的轉換方式完全一致。
    """
    if utc:
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    else:
        dt = datetime.fromtimestamp(timestamp)  # 跟 get_now_str 同一行邏輯
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def get_utc_offset() -> int:
    """
    回傳目前系統本地時區相對 UTC 的偏移量（小時），例如台灣回傳 8。

    寫進 CSV 的 'utc offset' 欄位，供之後用 local time + offset
    回推絕對時間 (UTC) 使用。
    """
    local_now = datetime.now().astimezone()
    offset = local_now.utcoffset()
    return int(offset.total_seconds() // 3600)


def discard_clip_frames(save_dir, csv_path=None, tag='', lock=None, reason=''):
    """
    丟掉一段「不會產出影片」的 JPEG 資料夾 (以及配對的 per-frame CSV)。

    沒有影片的資料夾留在當日資料夾裡只會佔空間、而且之後得人工進去清,
    所以只要確定不 encode 就直接刪掉。
    """
    with (lock or nullcontext()):
        try:
            if save_dir is not None:
                save_dir = p(save_dir)
                if save_dir.exists():
                    shutil.rmtree(save_dir)
                    logger.warning(f'[{tag}] 刪除未轉成影片的影像資料夾: {save_dir} ({reason})')

            if csv_path is not None:
                csv_path = p(csv_path)
                if csv_path.exists():
                    csv_path.unlink()
                    logger.warning(f'[{tag}] 刪除沒有對應影片的 csv: {csv_path} ({reason})')
        except Exception:
            logger.error(traceback.format_exc())


def encode_frames_to_video(save_dir, video_path, csv_path, fps, bitrate, tag='',
                           lock=None, cleanup_frames=True, drop_csv_on_failure=False,
                           keep_frames_on_failure=False) -> bool:
    """
    把 save_dir 底下的 JPEG 序列 encode 成 video_path,回傳是否真的產出影片。

    Clip 正常收尾與事後補救 (recover_orphan_clips) 共用這一段,所以獨立成
    module-level function,不依賴任何 Clip 的狀態。

    cleanup_frames:
        True (預設) -> encode 結束後把 JPEG 資料夾刪掉。成功的話影片已經有了,
                       沒必要繼續佔著 /mnt/reserved。
    keep_frames_on_failure:
        True  -> encode **失敗**時保留 JPEG 資料夾,交給 recover_orphan_clips 重試。
                 線上錄影 (Clip._run_ffmpeg) 一定要開:最常見的失敗是關程式時
                 ffmpeg 被連帶中斷,那是暫時性的、重試就會成功。把來源刪掉等於讓
                 CSV 永久指向一支永遠不會出現的影片 —— 2026-09-22 在機器上實測到
                 5 筆這種列 (每筆都留下一個沒有 mp4 的逐幀 csv 當指紋)。
        False (預設) -> 失敗也刪。recover_orphan_clips 自己重試時用這個:它已經是
                 第二次機會,而且是在沒有錄影競爭的空檔跑的,再失敗就是真的編不出來;
                 留著只會每輪重新搶硬體編碼器、又一直佔磁碟。
    drop_csv_on_failure:
        True -> 沒能產出影片時,把配對的 per-frame CSV 一起刪掉,不留孤兒 CSV。
    lock:
        跟 Clip.rmtree_lock 一樣的鎖,避免跟其他清理路徑同時 rmtree 同一個目錄。
    """
    success = False
    save_dir = p(save_dir) if save_dir is not None else None
    video_path = p(video_path)

    try:
        if save_dir is None or not save_dir.exists():
            logger.warning(f'[{tag}] 影像資料夾不存在,略過 encode: {save_dir}')
            return False

        frames = sorted(save_dir.glob('*.jpg'))
        if not frames or len(frames) < 2:
            logger.warning(f'[{tag}] no frame found in {save_dir} or number of frame < 2, skip ffmpeg encoding !')
            return False

        list_path = save_dir / 'frames.txt'
        with open(list_path, 'w', encoding='utf-8') as f:
            for frame in frames:
                f.write(f"file '{frame.name}'\n")
                f.write(f"duration {1 / fps}\n")
            f.write(f"file '{frames[-1].name}'\n")

        cmd = [
            'ffmpeg',
            '-nostdin',
            '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_path),
            '-r', str(round(fps, 2)),
            '-vsync', 'cfr',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'h264_v4l2m2m',
            '-b:v', f'{round(bitrate, 1)}M',
            '-maxrate', f'{round(bitrate * 2, 1)}M',
            '-bufsize', f'{round(bitrate * 4, 1)}M',
            '-g', str(int(fps * 2)),
            '-num_output_buffers', '32',
            '-num_capture_buffers', '32',
            str(video_path),
        ]
        logger.debug(f'ffmpeg command: {cmd}')

        logger.info(f'[{tag}] start encoding video: {video_path} (fps={fps:.2f}) ...')
        # start_new_session=True 把 ffmpeg 放到自己的 process group / session。
        #
        # 不這樣做的話,ffmpeg 跟 app 同在一個前景 process group:終端機的 Ctrl+C
        # (SIGINT 送給整個 group) 或 systemctl stop/restart 的 SIGTERM 會**同時**
        # 打到正在編碼的 ffmpeg,它就會在中途 "Exiting normally, received signal 2"
        # 然後以 returncode 255 收場。app 自己的 graceful_exit 救不了子程序。
        # 2026-09-22 機器上 5 筆「CSV 有檔名、影片不存在」全部都是這樣來的。
        #
        # 分家之後 ffmpeg 不再收到那些訊號,會自己把影片編完 (concat 的輸入是一份
        # 固定的檔案清單,不需要父程序在),之後沒人清的 JPEG 資料夾由下一次
        # recover_orphan_clips 的「影片已存在」分支收掉。
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,
        )
        t0 = time()
        stdout, stderr = proc.communicate()
        t1 = time()

        if proc.returncode == 0:
            success = True
            logger.success(f'[{tag}] encoded video successfully: {video_path}, frame: {len(frames)}, fps: {fps:.2f}, {t1 - t0:.3f} (s) !')
        else:
            logger.error(f'[{tag}] encode video failed! returncode={proc.returncode}\n{stderr}, {t1 - t0:.3f} (s)')
            if video_path.exists():
                video_path.unlink()
                logger.warning(f'[{tag}] deleted the {video_path} !')
    except Exception:
        logger.error(traceback.format_exc())
    finally:
        with (lock or nullcontext()):
            try:
                # 失敗且呼叫端要求保留時不刪:JPEG 是重編唯一的來源,刪了就再也補不回來。
                keep_frames = (not success) and keep_frames_on_failure
                if keep_frames and save_dir is not None and save_dir.exists():
                    logger.warning(
                        f'[{tag}] encode 失敗,保留影像資料夾等 recover_orphan_clips '
                        f'事後重編: {save_dir}'
                    )
                elif cleanup_frames and save_dir is not None and save_dir.exists():
                    shutil.rmtree(save_dir)
                    logger.warning(f'[{tag}] deleted the {save_dir}')
                if not success and drop_csv_on_failure and csv_path is not None:
                    csv_path = p(csv_path)
                    if csv_path.exists():
                        csv_path.unlink()
                        logger.warning(f'[{tag}] deleted the {csv_path}')
            except Exception:
                logger.error(traceback.format_exc())

    return success


class Clip:
    def __init__(self, root_dir, enable, tag, crf, bitrate, fps,
                 frame_size=None, preroll_frames=0, postroll_frames=0,
                 category=None):
        self.root_dir = root_dir
        self.tag = tag
        self.category = category or tag
        self.crf = crf
        self.bitrate = bitrate
        self.fps = fps
        self.frame_size = frame_size
        self.suffix = f'_{tag.lower()}' if tag else ''
        self.rmtree_lock = Lock()
        self.is_enable = enable
        self.date = get_now_str(time(), utc=False)[:8]
        self.session_id = None
        self.session_video_map: dict[str, tuple[str, float]] = {}

        # --- pre-roll / post-roll 設定 ---
        self.preroll_frames = preroll_frames
        self.postroll_frames = postroll_frames
        self._preroll_buffer = deque(maxlen=preroll_frames) if preroll_frames > 0 else None
        self._post_roll_remaining = 0
        self._pending_save_video = True

        if not self.is_enable:
            logger.warning(f'[{self.tag}] Clip function is disabled !')
        self._reset()

    # ---------- 對外主要介面 ----------
    def get_current_session_id(self) -> Optional[str]:
        return self.session_id

    def get_video_filename_for_session(self, session_id: str) -> str:
        entry = self.session_video_map.get(session_id)
        return entry[0] if entry else ""

    def feed(self, frame, timestamp) -> None:
        """
        每一幀都呼叫這個方法（不管有沒有在錄影）。

        - 尚未開始錄影：只更新 pre-roll 緩衝，供下次 start() 帶入。
        - 錄影中：正常寫入，並在 post-roll 倒數期間遞減，歸零時自動真正停止。
        """
        if not self.is_enable:
            return

        if self.frame_size is not None:
            frame = cv2.resize(frame, self.frame_size)

        if self._preroll_buffer is not None:
            self._preroll_buffer.append((frame, timestamp))

        if not self.is_running:
            return

        self.write_frame(frame, timestamp)

        if self._post_roll_remaining > 0:
            self._post_roll_remaining -= 1
            if self._post_roll_remaining <= 0:
                self._do_stop(self._pending_save_video)
                logger.info(f"[{self.tag}] post-roll 結束，錄影已停止")

    def write_frame(self, frame, timestamp):
        if not self.is_enable:
            return

        try:
            self.frame_q.put_nowait((frame, timestamp))
        except Full:
            self.n_discard_frame += 1  # 計算共丟棄了多少幀

    def start(self) -> Optional[str]:
        """
        開始錄影，自動把目前 pre-roll 緩衝的畫面一併帶入影片開頭。

        回傳這次要掛的 session id，沒有影片可掛就回傳 None —— 呼叫端必須用
        這個回傳值，不要自己再去問 get_current_session_id()：錄影被拒絕時
        self.session_id 還留著上一段的值，拿去查檔名會得到「這一單沒錄到影片，
        CSV 卻指向上一單影片」的錯誤紀錄。
        """

        if not self.is_enable:
            return None

        if self.is_running:
            if self._post_roll_remaining > 0:
                self._post_roll_remaining = 0
                logger.info(
                    f'[{self.tag}] 錄影中收到新的 start，'
                    f'取消原本的 post-roll'
                )
            else:
                logger.warning(
                    f'[{self.tag}] 已經在錄影中，忽略 start'
                )
            # 錄影確實在進行中，這一段畫面就在目前這支影片裡，掛同一個 session
            # 是正確的（連續兩單共用一支影片是既有設計）。
            return self.session_id

        # 上一段影片還在 encode
        if self.video_thread is not None and self.video_thread.is_alive():
            logger.warning(
                f'[{self.tag}] 上一段影片仍在 ffmpeg encode，'
                f'暫時不啟動新的錄影'
            )
            return None

        self.is_running = True
        self.session_id = uuid.uuid4().hex
        self._prune_session_map() # 定期清空資訊

        preroll = list(self._preroll_buffer) if self._preroll_buffer else []

        for item in preroll:
            try:
                self.frame_q.put_nowait(item)
            except Full:
                self.n_discard_frame += 1
                logger.warning(
                    f'[{self.tag}] pre-roll 畫面塞爆 queue，'
                    f'捨棄部分預錄畫面'
                )

        self.frame_thread = Thread(
            target=self._run,
            daemon=True
        )
        self.frame_thread.start()

        logger.info(
            f'[{self.tag}] 開始錄影，'
            f'帶入 {len(preroll)} 幀 pre-roll'
        )

        return self.session_id

    def request_stop(self, save_video: bool = True) -> None:
        """
        外部事件結束時呼叫。不會立刻真正停止，而是先讓 feed() 繼續
        寫入 postroll_frames 幀之後，才自動真正停止錄影，讓影片多帶到
        事件結束後幾秒的動作。
        """
        if not self.is_enable:
            return
        if not self.is_running:
            return
        if self._post_roll_remaining > 0:
            return  # 已經在倒數中，忽略重複請求

        if self.postroll_frames <= 0:
            self._do_stop(save_video)
            return

        self._pending_save_video = save_video
        self._post_roll_remaining = self.postroll_frames
        logger.info(f'[{self.tag}] 收到結束錄影請求，將再錄 {self.postroll_frames} 幀後停止')

    def stop(self, save_video: bool = True, is_interrupted: bool = False, wait_for_encode: bool = False, encode_timeout: float = 30.0,):
        """
        立即停止錄影。

        save_video:
            True  -> JPEG encode 成 MP4
            False -> 不進行 ffmpeg encode

        is_interrupted:
            True -> 表示異常/強制中斷情境

        wait_for_encode:
            True -> shutdown 時等待 ffmpeg 完成
            False -> ffmpeg 背景執行
        """

        self._post_roll_remaining = 0

        self._do_stop(
            save_video=save_video,
            is_interrupted=is_interrupted,
            wait_for_encode=wait_for_encode,
            encode_timeout=encode_timeout,
        )

    # ---------- 內部 ----------
    def _prune_session_map(self, max_age_seconds: float = 3600.0):
        now = time()
        expired = [
            sid for sid, (_, created_at) in self.session_video_map.items()
            if now - created_at > max_age_seconds
        ]
        for sid in expired:
            del self.session_video_map[sid]
        if expired:
            logger.info(f'[{self.tag}] 清掉 {len(expired)} 筆過期的 session_video_map 紀錄')

    def _do_stop(self, save_video, is_interrupted=False, wait_for_encode=False, encode_timeout=30.0,):
        if not self.is_enable:
            return

        self.is_running = False

        # --------------------------------------------------
        # 1. 等待 frame writer 把 queue 裡剩餘的 frame 寫完
        # --------------------------------------------------
        if self.frame_thread is not None:
            self.frame_thread.join(timeout=5.0)

            if self.frame_thread.is_alive():
                logger.warning(
                    f'[{self.tag}] frame writer thread '
                    f'仍未結束，5 秒 timeout'
                )
            else:
                self.frame_thread = None

        # --------------------------------------------------
        # 2. 統計 dropped frame
        # --------------------------------------------------
        if self.n_discard_frame > 0:
            logger.warning(
                f'[{self.tag}] discarded '
                f'{self.n_discard_frame} frames !'
            )

        # --------------------------------------------------
        # 3. 關閉 CSV
        # --------------------------------------------------
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        # --------------------------------------------------
        # 4. Encode
        # --------------------------------------------------
        if self.save_dir is not None:
            if save_video:
                self._encode_video(
                    wait=wait_for_encode,
                    timeout=encode_timeout,
                )
            else:
                # 確定不會 encode 成影片，就不要把 JPEG 資料夾留在當日資料夾裡
                # （以前這裡只印一行 log 就 reset，資料夾會一直卡著等人工清）
                logger.info(
                    f'[{self.tag}] skip ffmpeg encoding'
                )
                discard_clip_frames(
                    save_dir=self.save_dir,
                    csv_path=self.csv_path,
                    tag=self.tag,
                    lock=self.rmtree_lock,
                    reason='save_video=False',
                )

        # --------------------------------------------------
        # 5. 清理被中斷的資料
        # --------------------------------------------------
        if is_interrupted:
            with self.rmtree_lock:
                self.date = get_now_str(
                    time(),
                    utc=False
                )[:8]

                folder = (
                    p(self.root_dir)
                    / self.date
                    / self.category
                )

                if folder.exists():
                    for path in list(folder.glob('*')):
                        if path.is_dir():
                            shutil.rmtree(path)

        # --------------------------------------------------
        # 6. reset
        # --------------------------------------------------
        self._reset()

    def _run(self):
        try:
            while self.is_running:
                try:
                    # timestamp: time(), timezone: int, relative to UTC timezone
                    frame, timestamp = self.frame_q.get_nowait()
                except Empty:
                    sleep(0.05)
                    continue

                now_str = get_now_str(timestamp, utc=False)
                time_offset = get_utc_offset()

                # 第一個 frame
                if self.first_time is None:
                    logger.info(f"[{self.tag}] first frame's time information: {now_str} !")
                    self.date = now_str[:8]
                    self.save_dir = p(f'{self.root_dir}/{self.date}/{self.category}/{now_str}{self.suffix}')
                    self.csv_path = p(f'{self.root_dir}/{self.date}/{self.category}/{now_str}{self.suffix}.csv')
                    self.video_path = p(f'{self.root_dir}/{self.date}/{self.category}/{now_str}{self.suffix}.mp4')
                    self.session_video_map[self.session_id] = (self.video_path.name, time())
                    os.makedirs(self.save_dir, exist_ok=True)
                    self._write_csv_header()
                    self.first_time = (now_str, time_offset)
                    logger.success(f"[{self.tag}] the Clip thread started, "
                                   f"create clip's saved folder: {self.save_dir} and "
                                   f"csv path is {self.csv_path} !")

                # save frame and time
                dst_frame = f'{self.save_dir}/{now_str}.jpg'
                cv2.imwrite(dst_frame, frame)
                self._write_csv_row(now_str, time_offset)
                self.n_save_frame += 1

        except:
            logger.error(traceback.format_exc())
        finally:
            logger.success(f'[{self.tag}] clip frame queue thread stopped !')
            self.is_running = False

    def _write_csv_header(self):
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['local time', 'utc offset'])

    def _write_csv_row(self, now_str, time_offset):
        if self.csv is None:
            logger.error(f'[{self.tag}] csv writer is not inited, ignored to write row !')
            return
        self.csv.writerow([now_str, time_offset])

    def _encode_video(self, wait=False, timeout=30.0,):
        """背景執行 ffmpeg，需要時可等待完成。"""

        # 上一個 ffmpeg 還在跑
        if self.video_thread is not None:
            if self.video_thread.is_alive():
                # 這一段的影像資料夾會原地留下來，交給 recover_orphan_clips
                # 在空檔時撿回去補 encode，不會再變成要人工清的垃圾。
                logger.warning(
                    f'[{self.tag}] ffmpeg thread still running, '
                    f'skip new encoding (留待 recover_orphan_clips 事後補做): '
                    f'{self.save_dir}'
                )
                return

        data = {
            'save_dir': self.save_dir,
            'video_path': self.video_path,
            'csv_path': self.csv_path,
            'save_video': True,
            'tag': self.tag,
            'fps': self.fps,
            'bitrate': self.bitrate,
        }

        self.video_thread = Thread(
            target=self._run_ffmpeg,
            args=(data,),
            daemon=True,
        )

        self.video_thread.start()

        logger.info(
            f'[{self.tag}] ffmpeg encoding thread started'
        )

        if wait:
            logger.info(
                f'[{self.tag}] waiting for ffmpeg '
                f'(timeout={timeout}s)...'
            )

            self.video_thread.join(timeout)

            if self.video_thread.is_alive():
                logger.warning(
                    f'[{self.tag}] ffmpeg still running '
                    f'after {timeout}s'
                )
            else:
                logger.info(
                    f'[{self.tag}] ffmpeg encoding finished'
                )

    def _run_ffmpeg(self, data):
        """實際 encode 的工作交給 module-level 的 encode_frames_to_video，
        讓 recover_orphan_clips 事後補救時能走完全相同的一段邏輯。"""
        encode_frames_to_video(
            save_dir=data['save_dir'],
            video_path=data['video_path'],
            csv_path=data['csv_path'],
            fps=data['fps'],
            bitrate=data['bitrate'],
            tag=data['tag'],
            lock=self.rmtree_lock,
            # 線上這條路徑的失敗幾乎都是暫時性的(關程式時 ffmpeg 被連帶中斷、
            # 跟另一段錄影搶硬體編碼器),所以失敗要留著 JPEG 讓事後補編有來源。
            # CSV 早在收尾當下就寫好檔名了,來源一刪就變成永久的死連結。
            keep_frames_on_failure=True,
        )

    # ---------- 殘留錄影的事後補救 ----------
    def recover_orphan_clips(self, max_days: int = 10, min_idle_seconds: float = 120.0,
                             re_encode: bool = True) -> int:
        """
        掃掉 / 補救「有 JPEG 資料夾、卻沒有對應 mp4」的殘留錄影，回傳處理了幾筆。

        會留下這種資料夾的情況：
          - app 被 kill、斷電，錄影中途死掉，根本沒走到 encode
          - 上一段影片還在 ffmpeg encode，這一段的 encode 被跳過 (見 _encode_video)
          - encode 途中 process 被砍掉
        以前這些只能人工進 <root>/<日期>/<category>/ 裡面刪。

        判定條件刻意保守，避免誤刪正在錄的那一段：
          - 跳過目前這個 Clip 持有的 save_dir
          - 資料夾內最新檔案必須已經 min_idle_seconds 沒有再被寫入
          - 已經有 mp4 的話，那支 mp4 也必須已經 min_idle_seconds 沒被寫入 ——
            ffmpeg 跑在自己的 session、會活過 app 重啟，正在寫的 mp4 不能當成
            「已完成」而把它的 JPEG 來源刪掉

        「已經有 mp4、只是資料夾沒清掉」的殘骸隨時都可以刪，不動到編碼器；
        但「要補 encode」這種每一筆都會重新確認自己沒在錄影／沒在 encode，
        因為 h264_v4l2m2m 是硬體編碼器，跟現役的錄影搶會讓正式影片失敗。
        補不到的這一輪就留著，下一輪掃描會再撿一次。

        frame 數 >= 2 就拿去 encode 成影片 (re_encode=True)；不足 2 張、或 encode
        失敗，就連同配對的 per-frame CSV 一起刪掉。
        """
        if not self.is_enable:
            return 0

        root = p(self.root_dir)
        if not root.is_dir():
            return 0

        active = None
        if self.save_dir is not None:
            try:
                active = p(self.save_dir).resolve()
            except OSError:
                active = None

        now = time()
        handled = 0

        date_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and re.fullmatch(r'\d{8}', d.name)
        )[-max_days:]   # 更早的日期資料夾早就被上傳排程搬走了

        for date_dir in date_dirs:
            cat_dir = date_dir / self.category
            if not cat_dir.is_dir():
                continue

            for folder in sorted(f for f in cat_dir.iterdir() if f.is_dir()):
                try:
                    if active is not None and folder.resolve() == active:
                        continue

                    video_path = cat_dir / f'{folder.name}.mp4'
                    csv_path = cat_dir / f'{folder.name}.csv'
                    frames = sorted(folder.glob('*.jpg'))

                    # 這個資料夾還在被寫入嗎？看裡面最新的 mtime
                    mtimes = [folder.stat().st_mtime]
                    mtimes += [f.stat().st_mtime for f in frames[-5:]]
                    # mp4 也要算進去:ffmpeg 現在跑在自己的 session 裡(見
                    # encode_frames_to_video 的 start_new_session),所以它會活過
                    # app 的 Ctrl+C / restart。「Ctrl+C 之後馬上重開」時,開機這一輪
                    # 掃描會看到一個**還在寫**的 mp4,底下「影片已存在」那一支就會
                    # 把 ffmpeg 正在讀的 JPEG 資料夾刪掉(concat 是邊編邊讀,不是
                    # 一次載入),正在編的影片就毀了。編碼中的 mp4 mtime 是新的,
                    # 算進來這一輪就會自動讓位,下一輪再處理。
                    if video_path.exists():
                        mtimes.append(video_path.stat().st_mtime)
                    if now - max(mtimes) < min_idle_seconds:
                        continue

                    if video_path.exists():
                        # 影片已經好了，資料夾只是 encode 後沒清乾淨的殘骸
                        discard_clip_frames(
                            save_dir=folder, csv_path=None, tag=self.tag,
                            lock=self.rmtree_lock, reason='影片已存在',
                        )
                        handled += 1
                        continue

                    if re_encode and len(frames) >= 2:
                        # 每一筆都重新確認：現役錄影 / encode 一啟動就讓位
                        if self.is_running or (
                            self.video_thread is not None and self.video_thread.is_alive()
                        ):
                            logger.info(
                                f'[{self.tag}] 正在錄影 / encode 中，'
                                f'這一輪先不補 encode: {folder}'
                            )
                            continue

                        logger.warning(
                            f'[{self.tag}] 發現沒轉成影片的殘留資料夾，'
                            f'補做 encode: {folder} ({len(frames)} 張)'
                        )
                        encode_frames_to_video(
                            save_dir=folder,
                            video_path=video_path,
                            csv_path=csv_path,
                            fps=self.fps,
                            bitrate=self.bitrate,
                            tag=self.tag,
                            lock=self.rmtree_lock,
                            drop_csv_on_failure=True,
                        )
                    else:
                        discard_clip_frames(
                            save_dir=folder, csv_path=csv_path, tag=self.tag,
                            lock=self.rmtree_lock,
                            reason=f'影像數不足 ({len(frames)} 張)',
                        )

                    handled += 1
                except Exception:
                    logger.error(traceback.format_exc())

        if handled:
            logger.success(f'[{self.tag}] 殘留錄影掃描完成，共處理 {handled} 筆')

        return handled

    def _calc_fps_from_filenames(self, tag, frames, default_fps=30.0):
        """根據資料夾內第一張和最後一張圖片的檔名(時間戳)推算平均 fps。"""
        if len(frames) < 2:
            logger.warning(f'[{tag}] only {len(frames)} frame(s), cannot calc fps, '
                           f'fallback to default fps {default_fps} !')
            return default_fps

        fmt = "%Y%m%d %H%M%S.%f"  # 對應 get_now_str(utc=False) 的輸出格式
        try:
            first_dt = datetime.strptime(frames[0].stem, fmt)
            last_dt = datetime.strptime(frames[-1].stem, fmt)
        except ValueError as e:
            logger.error(f'[{tag}] failed to parse frame filename as datetime: {e}, '
                         f'fallback to default fps {default_fps} !')
            return default_fps

        duration = (last_dt - first_dt).total_seconds()
        if duration <= 0:
            logger.warning(f'[{tag}] invalid duration ({duration}s) calculated from filenames, '
                           f'fallback to default fps {default_fps} !')
            return default_fps

        fps = min((len(frames) - 1) / duration, default_fps)
        logger.info(f'[{tag}] calculated fps={fps:.2f} from {len(frames)} frames, '
                    f'duration={duration:.3f}s !')
        return fps

    def _reset(self):
        self.save_dir = None
        self.csv_path = None
        self.csv_file = None
        self.csv = None
        self.video_path = None
        self.frame_q = Queue(maxsize=256)
        self.n_discard_frame = 0
        self.n_save_frame = 0
        self.first_time = None
        self.frame_thread = None
        self.video_thread = None
        self.is_running = False
        self._post_roll_remaining = 0
        # 注意：_preroll_buffer 不清空！事件結束後緩衝區應該繼續累積
        # 「現在」的畫面，供下一次事件觸發使用，不能因為上一段錄影
        logger.success(f'[{self.tag}] reset !')


_UNSET = object()  # 用來跟「尚未初始化」跟「目前不在任何時段(None)」做區分


class PeriodicClipRecorder:
    """
    包一層在 Clip 外面，依設定的時段 (working hours) 自動控制 Clip 的
    start()/stop()，達成「固定時段持續錄影，每個區段切一支檔案」的效果
    —— 對應舊版 Video 的 working_hours 行為，但底層錄製引擎沿用 Clip
    (JPEG 序列 + ffmpeg 收尾)，不再走 Video 那條長時間運行容易寫不出
    檔案的 cv2.VideoWriter pipeline。

    跟事件觸發的 Clip 完全獨立：不共用 Clip instance，只共用呼叫端
    傳進來的「同一幀畫面」，本身不做任何 resize（沿用內部 Clip 的
    frame_size 設定，或呼叫端已經 resize 過就傳 None）。

    對外只暴露一個 feed()，什麼時候該開始/結束一段錄影完全由這個
    class 自己依時鐘判斷，呼叫端不需要也不應該手動呼叫底下 clip 的
    start()/stop()。
    """

    def __init__(self, clip: "Clip", periods: list[tuple[int, int]]):
        """
        Args:
            clip: 底下用來實際錄影的 Clip instance（建議 preroll_frames=0,
                  postroll_frames=0，因為連續錄影沒有「事件邊界」的概念）。
            periods: 要錄影的時段，例如 [(11, 12), (12, 13), ...] 代表
                     11點~12點、12點~13點分別各錄成一支檔案。
        """
        self.clip = clip
        self.periods = [set(range(s, e)) for s, e in periods]
        self._current_period_idx = _UNSET

    def _find_period_idx(self, dt) -> Optional[int]:
        hour = dt.hour
        for i, period in enumerate(self.periods):
            if hour in period:
                return i
        return None

    def feed(self, frame, timestamp: float) -> None:
        """每一幀都呼叫。內部自動判斷是否跨時段、要不要開始/結束錄影。"""
        now = datetime.fromtimestamp(timestamp)
        idx = self._find_period_idx(now)

        if idx != self._current_period_idx:
            if self.clip.is_running:
                logger.info(f'[{self.clip.tag}] 時段切換，結束目前這段連續錄影')
                self.clip.stop(save_video=True)  # 立即收尾，不需要 post-roll
            self._current_period_idx = idx
            if idx is not None:
                logger.info(f'[{self.clip.tag}] 進入錄影時段 (period_idx={idx})，開始新的一段')
                self.clip.start()

        if idx is not None:
            self.clip.feed(frame, timestamp)

    def stop(self, save_video: bool = True, wait_for_encode: bool = False, encode_timeout: float = 30.0,) -> None:
        if self.clip.is_running:
            self.clip.stop(
                save_video=save_video,
                wait_for_encode=wait_for_encode,
                encode_timeout=encode_timeout,
            )

        self._current_period_idx = _UNSET