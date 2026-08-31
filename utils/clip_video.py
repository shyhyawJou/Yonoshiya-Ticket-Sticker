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


def get_utc_offset() -> int:
    """
    回傳目前系統本地時區相對 UTC 的偏移量（小時），例如台灣回傳 8。

    寫進 CSV 的 'utc offset' 欄位，供之後用 local time + offset
    回推絕對時間 (UTC) 使用。
    """
    local_now = datetime.now().astimezone()
    offset = local_now.utcoffset()
    return int(offset.total_seconds() // 3600)


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

    def start(self):
        """開始錄影，自動把目前 pre-roll 緩衝的畫面一併帶入影片開頭。"""

        if not self.is_enable:
            return

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
            return

        # 上一段影片還在 encode
        if self.video_thread is not None and self.video_thread.is_alive():
            logger.warning(
                f'[{self.tag}] 上一段影片仍在 ffmpeg encode，'
                f'暫時不啟動新的錄影'
            )
            return

        self.is_running = True

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
                logger.info(
                    f'[{self.tag}] skip ffmpeg encoding'
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
                logger.warning(
                    f'[{self.tag}] ffmpeg thread still running, '
                    f'skip new encoding'
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
        try:
            # 變數
            save_dir = data['save_dir']
            video_path = data['video_path']
            csv_path = data['csv_path']
            save_video = data['save_video']
            tag = data['tag']
            fps = data['fps']
            bitrate = data['bitrate']

            # 不運作
            if not save_video:
                return

            frames = sorted(save_dir.glob('*.jpg'))
            if not frames or len(frames) < 2:
                logger.warning(f'[{tag}] no frame found in {save_dir} or number of frame < 2, skip ffmpeg encoding !')
                return

            #fps = self._calc_fps_from_filenames(tag, frames)

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
            logger.info(f'ffmpeg command: {cmd}')

            logger.info(f'[{tag}] start encoding video: {video_path} (fps={fps:.2f}) ...')
            proc = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            t0 = time()
            stdout, stderr = proc.communicate()
            t1 = time()

            if proc.returncode == 0:
                logger.success(f'[{tag}] encoded video successfully: {video_path}, frame: {len(frames)}, fps: {fps:.2f}, {t1 - t0:.3f} (s) !')
            else:
                logger.error(f'[{tag}] encode video failed! returncode={proc.returncode}\n{stderr}, {t1 - t0:.3f} (s)')
                if video_path.exists():
                    video_path.unlink()
                    logger.warning(f'[{tag}] deleted the {video_path} !')
        except:
            logger.error(traceback.format_exc())
        finally:
            with self.rmtree_lock:
                if save_dir and save_dir.exists():
                    shutil.rmtree(save_dir)
                    logger.warning(f'[{tag}] deleted the {save_dir}')
                if not save_video and csv_path.exists():
                    csv_path.unlink()
                    logger.warning(f'[{tag}] deleted the {csv_path}')

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