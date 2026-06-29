import cv2
from pathlib import Path as p
from datetime import datetime
import os
from os.path import dirname, exists
from time import sleep
import queue
from loguru import logger
from threading import Lock, Thread
import traceback



class Video:
    def __init__(self, base_dir, frame_w, frame_h, fps, suffix=''):
        self.base_dir = base_dir
        self.date = None
        self.hour = None
        self.period_idx = None
        self.video_writer = None
        self.time_writer = None
        self.video_path = None
        self.time_path = None
        self.lock = Lock()
        self.write_thread = None
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.fps = fps
        self.suffix = suffix
        self.num_write = 0
        self.frame_q = queue.Queue(512)
        self.running = True
        periods = [
            (11, 12), #1
            (12, 13), #2
            (13, 14), #3
            (14, 15), #4
            (15, 16), #5
            (16, 17), #6
            (17, 18), #7
            (18, 19), #8
            (19, 20)  #9
        ]
        self.working_hours = self._make_working_hour(periods)
        self._init_writer()

    def write_frame(self, frame):
        if self._change_period():
            logger.info('錄影時段發生變動')
            self._clear_queue()
            self.stop()
            self._init_writer()

        if frame is None or not self.is_working_period():
            return

        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.frame_q.put_nowait((ts, frame))
        except queue.Full:
            pass

    @property
    def today_dir(self):
        return os.path.join(self.base_dir, self.date) if self.date else self.base_dir

    def is_working_period(self):
        return self.period_idx is not None

    def stop(self):
        self._stop_write()

        with self.lock:
            if self.video_writer:
                try:
                    self.video_writer.release()
                    sleep(0.2)
                    size = os.stat(self.video_path).st_size / 1e6
                    logger.info(f'保存 {self.video_path} 成功, ' \
                                f'共寫入 {self.num_write} 幀, ' \
                                f'大小: {size} MB')
                except:
                    logger.error(f'保存 {self.video_path} 失敗, 錯誤: {traceback.format_exc()}')
                finally:
                    self.video_writer = None
                    self.video_path = None
                    self.num_write = 0

            if self.time_writer:
                try:
                    self.time_writer.flush()
                    self.time_writer.close()
                    logger.info(f'保存時戳檔 {self.time_path} 成功')
                except:
                    logger.error(f'保存時戳檔 {self.time_path} 失敗, 錯誤: {traceback.format_exc()}')
                finally:
                    self.time_writer = None
                    self.time_path = None

        if self.finish_marker_path:
            try:
                with open(self.finish_marker_path, 'w'):
                    pass
                logger.info(f'保存結束標記檔 {self.finish_marker_path} 成功')
            except:
                logger.error(f'保存結束標記檔 {self.finish_marker_path} 失敗, 錯誤: {traceback.format_exc()}')
            finally:
                self.finish_marker_path = None

        logger.info('已成功釋放影片相關資源')

    def _init_writer(self):
        self._decide_path()

        if self.video_path:
            self._init_video_writer()

        if self.time_path:
            self._init_time_writer()

        if self.is_working_period():
            self._start_write()

    def _write(self):
        logger.info('開始執行寫影片的執行緒')

        while self.running:
            try:
                ts, frame = self.frame_q.get(timeout=1.)
                # frame = cv2.resize(frame, (self.frame_w, self.frame_h))
            except queue.Empty:
                sleep(0.5)
                continue
            
            if not self.video_writer or not self.time_writer:
                sleep(1.)
                continue

            with self.lock:
                try:
                    self.video_writer.write(frame)
                except:
                    logger.error(f'寫入 frame 發生錯誤: {traceback.format_exc()}')
                    continue
            
                try:
                    self.time_writer.write(f'{ts}\n')
                except:
                    logger.error(f'寫入時戳發生錯誤: {traceback.format_exc()}')
                    continue

            self.num_write += 1
            if self.num_write % 10000 == 0:
                logger.info(f'已經寫入 {self.num_write} 幀 frame')

        logger.info('已停止寫影片的執行緒')

    def _decide_path(self):
        self._update_state()

        base_dir = f'{self.base_dir}/{self.date}'
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f'影片資料夾: {base_dir}, 會錄影的時段: {self.working_hours}')
        
        self.period_idx = self._find_period()

        if self.period_idx is None:
            self.video_path = None
            self.time_path = None
            self.finish_marker_path = None
            logger.warning('現在不是錄影時段, 不進行錄影')
        else:
            period_idx = self.period_idx + 1
            self.video_path = f'{self.base_dir}/{self.date}/{self.date}{self.suffix}_{period_idx}.mp4'
            self.time_path = f'{self.base_dir}/{self.date}/{self.date}{self.suffix}_{period_idx}.csv'
            self.finish_marker_path = f'{self.base_dir}/{self.date}/{self.date}{self.suffix}_{period_idx}.txt'

            if exists(self.video_path):
                try:
                    self.video_path = self._new_path(self.video_path)
                except:
                    logger.error(traceback.format_exc())

            if exists(self.time_path):
                try:
                    self.time_path = self._new_path(self.time_path)
                except:
                    logger.error(traceback.format_exc())

            if exists(self.finish_marker_path):
                try:
                    self.finish_marker_path = self._new_path(self.finish_marker_path)  
                except:
                    logger.error(traceback.format_exc())

            period = self.working_hours[self.period_idx]
            logger.info(f'現在是錄影時段: {min(period)}點 ~ {max(period)}點')

    def _make_working_hour(self, periods):
        working_hours = []
        for start, end in periods:
            working_hours.append(set(range(start, end)))
        return working_hours
    
    def _find_period(self):
        hour = datetime.now().hour
        for i, period in enumerate(self.working_hours):
            if hour in period:
                return i
        return None

    def _update_state(self):
        now = datetime.now()
        self.date = now.strftime('%Y%m%d')
        self.hour = now.hour

    def _init_video_writer(self):
        pipeline = (
            f'appsrc ! videoconvert ! '
            f'v4l2h264enc extra-controls="encode,video_bitrate=3000000,video_gop_size={self.fps}" ! '
            f'h264parse ! mp4mux ! filesink location={self.video_path} sync=false'
        )
        
        writer = cv2.VideoWriter(
            pipeline,
            cv2.CAP_GSTREAMER,
            0,
            self.fps,
            (self.frame_w, self.frame_h),
            True
        )

        with self.lock:
            if writer.isOpened():
                self.video_writer = writer
                logger.info(f'開始把 frame 寫入影片: {self.video_path}')
                logger.info(f'影片設定: {pipeline}')
            else:
                self.video_writer = None
                logger.error(f'{traceback.format_exc()}')
                raise Exception('創建 cv2.VideoWriter 失敗')

    def _init_time_writer(self):
        with self.lock:
            try:
                self.time_writer = open(self.time_path, 'w')
                logger.info(f'開始把 frame 時戳寫入檔案: {self.time_path}')
            except:
                self.time_writer = None
                logger.error(f'{traceback.format_exc()}')
                raise Exception('創建 time writer 失敗')

    def _change_period(self):
        idx = self._find_period()
        return idx != self.period_idx

    def _clear_queue(self):
        while not self.frame_q.empty():
            try:
                self.frame_q.get_nowait()
            except queue.Empty:
                break
        logger.info('已清空剩餘的 frame queue')

    def _start_write(self):
        self.running = True
        self.write_thread = Thread(target=self._write, daemon=True)
        self.write_thread.start()

    def _stop_write(self):
        self.running = False
        if self.write_thread:
            self.write_thread.join(2.)
        self.write_thread = None
    
    def _new_path(self, path):
        path = p(path)
        paths = sorted(mp4_path for mp4_path in p(dirname(path)).glob('*.mp4') 
                       if mp4_path.stem.startswith(path.stem))
        
        if len(paths) > 0:
            path = p(paths[-1]).with_suffix(path.suffix)

        parts = path.stem.split('_')
        last = parts[-1]
        if last[0] == 'p':
            parts[-1] = f'p{int(last[1:]) + 1}'
        else:
            parts[-1] = parts[-1] + '_p2'
        
        new_stem = '_'.join(parts)
        new_path = str(path.with_stem(new_stem))
        return new_path
