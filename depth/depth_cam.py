from pathlib import Path as p
import cv2
from loguru import logger
import traceback
from time import time
import numpy as np
from queue import Queue, Empty, Full
from threading import Thread
import os
from os.path import dirname
from datetime import datetime
from pyorbbecsdk import Config, Context, OBSensorType, Pipeline, get_version



class ORBBEC_CAMERA:
    def __init__(self, custom_cfg):
        self.custom_cfg = custom_cfg
        self.enable = custom_cfg['enable_camera']
        self.stream_thread = None
        self.is_running = False
        self.is_first_frame = True
        self.is_terminated = False
        self.pipeline = None
        self.frame_q = Queue(1)
        if not self.enable:
            logger.warning('深度相機不會開啟, 因為 config.yaml 設定成 disable !')

    def get_frame(self, timeout=0.1):
        if not self.enable:
            return

        try:
            frame = self.frame_q.get(timeout=timeout)
        except Empty:
            frame = None
        except:
            logger.error(traceback.format_exc())
        return frame

    def start(self):
        if self.is_terminated:
            logger.error('這個深度相機物件已經結束, 請重新創建一個物件')
            return
        if not self.enable:
            logger.error('目前 config 設定為 disable !')
            return
        self.is_running = True
        self.stream_thread = Thread(target=self._run, daemon=True)
        self.stream_thread.start()

    def stop(self):
        if not self.enable:
            return
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(2)
            self.stream_thread = None
        self.pipeline.stop()
        self.is_terminated = True
        logger.success('深度相機已經關閉 !')

    @staticmethod
    def draw(frame):
        heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        heatmap[frame == 0] = 0
        return heatmap

    def _run(self):
        self._init()

        try:
            while self.is_running:
                frames = self.pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                frame = frames.get_depth_frame()
                if frame is None:
                    continue

                width = frame.get_width()
                height = frame.get_height()
                scale = frame.get_depth_scale()

                raw = np.frombuffer(frame.get_data(), dtype=np.uint16)
                depth_mm = raw.reshape(height, width).astype(np.float32) * scale

                image = np.clip(depth_mm, 0, self.max_depth_mm)
                image = (image * 255 / self.max_depth_mm).astype(np.uint8)

                # 印 shape
                if self.is_first_frame:
                    logger.info(f'深度圖高寬: {image.shape}')
                    self.is_first_frame = False

                # 存當下幀
                if self.frame_q.full():
                    try:
                        self.frame_q.get_nowait()
                    except Empty:
                        pass
                self.frame_q.put(image)

                # FPS 
                self.fps_logger.log()

        except MyError as e:
            logger.error(f'啟動深度相機失敗: {e}')
        except:
            logger.error(f'啟動深度相機失敗: {traceback.format_exc()}')

    def _init(self):
        try:
            self.device_id = self.custom_cfg['id']
            self.max_depth_mm = self.custom_cfg['max_depth_mm']
            self.context = Context()
            self.config = Config()
            self.device = self._get_device(self.device_id)
            self.device_info = self._get_device_info()
            self.pipeline = Pipeline(self.device)
            self.fps_logger = My_Logger(**self.custom_cfg)
            self._start_stream()
            logger.success(f'深度相機初始化完畢 !')
            logger.info(f'\n\tpid: {self.device_info["pid"]:04x}\n'
                        f'\tserial number: {self.device_info["serial number"]}\n'
                        f'\tfirmware version: {self.device_info["firmware version"]}\n'
                        f'\tsdk version: {get_version()}\n'
                        f'\tconnection type: {self.device_info["connection type"]}\n')
            
        except MyError as e:
            logger.error(f'啟動深度相機失敗: {e}')
        except:
            logger.error(f'啟動深度相機失敗: {traceback.format_exc()}')

    def _get_device(self, idx=0):
        devices = self.context.query_devices()
        if devices.get_count() == 0:
            raise MyError("no Orbbec camera detected")
        device = devices.get_device_by_index(idx)
        return device

    def _get_device_info(self):
        device_info = self.device.get_device_info()
        name = device_info.get_name()
        pid = device_info.get_pid()
        serial = device_info.get_serial_number()
        firmware = device_info.get_firmware_version()
        connection = device_info.get_connection_type()
        info = {'name': name, 'pid': pid, 'serial number': serial, 
                'firmware version': firmware, 'connection type': connection}
        return info

    def _start_stream(self):
        profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        profile = profiles.get_default_video_stream_profile()
        if profile is None:
            raise MyError("no default depth profile is available")
        logger.info(f"Profile: {profile}")
        self.config.enable_stream(profile)
        self.pipeline.start(self.config)


class My_Logger:
    def __init__(self, fps_log_interval, log_tag, enable_fps_log, **kwargs):
        self.t = time()
        self.interval = fps_log_interval
        self.count = 0
        self.name = log_tag
        self.enable = enable_fps_log

    def log(self):
        if not self.enable:
            return
        self.count += 1
        cur = time()
        elapsed = cur - self.t
        if elapsed > self.interval:
            logger.warning(f'[{self.name}] FPS: {self.count / elapsed:.3f}')
            self.count = 0
            self.t = cur


class MyError(Exception):
    pass


