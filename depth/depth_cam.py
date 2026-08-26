from pathlib import Path as p
import cv2
from loguru import logger
import traceback
from time import time
import numpy as np
from queue import Queue, Empty, Full
from threading import Thread
import yaml
import signal
from pyorbbecsdk import Config, Context, OBSensorType, Pipeline, get_version



class ORBBEC_CAMERA:
    def __init__(self, custom_cfg):
        try:
            self.custom_cfg = custom_cfg
            self.context = Context()
            self.config = Config()
            self.device = self._get_device(self.custom_cfg['id'])
            self.device_info = self._get_device_info()
            self.pipeline = Pipeline(self.device)
            self.stream_thread = None
            self.is_running = False
            self.is_first_frame = True
            self.is_terminated = False
            self.frame_q = Queue(256)
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

    def read(self):
        frame = None
        ret = False
        try:
            frame = self.frame_q.get_nowait()
            ret = True
        except Empty:
            pass
        except:
            logger.error(traceback.format_exc())
        return ret, frame

    def start(self):
        if self.is_terminated:
            logger.error('這個深度相機物件已經結束, 請重新創建一個物件')
            return
        self.is_running = True
        self.stream_thread = Thread(target=self._run, daemon=True)
        self.stream_thread.start()

    def stop(self):
        self.is_running = False
        if self.stream_thread:
            self.stream_thread.join(2)
            self.stream_thread = None
        self.pipeline.stop()
        self.is_terminated = True
        logger.success('深度相機已經關閉 !')

    def _run(self):
        try:
            t = None
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

                image = np.clip(depth_mm, 0, self.custom_cfg['max_depth_mm'])
                image = (image * 255 / self.custom_cfg['max_depth_mm']).astype(np.uint8)

                # 印 shape
                if self.is_first_frame:
                    logger.info(f'深度圖高寬: {image.shape}')
                    self.is_first_frame = False

                # 存當下幀
                try:
                    self.frame_q.put_nowait(image)
                except Full:
                    pass

                # FPS 
                if t is None:
                    t = time()
                else:
                    cur = time()
                    logger.debug(f'FPS: {1 / (cur - t):.2f}')
                    t = cur

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


class MyError(Exception):
    pass



if __name__ == '__main__':
    with open('tasks/ocr/config.yaml') as f:
        cfg = yaml.safe_load(f)['depth_camera']

    cam = ORBBEC_CAMERA(cfg)
    cam.start()

    is_running = True

    def handle_signal(signum, frame):
        global is_running
        is_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while is_running:
            ret, frame = cam.read()
            if not ret:
                continue

            heatmap = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
            heatmap[frame == 0] = 0

            cv2.imshow(f"Depth", heatmap)
            cv2.waitKey(1)
    except Exception:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()
