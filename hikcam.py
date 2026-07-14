# -- coding: utf-8 --

import sys
import os
import platform
from ctypes import *
import cv2
from loguru import logger
import numpy as np
import traceback
import ctypes
import re
import time
import yaml
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from config import Config
from mmr_engine import Rotated_RTMDET
from config import _load_yaml
from mqtt_bus import MqttBus



# 兼容不同操作系统加载 动态库
currentsystem = platform.system()
if currentsystem == 'Windows':
    sys.path.append(os.path.join(os.getenv('MVCAM_COMMON_RUNENV'), "Samples", "Python", "MvImport"))
else:
    sys.path.append(os.path.join("..", "..", "MvImport"))

try:
    from MvCameraControl_class import *
except Exception as e:
    logger.warning(f'{e}')

# 兼容Python 2.x和3.x的输入处理
if sys.version_info[0] < 3:
    input_func = raw_input
else:
    input_func = input


# ---------------------------------------------------------------------------
# HikDeviceManager：負責「與單一相機無關」的全域層級操作
#   - 字串解碼（decoding_char）
#   - 列舉裝置（enum_devices）
#   - SDK 初始化 / 反初始化（initialize / finalize），方便搭配 with 使用
# ---------------------------------------------------------------------------
class HikDeviceManager:
    """
    負責 SDK 初始化/反初始化、列舉裝置、字串解碼等與單一相機無關的全域操作。

    一般使用者不需要直接操作這個 class —— HikCamera 內部已經自動建立並管理一個
    HikDeviceManager 實例。只有在你想自己客製化裝置列舉流程時才需要直接用它，例如:

        mgr = HikDeviceManager()
        mgr.initialize()
        device_list = mgr.enum_devices()
        mgr.finalize()
    """

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
        return False

    # ---------------- SDK 初始化 ----------------
    @staticmethod
    def initialize():
        """初始化 SDK，並印出版本號"""
        MvCamera.MV_CC_Initialize()
        sdk_version = MvCamera.MV_CC_GetSDKVersion()
        logger.info("SDKVersion[0x%x]" % sdk_version)
        return sdk_version

    @staticmethod
    def finalize():
        """反初始化 SDK"""
        MvCamera.MV_CC_Finalize()

    # ---------------- 字串解碼 ----------------
    @staticmethod
    def decoding_char(ctypes_char_array):
        """
        安全地从 ctypes 字符数组中解码出字符串。
        适用于 Python 2.x 和 3.x，以及 32/64 位环境。
        """
        byte_str = memoryview(ctypes_char_array).tobytes()

        null_index = byte_str.find(b'\x00')
        if null_index != -1:
            byte_str = byte_str[:null_index]

        for encoding in ['gbk', 'utf-8', 'latin-1']:
            try:
                return byte_str.decode(encoding)
            except UnicodeDecodeError:
                continue

        return byte_str.decode('latin-1', errors='replace')

    # ---------------- 列舉裝置 ----------------
    def enum_devices(self):
        """
        列舉所有可用裝置，回傳 deviceList (MV_CC_DEVICE_INFO_LIST)
        """
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
                      | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            raise RuntimeError("enum devices fail! ret[0x%x]" % ret)

        if deviceList.nDeviceNum == 0:
            raise RuntimeError("find no device!")

        logger.info("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            self._print_device_info(deviceList, i)

        return deviceList

    def _print_device_info(self, deviceList, index):
        """印出單一裝置的型號、序號/IP 等資訊"""
        mvcc_dev_info = cast(deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        decode = self.decoding_char

        if mvcc_dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            logger.info("gige device: [%d]" % index)
            logger.info("device model name: %s" % decode(
                mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName))

            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            logger.info("current ip: %d.%d.%d.%d" % (nip1, nip2, nip3, nip4))
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            logger.info("u3v device: [%d]" % index)
            logger.info("device model name: %s" % decode(
                mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName))
            logger.info("user serial number: %s" % decode(
                mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber))
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CAMERALINK_DEVICE:
            logger.info("CML device: [%d]" % index)
            logger.info("device model name: %s" % decode(
                mvcc_dev_info.SpecialInfo.stCMLInfo.chModelName))
            logger.info("user serial number: %s" % decode(
                mvcc_dev_info.SpecialInfo.stCMLInfo.chSerialNumber))
        elif mvcc_dev_info.nTLayerType == MV_GENTL_CXP_DEVICE:
            logger.info("CXP device: [%d]" % index)
            logger.info("device model name: %s" % decode(
                mvcc_dev_info.SpecialInfo.stCXPInfo.chModelName))
            logger.info("user serial number: %s" % decode(
                mvcc_dev_info.SpecialInfo.stCXPInfo.chSerialNumber))
        elif mvcc_dev_info.nTLayerType == MV_GENTL_XOF_DEVICE:
            logger.info("XoF device: [%d]" % index)
            logger.info("device model name: %s" % decode(
                mvcc_dev_info.SpecialInfo.stXoFInfo.chModelName))
            logger.info("user serial number: %s" % decode(
                mvcc_dev_info.SpecialInfo.stXoFInfo.chSerialNumber))

    # ---------------- 取單一裝置資訊（給 HikCamera 用） ----------------
    @staticmethod
    def get_device_info(deviceList, index):
        """
        從 deviceList 中取出指定 index 的 MV_CC_DEVICE_INFO 結構，
        供 HikCamera(device_info) 使用。
        """
        if index >= deviceList.nDeviceNum:
            raise ValueError("device index %d out of range (found %d devices)"
                              % (index, deviceList.nDeviceNum))
        return cast(deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents


# ---------------------------------------------------------------------------
# 主角：HikCamera class
# ---------------------------------------------------------------------------
class HikCamera:
    """
    封裝海康相機的 SDK 初始化、裝置列舉、開啟、參數設定、取像、存檔、關閉流程。

    內部會自動建立並管理一個 HikDeviceManager，負責 SDK 的初始化/反初始化與裝置列舉，
    所以外部只需要面對 HikCamera 這一個物件即可。

    用法 (with，自動釋放):
        with HikCamera(device_index=0, cfg=cfg) as hc:
            ok, img = hc.read()
            hc.save_image(img, save_type=1)

    用法 (手動釋放):
        hc = HikCamera(device_index=0, cfg=cfg)
        ok, img = hc.read()
        hc.save_image(img)
        hc.release()    # 一行關掉相機 + SDK
    """

    # Mono8 的像素格式碼（用於判斷是否為黑白相機）
    PIXEL_TYPE_MONO8 = 17301505

    def __init__(self, device_index, cfg: Config, mqtt: MqttBus):
        """
        :param device_index: 要開啟的裝置在列舉結果中的索引
        :param cfg: Config 物件，內含 runtime.camera.source 與 camera_params 等設定
        """
        self.device_index = device_index
        self.grab_timeout_ms = 5000

        # HikCamera 內部自己持有一個 HikDeviceManager，負責 SDK 生命週期與列舉裝置
        self.device_manager = HikDeviceManager()
        self.device_info = None   # 開啟後才會填入，供除錯/查詢用

        self.cam = MvCamera()
        self.cfg = cfg
        self.cfg_path = Path("tasks") / 'ocr' / "config.yaml"
        self._is_sdk_initialized = False
        self._is_open = False
        self._is_grabbing = False
        self.stOutFrame = None
        self.dst_buffer_size = None
        self.dst_buffer = None
        self.dst_ptr = None

        # mqtt
        self.mqtt = mqtt

        device_count = HikCamera.list_available_devices()
        if self.cfg.runtime.camera.source >= device_count:
            raise ValueError(f"valid camera ID are {range(0, device_count + 1)}")

        try:
            self.open()
        except Exception:
            self.release()
            logger.error(traceback.format_exc())
            raise

        # --- parameter update
        self._config_dirty = False
        self._config_last_update = 0.0
        self._config_lock = threading.Lock()
        self.save_delay = 5.0

        self._config_save_thread = threading.Thread(
            target=self._config_save_worker,
            daemon=True
        )
        self._config_save_thread.start()

    # ---------------- 列舉裝置（不需要知道 HikDeviceManager 也能用） ----------------
    @classmethod
    def list_available_devices(cls):
        """
        列舉目前可用的裝置數量（並印出每台裝置的型號/序號等資訊），方便使用者
        決定要用哪個 device_index 來建立 HikCamera。

        內部會短暫初始化 SDK、列舉、再反初始化，跟之後 HikCamera(device_index=...).open()
        各自獨立，互不影響。

        :return: 找到的裝置數量 (int)
        """
        mgr = HikDeviceManager()
        mgr.initialize()
        try:
            device_list = mgr.enum_devices()
            return device_list.nDeviceNum
        finally:
            mgr.finalize()

    # ---------------- context manager 支援 ----------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        # 不吞掉例外，讓外層照常看到錯誤
        return False

    # ---------------- 開啟相機與設定參數 ----------------
    def open(self):
        """
        完整開啟流程：初始化 SDK → 列舉裝置 → 取得指定裝置 → 建立 handle →
        開裝置 → 設定參數 → 開始取流
        """
        # 1. 初始化 SDK（透過內部的 device_manager）
        self.device_manager.initialize()
        self._is_sdk_initialized = True

        # 2. 列舉裝置，取得指定 index 的裝置資訊
        device_list = self.device_manager.enum_devices()
        self.device_info = self.device_manager.get_device_info(device_list, self.device_index)

        # 3. 建立 handle
        ret = self.cam.MV_CC_CreateHandle(self.device_info)
        if ret != 0:
            raise RuntimeError("create handle fail! ret[0x%x]" % ret)

        # 4. 開裝置；若失敗，handle 已建立但裝置未開，要銷毀 handle 再往外丟例外
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError("open device fail! ret[0x%x]" % ret)
        self._is_open = True

        # ch:探测网络最佳包大小(只对GigE相机有效)
        if self.device_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    logger.warning("Set Packet Size fail! ret[0x%x]" % ret)
            else:
                logger.warning("Get Packet Size fail! ret[0x%x]" % nPacketSize)

        # ch:设置触发模式为off；失敗時裝置已開，要關裝置 + 銷毀 handle 再丟例外
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            self.cam.MV_CC_CloseDevice()
            self._is_open = False
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError("set trigger mode fail! ret[0x%x]" % ret)

        # 設定相機參數
        self.set_camera_parameters(display=True)
        self.list_all_camera_parameters()

        # 開始取流；失敗時同樣要關裝置 + 銷毀 handle 再丟例外
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.cam.MV_CC_CloseDevice()
            self._is_open = False
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError("start grabbing fail! ret[0x%x]" % ret)
        self._is_grabbing = True

        # 設定 "去馬賽克" 演算法
        ret = self.cam.MV_CC_SetBayerCvtQuality(1)
        if ret != 0:
            logger.warning(f"Set Bayer Cvt Quality fail! ret[0x{ret:x}]")

        # 創建必要 buffer
        self.stOutFrame = MV_FRAME_OUT()
        memset(byref(self.stOutFrame), 0, sizeof(self.stOutFrame))
        
        w = self.cfg.runtime.camera.width
        h = self.cfg.runtime.camera.height
        self.dst_buffer_size = w * h * 3
        self.dst_buffer = (c_ubyte * self.dst_buffer_size)()
        self.dst_ptr = cast(self.dst_buffer, POINTER(c_ubyte))

        return self

    def set_camera_parameters(self, params=None, display=True):
        """ 如果是透過指令的話, 一次只會有一個參數 """
        from_cmd = params is not None
        
        # command or config
        if from_cmd:
            logger.info('=' * 25 + ' Set Camera Parameter by [command]' + '=' * 25)
            msg = params
            name = msg['control']
            info = self.cfg.camera_params['hik'].get(name)
            if info is None:
                logger.error(f'camera parameter [{name}] is not in the settable list !')
                return
            params = {f'{name}': info}
            params[name]['value'] = msg['value']
            logger.debug(f'new camera parameter: {params}')
        else:
            logger.info('=' * 25 + ' Set Camera Parameter by [config]' + '=' * 25)
            params = self.cfg.camera_params['hik']

        # set
        # 記錄每個參數是否設定成功，失敗的參數不能拿去觸發存 config 流程
        set_success = {}

        for name, info in params.items():
            value = info['value']
            typ = info['type']

            if typ == 'float':
                ret = self.cam.MV_CC_SetFloatValue(name, value)
            elif typ == 'int':
                ret = self.cam.MV_CC_SetIntValue(name, value)
            elif typ == 'bool':
                ret = self.cam.MV_CC_SetBoolValue(name, value)
            elif typ == 'enum':
                ret = self.cam.MV_CC_SetEnumValue(name, value)
            elif typ == 'string':
                ret = self.cam.MV_CC_SetEnumValueByString(name, value)
            else:
                raise ValueError(f'invalid camera parameter type: {typ}')

            set_success[name] = (ret == 0)

            if ret == 0:
                logger.success(f'Set [{name}] to {value} !')
            else:
                logger.warning(f'Set camera parameter [{name}] failed ! ret[0x{ret:x}]')

        logger.info('=' * 60)

        # 印出所有相機參數和有效的範圍值
        if display:
            self.display_camera_parameters()

        if from_cmd:
            if set_success.get(msg['control']):
                self._update_camera_config(msg['control'], msg['value'])
            else:
                logger.warning(
                    f"參數 [{msg['control']}] 設定失敗，不觸發存 config 流程，"
                    f"config.yaml 維持原本的值"
                )
        self.mqtt.publish_system({'type': 'CAMERA_PARAMS_DONE'})

    def reset_camera_parameters(self):
        data = _load_yaml(self.cfg_path)
        need_updates = []
        for name, info in data['camera_params']['hik'].items():
            default = info.get('default')
            if default:
                info['value'] = default
                need_updates.append({'control': name, 'value': default})

        self.cfg.camera_params['hik'] = data['camera_params']['hik']
        
        for params in need_updates:
            self.set_camera_parameters(params, True)

    def display_camera_parameters(self, params=None):
        """
        顯示相機參數的目前值與有效範圍（min/max/step，或 enum 的候選值列表）。

        :param params: 要查詢的參數字典，格式同 cfg.camera_params['hik']
                        （key=參數名, value=dict 含 'type'）。
                        若為 None，預設查詢 self.cfg.camera_params['hik'] 裡的所有參數。
        :return: dict，key=參數名，value=查到的資訊 dict（查詢失敗則為 None）
        """
        if not self._is_open:
            logger.warning("camera is not open, cannot query parameter range!")
            return {}

        if params is None:
            params = self.cfg.camera_params['hik']

        result = {}

        logger.info('=' * 25 + f' Camera Parameter Range ' + '=' * 25)

        for name, info in params.items():
            typ = info['type']

            if typ == 'float':
                val = MVCC_FLOATVALUE()
                ret = self.cam.MV_CC_GetFloatValue(name, val)
                if ret == 0:
                    data = {'cur': val.fCurValue, 'min': val.fMin, 'max': val.fMax}
                    logger.info(f"[{name}] cur={data['cur']}, min={data['min']}, max={data['max']}")
                else:
                    data = None
                    logger.warning(f"cannot read parameter [{name}], ret[0x{ret:x}]")

            elif typ == 'int':
                val = MVCC_INTVALUE_EX()
                ret = self.cam.MV_CC_GetIntValueEx(name, val)
                if ret == 0:
                    data = {'cur': val.nCurValue, 'min': val.nMin, 'max': val.nMax, 'step': val.nInc}
                    logger.info(
                        f"[{name}] cur={data['cur']}, min={data['min']}, "
                        f"max={data['max']}, step={data['step']}"
                    )
                else:
                    data = None
                    logger.warning(f"cannot read parameter [{name}], ret[0x{ret:x}]")

            elif typ == 'bool':
                val = c_bool()
                ret = self.cam.MV_CC_GetBoolValue(name, val)
                if ret == 0:
                    data = {'cur': val.value}
                    logger.info(f"[{name}] cur={data['cur']}")
                else:
                    data = None
                    logger.warning(f"cannot read parameter [{name}], ret[0x{ret:x}]")

            elif typ in ('enum', 'string'):
                val = MVCC_ENUMVALUE()
                ret = self.cam.MV_CC_GetEnumValue(name, val)
                if ret == 0:
                    supported = list(val.nSupportValue[:val.nSupportedNum])
                    data = {'cur': val.nCurValue, 'supported': supported}
                    logger.info(f"[{name}] cur={data['cur']}, supported={data['supported']}")
                else:
                    data = None
                    logger.warning(f"cannot read parameter [{name}], ret[0x{ret:x}]")

            else:
                data = None
                logger.warning(f"unknown parameter type [{typ}] for [{name}], skip")

            result[name] = data

        logger.info('=' * 78)
        return result

    def list_all_camera_parameters(self, xml_path='./camera_features.xml'):
        """
        列出相機所有可用參數（非僅 config 中列出的）。

        做法：
        1. 用 MV_CC_FeatureSave 匯出目前參數快照（注意：這其實是海康的
        GenApi persistence 純文字檔，不是完整 XML feature tree，
        只包含「參數名稱 + 目前值」，不含型別/範圍資訊）。
        2. 從檔案解析出參數名稱清單。
        3. 對每個名稱逐一嘗試 Get*Value API 探測型別。
        4. 複用 display_camera_parameters 查詢完整的 cur/min/max/step 等資訊。
        """
        if not self._is_open:
            logger.warning("camera is not open, cannot query parameter range!")
            return {}

        ret = self.cam.MV_CC_FeatureSave(xml_path)
        if ret != 0:
            logger.warning(f"cannot dump camera feature snapshot, ret[0x{ret:x}]")
            return {}

        names = self._parse_param_names(xml_path)
        logger.info(f"found {len(names)} candidate parameter names")

        if not names:
            logger.warning("no parameter name parsed, check file format manually")
            return {}

        params = {}
        for name in names:
            typ = self._probe_param_type(name)
            if typ:
                params[name] = {'type': typ}
            else:
                logger.warning(f"cannot resolve type for parameter [{name}], skip")

        logger.info(f"resolved {len(params)} / {len(names)} parameters with known type")

        return self.display_camera_parameters(params)

    def _parse_param_names(self, path):
        """從 GenApi persistence 純文字檔解析出參數名稱清單（去除註解與空行）。"""
        names = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # 假設格式："ParamName"    Value  或  ParamName   Value
                # 若實際格式不同，請依 sample lines 調整這個 regex
                m = re.match(r'^"?([A-Za-z_][A-Za-z0-9_]*)"?\s', line)
                if m:
                    names.append(m.group(1))
        return sorted(set(names))

    def _probe_param_type(self, name):
        """依序嘗試 bool / int / float / enum 的 Get 方法，回傳第一個成功的型別。"""
        val = c_bool()
        if self.cam.MV_CC_GetBoolValue(name, val) == 0:
            return 'bool'

        val = MVCC_INTVALUE_EX()
        if self.cam.MV_CC_GetIntValueEx(name, val) == 0:
            return 'int'

        val = MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(name, val) == 0:
            return 'float'

        val = MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue(name, val) == 0:
            return 'enum'

        return None

    def _update_camera_config(self, key: str, value) -> None:
        """
        更新 config.yaml 文件中的 camera 參數
        """
        param_rules = self.cfg.camera_params['hik']
        if key not in param_rules:
            logger.warning(f"參數 '{key}' 不在允許修改，已略過")
            return False

        rule = param_rules[key]
        valid_value = value
        try:
            if rule["type"] == 'string':
                valid_value = str(value).strip()
                if valid_value not in rule["allowed"]:
                    logger.warning(f"參數 '{key}' 的值 '{valid_value}' 不合法，允許的值: {rule['allowed']}")
                    return
            elif rule["type"] == 'float':
                valid_value = float(value)
                if valid_value < rule["min"] or valid_value > rule["max"]:
                    logger.warning(f"參數 '{key}' 的值 '{valid_value}' 超出範圍 ({rule['min']} ~ {rule['max']})")
                    return
        except ValueError:
            logger.error(f"參數 '{key}' 的數值型態錯誤 (無法轉換為 {rule['type'].__name__})")

        self._set_save_config_flag()

        rule['value'] = value
        logger.info(f"[MEMORY UPDATE] the camera parameter [{key}]'s value to {value} !")
        logger.debug(f'new camera config: {self.cfg.camera_params["hik"]}')

    def _config_save_worker(self):
        logger.success('start config update thread!')

        while self._is_open:
            try:
                need_save = False

                with self._config_lock:
                    if (
                        self._config_dirty
                        and time.time() - self._config_last_update >= self.save_delay
                    ):
                        self._config_dirty = False
                        need_save = True

                if need_save:
                    self._save_camera_config()

                time.sleep(0.1)
            except:
                logger.error(traceback.format_exc())

    # ---------------- 取像 ----------------
    def read(self):
        """
        使用 MV_CC_GetImageBuffer 取一張影像（SDK 內部管理 buffer，不自行配置）。
        取得後立刻把資料複製出來（string_at 會做記憶體複製），再呼叫
        MV_CC_FreeImageBuffer 把 buffer 還給 SDK，避免持有 SDK 內部記憶體的參照。

        每次呼叫都使用全新的 MV_FRAME_OUT() 結構（而不是共用同一個 instance attribute），
        避免任何「結構殘留上一輪資料」的可能性。

        回傳已經做好色彩空間轉換的 numpy array (BGR 或 Mono)。
        對應到 frame_info 的部分以 self._last_frame_info 暴露給外部（含寬高、幀號、像素格式）。
        """
        stOutFrame = self.stOutFrame

        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, self.grab_timeout_ms)
        if ret != 0 or not stOutFrame.pBufAddr:
            logger.warning("MV_CC_GetImageBuffer failed! ret[0x%x]" % ret)
            return False, None

        try:
            width = stOutFrame.stFrameInfo.nWidth
            height = stOutFrame.stFrameInfo.nHeight
            pixel_type = stOutFrame.stFrameInfo.enPixelType
            frame_len = stOutFrame.stFrameInfo.nFrameLen
            frame_id = stOutFrame.stFrameInfo.nFrameNum

            w = self.cfg.runtime.camera.width
            h = self.cfg.runtime.camera.height
            if width != w or height != h:
                logger.error(f'frame size is different from setting ! real frame size: {w, h}')
                return False, None

            logger.trace(f'stOutFrame.stFrameInfo.nWidth: {width}')
            logger.trace(f'stOutFrame.stFrameInfo.nHeight: {height}')
            logger.trace(f'stOutFrame.stFrameInfo.enPixelType: {pixel_type}')
            logger.trace(f'stOutFrame.stFrameInfo.nFrameLen: {frame_len}')
            logger.trace(f'stOutFrame.stFrameInfo.nFrameNum: {frame_id}')

            stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(stConvertParam), 0, sizeof(stConvertParam))

            src_ptr = cast(stOutFrame.pBufAddr, POINTER(c_ubyte))

            stConvertParam.nWidth = width
            stConvertParam.nHeight = height
            stConvertParam.pSrcData = src_ptr
            stConvertParam.nSrcDataLen = frame_len
            stConvertParam.enSrcPixelType = pixel_type  # 原始 Bayer 格式
            stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed       # 目標格式,給 OpenCV 用 BGR

            # 計算目的端 buffer 大小,並配置
            stConvertParam.pDstBuffer = self.dst_ptr
            stConvertParam.nDstBufferSize = self.dst_buffer_size

            ret = self.cam.MV_CC_ConvertPixelType(stConvertParam)
            if ret != 0:
                logger.error(f"Convert Pixel Type failed ! ret[0x{ret:x}]")
                return False, None

            img_out = np.frombuffer(self.dst_buffer, dtype=np.uint8).copy()
            img_out = img_out.reshape(height, width, 3)

        finally:
            ret = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
            if ret != 0:
                logger.error(f"Free Image Buffer failed ! ret[0x{ret:x}]")
                return False, None

        return img_out is not None, img_out

    # ---------------- 存檔 ----------------
    def save_image(self, img, save_type=4, file_path=None):
        """
        將 read() 回傳的影像存成檔案。

        :param img: numpy array 影像（BGR 或灰階）
        :param save_type: 0-raw(不支援於此函式,請用 save_raw), 1-jpg, 2-bmp, 3-tif, 4-png
        :param file_path: 自訂檔名；None 則自動依寬高與幀號命名
        """
        ext_map = {1: ".jpg", 2: ".bmp", 3: ".tif", 4: ".png"}
        ext = ext_map.get(int(save_type), ".png")

        if file_path is None:
            fi = getattr(self, "_last_frame_info", None)
            if fi is not None:
                file_path = "Image_w%d_h%d_fn%d%s" % (fi.nWidth, fi.nHeight, fi.nFrameNum, ext)
            else:
                file_path = "Image%s" % ext

        img_to_save = img
        cv2.imwrite(file_path, img_to_save)

        logger.success("影像已完美生成，路徑：%s" % file_path)
        return file_path

    def isOpened(self):
        return self._is_open

    # ---------------- 關閉相機 ----------------
    def release(self):
        """
        完整關閉流程：停止取流 → 關閉裝置 → 銷毀 handle → 反初始化 SDK
        任何步驟失敗都不中斷，盡量釋放所有資源（相機 + SDK 都會處理）。
        """
        if self._is_grabbing:
            ret = self.cam.MV_CC_StopGrabbing()
            if ret != 0:
                logger.warning("stop grabbing fail! ret[0x%x]" % ret)
            self._is_grabbing = False

        if self._is_open:
            ret = self.cam.MV_CC_CloseDevice()
            if ret != 0:
                logger.warning("close device fail! ret[0x%x]" % ret)
            self._is_open = False

        self.cam.MV_CC_DestroyHandle()

        # 連同 SDK 一起反初始化，外部只需呼叫這一個 release() 就能釋放乾淨
        if self._is_sdk_initialized:
            self.device_manager.finalize()
            self._is_sdk_initialized = False
        logger.success('release the camera !')

    def _set_save_config_flag(self):
        with self._config_lock:
            self._config_dirty = True
            self._config_last_update = time.time()
            logger.warning(f'camera config is updated, new config will be saved in {self.save_delay} (s) !')

    def _save_camera_config(self):
        tmp_path = self.cfg_path.with_stem('tmp_' + self.cfg_path.stem)
        data = _load_yaml(self.cfg_path)
        data['camera_params']['hik'] = self.cfg.camera_params['hik']
        with open(tmp_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, indent=4, allow_unicode=True, sort_keys=False)
        os.replace(tmp_path, self.cfg_path)
        logger.success(f"[FILE UPDATE] saved the {self.cfg_path} !")

# ---------------------------------------------------------------------------
# main：互動式選擇裝置，開啟、取一張圖、存檔
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        USE_MMR = False

        logger.remove()
        logger.add(sys.stderr, level='INFO')

        # 步驟 1：列舉裝置數量，讓使用者看清楚有哪些相機可選
        # （HikCamera 內部會自己用它的 HikDeviceManager 完成，外部不需要碰到那個名字）
        device_count = HikCamera.list_available_devices()

        from config import load_config
        cfg = load_config('tasks/ocr/config.yaml')
        class_names = [c.name for c in cfg.classes]

        if USE_MMR:
            mmr = Rotated_RTMDET(
                path=cfg.runtime.model.object_det, 
                classes=class_names, 
                conf_thresh=cfg.thresholds.ai_conf,
                iou_thresh=cfg.thresholds.ai_iou
            )  
        else:
            mmr = None

        # 如果不想用 with，也可以這樣手動寫，效果一樣:
        hc = HikCamera(cfg.runtime.camera.source, cfg=cfg)
        i = 0
        while True:
            ok, img = hc.read()
            if mmr is not None:
                plotted, boxes = mmr.detect(img, True)
            i += 1
            logger.info(f'第 {i} frame')
        hc.release()   # 一行同時關閉相機 + 反初始化 SDK

    except Exception as e:
        logger.error(e)