"""
Orbbec 深度摄像头模块
通过 ctypes 封装 libOrbbecSDK.so (v1.10.27)，替换 D435 的 pyrealsense2
支持 Astra Mini S 等 OpenNI 协议设备

功能:
  - RGB + Depth 双流获取
  - D2C (Depth-to-Color) 对齐
  - 2D 像素坐标 -> 3D 空间坐标转换
  - 深度图可视化
  - 相机内参保存
"""

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_bool, c_int, c_int16, c_uint32, c_uint64,
    c_float, c_double, c_uint8, c_uint16,
    POINTER, Structure, CFUNCTYPE, byref, cast, sizeof, pointer
)
import numpy as np
import cv2
import os
import json
from pathlib import Path


# ===== SDK 库路径 =====
_SDK_LIB_PATH = os.environ.get(
    "ORBBEC_SDK_LIB",
    "/home/jiaxintang/Orbbec/OrbbecSDK_C_C++_v1.10.27_20250925_0549823_linux_x64_release/"
    "OrbbecSDK_v1.10.27/SDK/lib/libOrbbecSDK.so"
)

# ===== 枚举常量 =====
OB_SENSOR_COLOR = 2
OB_SENSOR_DEPTH = 3
OB_SENSOR_IR = 1

OB_STREAM_COLOR = 2
OB_STREAM_DEPTH = 3

OB_FRAME_COLOR = 2
OB_FRAME_DEPTH = 3

OB_FORMAT_RGB = 22
OB_FORMAT_BGR = 23
OB_FORMAT_YUYV = 0
OB_FORMAT_UYVY = 2
OB_FORMAT_Y16 = 8
OB_FORMAT_Z16 = 28
OB_FORMAT_Y11 = 10
OB_FORMAT_MJPG = 5
OB_FORMAT_ANY = 0xff

ALIGN_D2C_HW_MODE = 0
ALIGN_D2C_SW_MODE = 1
ALIGN_DISABLE = 2

OB_LOG_SEVERITY_OFF = 5
OB_LOG_SEVERITY_ERROR = 3
OB_LOG_SEVERITY_WARN = 2
OB_LOG_SEVERITY_INFO = 1

OB_WIDTH_ANY = 0
OB_HEIGHT_ANY = 0
OB_FPS_ANY = 0

# Frame format enum
OB_FRAME_VIDEO = 0
OB_FRAME_IR = 1
OB_FRAME_COLOR_V = 2
OB_FRAME_DEPTH_V = 3
OB_FRAME_SET = 7
OB_FRAME_POINTS = 8

# =====  SDK 库加载 =====
_lib = None


def _get_lib():
    global _lib
    if _lib is not None:
        return _lib

    lib_dir = os.path.dirname(_SDK_LIB_PATH)
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = lib_dir + ":" + ld_path

    try:
        _lib = ctypes.cdll.LoadLibrary(_SDK_LIB_PATH)
        print(f"loaded SDK: {_SDK_LIB_PATH}")
    except OSError as e:
        raise RuntimeError(
            f"cannot load libOrbbecSDK.so: {e}\n"
            f"path: {_SDK_LIB_PATH}\n"
            f"set env ORBBEC_SDK_LIB if the path is wrong"
        )
    return _lib


# ===== 结构体 (SDK uses #pragma pack(1)) =====

class OBCameraIntrinsic(Structure):
    _pack_ = 1
    _fields_ = [
        ("fx", c_float), ("fy", c_float),
        ("cx", c_float), ("cy", c_float),
        ("width", c_int16), ("height", c_int16),
    ]


class OBCameraDistortion(Structure):
    _pack_ = 1
    _fields_ = [
        ("k1", c_float), ("k2", c_float), ("k3", c_float),
        ("k4", c_float), ("k5", c_float), ("k6", c_float),
        ("p1", c_float), ("p2", c_float),
    ]


class OBD2CTransform(Structure):
    _pack_ = 1
    _fields_ = [
        ("rot", c_float * 9), ("trans", c_float * 3),
    ]


class OBCameraParam(Structure):
    _pack_ = 1
    _fields_ = [
        ("depthIntrinsic", OBCameraIntrinsic),
        ("rgbIntrinsic", OBCameraIntrinsic),
        ("depthDistortion", OBCameraDistortion),
        ("rgbDistortion", OBCameraDistortion),
        ("transform", OBD2CTransform),
        ("isMirrored", c_bool),
    ]


# ===== 函数包装 =====

def _f(name, restype, *argtypes):
    lib = _get_lib()
    func = getattr(lib, name)
    func.restype = restype
    func.argtypes = argtypes
    return func


# Context
_ob_create_context = None
_ob_delete_context = None
_ob_query_device_list = None
_ob_set_logger_severity = None

# Device list
_ob_device_list_device_count = None
_ob_device_list_get_device = None
_ob_delete_device_list = None
_ob_delete_device = None

# Pipeline
_ob_create_pipeline_with_device = None
_ob_delete_pipeline = None
_ob_pipeline_start_with_config = None
_ob_pipeline_stop = None
_ob_pipeline_wait_for_frameset = None
_ob_pipeline_get_camera_param = None

# Config
_ob_create_config = None
_ob_delete_config = None
_ob_config_enable_video_stream = None
_ob_config_set_align_mode = None

# Frame
_ob_frameset_depth_frame = None
_ob_frameset_color_frame = None
_ob_frame_data = None
_ob_frame_data_size = None
_ob_frame_format = None
_ob_frame_get_type = None
_ob_video_frame_width = None
_ob_video_frame_height = None
_ob_depth_frame_get_value_scale = None
_ob_delete_frame = None


def _init_functions():
    global _ob_create_context, _ob_delete_context
    global _ob_query_device_list, _ob_set_logger_severity
    global _ob_device_list_device_count, _ob_device_list_get_device
    global _ob_delete_device_list, _ob_delete_device
    global _ob_create_pipeline_with_device, _ob_delete_pipeline
    global _ob_pipeline_start_with_config, _ob_pipeline_stop
    global _ob_pipeline_wait_for_frameset, _ob_pipeline_get_camera_param
    global _ob_create_config, _ob_delete_config
    global _ob_config_enable_video_stream, _ob_config_set_align_mode
    global _ob_frameset_depth_frame, _ob_frameset_color_frame
    global _ob_frame_data, _ob_frame_data_size
    global _ob_frame_format, _ob_frame_get_type
    global _ob_video_frame_width, _ob_video_frame_height
    global _ob_depth_frame_get_value_scale, _ob_delete_frame

    if _ob_create_context is not None:
        return

    _ob_create_context = _f("ob_create_context", c_void_p, POINTER(c_void_p))
    _ob_delete_context = _f("ob_delete_context", None, c_void_p, POINTER(c_void_p))
    _ob_query_device_list = _f("ob_query_device_list", c_void_p, c_void_p, POINTER(c_void_p))
    _ob_set_logger_severity = _f("ob_set_logger_severity", None, c_int, POINTER(c_void_p))

    _ob_device_list_device_count = _f("ob_device_list_device_count", c_uint32, c_void_p, POINTER(c_void_p))
    _ob_device_list_get_device = _f("ob_device_list_get_device", c_void_p, c_void_p, c_uint32, POINTER(c_void_p))
    _ob_delete_device_list = _f("ob_delete_device_list", None, c_void_p, POINTER(c_void_p))
    _ob_delete_device = _f("ob_delete_device", None, c_void_p, POINTER(c_void_p))

    _ob_create_pipeline_with_device = _f("ob_create_pipeline_with_device", c_void_p, c_void_p, POINTER(c_void_p))
    _ob_delete_pipeline = _f("ob_delete_pipeline", None, c_void_p, POINTER(c_void_p))
    _ob_pipeline_start_with_config = _f("ob_pipeline_start_with_config", None, c_void_p, c_void_p, POINTER(c_void_p))
    _ob_pipeline_stop = _f("ob_pipeline_stop", None, c_void_p, POINTER(c_void_p))
    _ob_pipeline_wait_for_frameset = _f("ob_pipeline_wait_for_frameset", c_void_p, c_void_p, c_uint32, POINTER(c_void_p))
    _ob_pipeline_get_camera_param = _f("ob_pipeline_get_camera_param", OBCameraParam, c_void_p, POINTER(c_void_p))

    _ob_create_config = _f("ob_create_config", c_void_p, POINTER(c_void_p))
    _ob_delete_config = _f("ob_delete_config", None, c_void_p, POINTER(c_void_p))
    _ob_config_enable_video_stream = _f("ob_config_enable_video_stream", None, c_void_p, c_int, c_int, c_int, c_int, c_int, POINTER(c_void_p))
    _ob_config_set_align_mode = _f("ob_config_set_align_mode", None, c_void_p, c_int, POINTER(c_void_p))

    _ob_frameset_depth_frame = _f("ob_frameset_depth_frame", c_void_p, c_void_p, POINTER(c_void_p))
    _ob_frameset_color_frame = _f("ob_frameset_color_frame", c_void_p, c_void_p, POINTER(c_void_p))
    _ob_frame_data = _f("ob_frame_data", c_void_p, c_void_p, POINTER(c_void_p))
    _ob_frame_data_size = _f("ob_frame_data_size", c_uint32, c_void_p, POINTER(c_void_p))
    _ob_frame_format = _f("ob_frame_format", c_int, c_void_p, POINTER(c_void_p))
    _ob_frame_get_type = _f("ob_frame_get_type", c_int, c_void_p, POINTER(c_void_p))
    _ob_video_frame_width = _f("ob_video_frame_width", c_uint32, c_void_p, POINTER(c_void_p))
    _ob_video_frame_height = _f("ob_video_frame_height", c_uint32, c_void_p, POINTER(c_void_p))
    _ob_depth_frame_get_value_scale = _f("ob_depth_frame_get_value_scale", c_float, c_void_p, POINTER(c_void_p))
    _ob_delete_frame = _f("ob_delete_frame", None, c_void_p, POINTER(c_void_p))


def _ne():
    """创建空 error 指针"""
    return c_void_p(0)


class OBError(Exception):
    pass


# ===== OrbbecCamera 类 =====

class OrbbecCamera:
    """Orbbec 深度摄像头 (Astra Mini S 等 OpenNI 协议设备)

    用法:
        camera = OrbbecCamera(width=640, height=480, fps=30)
        color_img, depth_img, depth_data = camera.get_frames()
        point_3d = camera.get_3d_point(x, y, depth_data)
        camera.stop()
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self._context = None
        self._device = None
        self._pipeline = None
        self._config = None
        self._started = False

        # Intrinsics
        self.intrinsics = None
        self.depth_intrinsics = None
        self.depth_scale = 0.001  # default: 1mm per unit

        # Color format detection
        self._color_format = OB_FORMAT_RGB

        _init_functions()

        # Suppress SDK log output
        e = _ne()
        _ob_set_logger_severity(OB_LOG_SEVERITY_OFF, byref(e))

        # Create context
        self._context = _ob_create_context(byref(e))
        if not self._context:
            raise OBError("failed to create SDK context")

        # Enumerate devices
        dev_list = _ob_query_device_list(self._context, byref(e))
        if not dev_list:
            raise OBError("failed to enumerate devices")

        count = _ob_device_list_device_count(dev_list, byref(e))
        if count == 0:
            _ob_delete_device_list(dev_list, byref(e))
            raise OBError("no Orbbec device found, check USB connection")

        print(f"found {count} device(s)")

        # Open first device
        self._device = _ob_device_list_get_device(dev_list, 0, byref(e))
        _ob_delete_device_list(dev_list, byref(e))
        if not self._device:
            raise OBError("failed to open device")

        # Create pipeline
        self._pipeline = _ob_create_pipeline_with_device(self._device, byref(e))
        if not self._pipeline:
            raise OBError("failed to create pipeline")

        # Create config
        self._config = _ob_create_config(byref(e))
        if not self._config:
            raise OBError("failed to create config")

        # Enable color stream - use ANY format, let SDK choose
        _ob_config_enable_video_stream(
            self._config, OB_STREAM_COLOR,
            width, height, fps, OB_FORMAT_ANY, byref(e))

        # Enable depth stream - use ANY format, SDK defaults to Y11 for Astra Mini S
        _ob_config_enable_video_stream(
            self._config, OB_STREAM_DEPTH,
            width, height, fps, OB_FORMAT_ANY, byref(e))

        # 关掉 D2C 对齐，直接用深度内参计算 3D 坐标
        # D2C 对齐会扭曲深度值导致测距不准
        _ob_config_set_align_mode(self._config, ALIGN_DISABLE, byref(e))
        self._d2c_mode = "off"

        # Start pipeline
        _ob_pipeline_start_with_config(self._pipeline, self._config, byref(e))
        self._started = True

        # Get intrinsics from first frame
        self._init_intrinsics()

        print(f"Orbbec camera ready: {width}x{height}@{fps}fps, "
              f"D2C={self._d2c_mode}, depth_scale={self.depth_scale:.4f}")

    def _init_intrinsics(self):
        """获取相机内参 (pipeline 启动后调用)"""
        e = _ne()

        # Wait for first frameset to ensure pipeline is streaming
        for _ in range(50):
            fs = _ob_pipeline_wait_for_frameset(self._pipeline, 100, byref(e))
            if fs:
                _ob_delete_frame(fs, byref(e))
                break

        params = _ob_pipeline_get_camera_param(self._pipeline, byref(e))

        intr = params.rgbIntrinsic
        self.intrinsics = type('Intrinsics', (), {
            'fx': intr.fx, 'fy': intr.fy,
            'ppx': intr.cx, 'ppy': intr.cy,
            'width': intr.width, 'height': intr.height,
        })()

        dintr = params.depthIntrinsic
        self.depth_intrinsics = type('Intrinsics', (), {
            'fx': dintr.fx, 'fy': dintr.fy,
            'ppx': dintr.cx, 'ppy': dintr.cy,
            'width': dintr.width, 'height': dintr.height,
        })()

    def get_frames(self):
        """获取对齐的 Depth + Color 帧

        Returns:
            color_image: BGR numpy array
            depth_image: 16-bit numpy array (raw * scale = mm)
            depth_frame_data: same as depth_image (for get_3d_point)
        """
        if not self._started:
            return None, None, None

        e = _ne()
        fs = _ob_pipeline_wait_for_frameset(self._pipeline, 1000, byref(e))
        if not fs:
            return None, None, None

        depth_frame = None
        color_frame = None
        try:
            depth_frame = _ob_frameset_depth_frame(fs, byref(e))
            color_frame = _ob_frameset_color_frame(fs, byref(e))

            if not depth_frame or not color_frame:
                return None, None, None

            # --- Depth data ---
            depth_data = _ob_frame_data(depth_frame, byref(e))
            dw = _ob_video_frame_width(depth_frame, byref(e))
            dh = _ob_video_frame_height(depth_frame, byref(e))

            # Get depth value scale (once)
            if self.depth_scale < 0.0011:
                s = _ob_depth_frame_get_value_scale(depth_frame, byref(e))
                if s > 0:
                    self.depth_scale = s
                    print(f"  depth value scale: {self.depth_scale}")

            depth_image = np.zeros((dh, dw), dtype=np.uint16)
            if depth_data and dw * dh > 0:
                ctypes.memmove(
                    depth_image.ctypes.data_as(POINTER(c_uint16)),
                    depth_data, dh * dw * 2)

            # --- Color data ---
            color_data = _ob_frame_data(color_frame, byref(e))
            cw = _ob_video_frame_width(color_frame, byref(e))
            ch = _ob_video_frame_height(color_frame, byref(e))
            cf = _ob_frame_format(color_frame, byref(e))

            if color_data and cw * ch > 0:
                if cf in (OB_FORMAT_RGB, OB_FORMAT_BGR):
                    raw = np.zeros((ch, cw, 3), dtype=np.uint8)
                    ctypes.memmove(
                        raw.ctypes.data_as(POINTER(c_uint8)),
                        color_data, ch * cw * 3)
                    if cf == OB_FORMAT_RGB:
                        color_image = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                    else:
                        color_image = raw
                elif cf in (OB_FORMAT_YUYV, OB_FORMAT_UYVY):
                    raw = np.zeros((ch, cw, 2), dtype=np.uint8)
                    ctypes.memmove(
                        raw.ctypes.data_as(POINTER(c_uint8)),
                        color_data, ch * cw * 2)
                    if cf == OB_FORMAT_YUYV:
                        color_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_YUYV)
                    else:
                        color_image = cv2.cvtColor(raw, cv2.COLOR_YUV2BGR_UYVY)
                elif cf == OB_FORMAT_MJPG:
                    buf = np.ctypeslib.as_array(
                        cast(color_data, POINTER(c_uint8)),
                        shape=(_ob_frame_data_size(color_frame, byref(e)),))
                    color_image = cv2.imdecode(buf.copy(), cv2.IMREAD_COLOR)
                else:
                    raw = np.zeros((ch, cw, 3), dtype=np.uint8)
                    ctypes.memmove(
                        raw.ctypes.data_as(POINTER(c_uint8)),
                        color_data,
                        min(ch * cw * 3,
                            _ob_frame_data_size(color_frame, byref(e))))
                    color_image = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            else:
                color_image = None

            return color_image, depth_image, depth_image

        finally:
            # 必须逐个释放帧，否则 SDK buffer pool 会耗尽
            if depth_frame:
                _ob_delete_frame(depth_frame, byref(e))
            if color_frame:
                _ob_delete_frame(color_frame, byref(e))
            if fs:
                _ob_delete_frame(fs, byref(e))

    def get_depth_colormap(self, depth_image, min_depth=0.0, max_depth=10.0):
        """深度图彩色可视化

        Args:
            depth_image: 原始深度图 (raw value * depth_scale = mm)
            min_depth:   最小深度 (米)
            max_depth:   最大深度 (米)

        Returns:
            BGR colormap
        """
        depth_m = depth_image.astype(np.float32) * self.depth_scale / 1000.0
        mask = (depth_m >= min_depth) & (depth_m <= max_depth)
        masked = depth_image.copy()
        masked[~mask] = 0
        return cv2.applyColorMap(
            cv2.convertScaleAbs(masked, alpha=0.03), cv2.COLORMAP_JET)

    def get_3d_point(self, x, y, depth_frame_data, sample_radius=5):
        """像素坐标 -> 3D 世界坐标 (以米为单位)

        D2C 对齐后目标中心可能出现深度空洞，因此在中心周围区域采样，
        取非零深度的中位数。

        使用小孔成像模型:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth_raw * depth_scale / 1000.0  (raw*depth_scale = mm)

        Args:
            x:               像素 x
            y:               像素 y
            depth_frame_data: numpy depth array (raw values)
            sample_radius:   采样半径 (像素)

        Returns:
            {'x', 'y', 'z', 'distance'} in meters, or None
        """
        if self.intrinsics is None or depth_frame_data is None:
            return None

        h, w = depth_frame_data.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return None

        # 区域采样：取非零深度的中位数
        r = sample_radius
        y1, y2 = max(0, y - r), min(h, y + r + 1)
        x1, x2 = max(0, x - r), min(w, x + r + 1)
        region = depth_frame_data[y1:y2, x1:x2]
        valid = region[region > 0]
        if len(valid) == 0:
            return None

        depth_raw = float(np.median(valid))
        z_mm = depth_raw * self.depth_scale
        z_m = z_mm / 1000.0
        if z_m <= 0:
            return None

        # 无 D2C 对齐，深度像素坐标使用深度内参
        intr = self.depth_intrinsics if self.depth_intrinsics else self.intrinsics
        x_m = (x - intr.ppx) * z_m / intr.fx
        y_m = (y - intr.ppy) * z_m / intr.fy

        dist = np.sqrt(x_m**2 + y_m**2 + z_m**2)
        return {'x': x_m, 'y': y_m, 'z': z_m, 'distance': dist}

    def stop(self):
        """停止摄像头，释放资源"""
        e = _ne()
        if self._started and self._pipeline:
            _ob_pipeline_stop(self._pipeline, byref(e))
            self._started = False

        if self._pipeline:
            _ob_delete_pipeline(self._pipeline, byref(e))
            self._pipeline = None
        if self._config:
            _ob_delete_config(self._config, byref(e))
            self._config = None
        if self._device:
            _ob_delete_device(self._device, byref(e))
            self._device = None
        if self._context:
            _ob_delete_context(self._context, byref(e))
            self._context = None

        print("Orbbec camera stopped")
