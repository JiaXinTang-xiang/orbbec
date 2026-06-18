"""
Orbbec + YOLO 目标检测与追踪主程序
功能:
  1. Orbbec 深度摄像头获取 RGB + 深度图
  2. YOLO 检测目标
  3. 抗灯光干扰过滤
  4. 获取目标的三维坐标和距离 (坐标记忆)
  5. PID 追踪控制 (可选)
  6. 串口通讯发送数据 (可选)
  7. 相机内参保存
"""

import os
# 抑制 OpenCV Qt 字体警告 (不影响功能)
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

import cv2
import json
import numpy as np
from pathlib import Path
from orbbec_camera import OrbbecCamera
from detector import YOLODetector
from pid_controller import PIDController
from anti_light import filter_detections
from serial_comm import SerialCommunicator


# ===== 配置 =====
MODEL_PATH = "model/best.pt"       # 模型路径
CONF = 0.5                          # 置信度阈值
IOU = 0.7                           # IoU 阈值
IMAGE_WIDTH = 640                   # 图像宽度
IMAGE_HEIGHT = 480                  # 图像高度
FPS = 30                            # 帧率
SHOW_DEPTH = True                   # 是否显示深度图
USE_PID = False                     # 是否启用 PID 追踪
USE_SERIAL = False                  # 是否启用串口通讯
SERIAL_PORT = "/dev/ttyUSB0"       # 串口路径
MIN_VARIANCE = 100                  # 抗灯光最小方差阈值

# PID 参数
pid_x = PIDController(kp=0.3, ki=0.0, kd=0.1, deadband=10)
pid_y = PIDController(kp=0.3, ki=0.0, kd=0.1, deadband=10)

# 串口数据帧 (19 字节，保留 MTI 格式)
msg = [0, 127, 127, 127, 127, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def save_intrinsics(camera):
    """保存相机内参到 JSON 文件"""
    if camera.intrinsics is None:
        return
    intr = camera.intrinsics
    params = {
        'fx': intr.fx, 'fy': intr.fy,
        'ppx': intr.ppx, 'ppy': intr.ppy,
        'width': intr.width, 'height': intr.height,
        'depth_scale': camera.depth_scale
    }
    with open('intrinsics.json', 'w') as f:
        json.dump(params, f, indent=2)
    print("相机内参已保存到 intrinsics.json")


def update_serial_msg(ser, cx, cy, point_3d, use_pid, center_x, center_y):
    """更新串口数据帧并发送

    Args:
        ser: SerialCommunicator 实例
        cx, cy: 目标像素坐标
        point_3d: 三维坐标 dict
        use_pid: 是否使用 PID
        center_x, center_y: 图像中心坐标
    """
    if use_pid:
        output_x = int(pid_x.update(cx, target=center_x))
        output_y = int(pid_y.update(cy, target=center_y))
    else:
        output_x = int(cx * 255 / IMAGE_WIDTH)
        output_y = int(cy * 255 / IMAGE_HEIGHT)

    # 映射到 0-255
    output_x = max(0, min(255, output_x))
    output_y = max(0, min(255, output_y))

    # 更新数据帧
    msg[1] = output_x
    msg[2] = output_y
    msg[3] = 127  # 右摇杆 X (未使用)
    msg[4] = 127  # 右摇杆 Y (未使用)

    ser.send(msg)


def main():
    # 图像中心点
    center_x = IMAGE_WIDTH // 2
    center_y = IMAGE_HEIGHT // 2

    # 坐标记忆
    last_coordinate = None

    # ===== 初始化 =====
    camera = OrbbecCamera(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, fps=FPS)
    detector = YOLODetector(model_path=MODEL_PATH, conf=CONF, iou=IOU)

    # 保存内参
    save_intrinsics(camera)

    # 串口初始化
    ser = None
    if USE_SERIAL:
        ser = SerialCommunicator()
        ser.list_ports()
        if not ser.open(port=SERIAL_PORT):
            print("串口打开失败，继续运行 (无串口模式)")
            ser = None

    print("\n按 q 退出")
    print("=" * 40)

    try:
        while True:
            # 获取帧
            color_image, depth_image, depth_frame_data = camera.get_frames()
            if color_image is None:
                continue

            # 同步 YOLO 推理
            detections, annotated = detector.detect(color_image)

            # 抗灯光过滤
            detections = filter_detections(
                color_image, detections, min_variance=MIN_VARIANCE
            )

            # 处理检测结果
            point_3d = None
            for det in detections:
                cx, cy = det['center']
                cls_name = det['class_name']
                conf = det['confidence']

                # 获取三维坐标
                point_3d = camera.get_3d_point(cx, cy, depth_frame_data)

                # 坐标记忆: 深度为 0 时用上一帧坐标
                if point_3d is not None:
                    last_coordinate = point_3d
                elif last_coordinate is not None:
                    point_3d = last_coordinate

                if point_3d:
                    # 在图像上标注信息
                    info = (
                        f"{cls_name} {conf:.0%} "
                        f"| {point_3d['distance']:.2f}m"
                    )
                    cv2.putText(annotated, info, (cx - 60, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

                    # 控制台输出
                    print(
                        f"检测到 {cls_name}: "
                        f"像素({cx},{cy}) "
                        f"距离{point_3d['distance']:.3f}m "
                        f"xyz=({point_3d['x']:.3f},"
                        f"{point_3d['y']:.3f},"
                        f"{point_3d['z']:.3f})  ",
                        end="\r", flush=True
                    )

                    # 串口发送
                    if ser:
                        update_serial_msg(
                            ser, cx, cy, point_3d,
                            USE_PID, center_x, center_y
                        )
                else:
                    # 检测到目标但深度无效 - 打印调试信息
                    if depth_frame_data is not None:
                        r = 5
                        h, w = depth_frame_data.shape
                        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
                        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
                        region = depth_frame_data[y1:y2, x1:x2]
                        nz = region[region > 0]
                        print(
                            f"检测到 {cls_name} @({cx},{cy}) "
                            f"采样区域非零像素: {len(nz)}/{region.size}",
                            end="\r", flush=True
                        )

                # 只处理第一个检测到的目标
                break

            # 无检测目标时发送中立值
            if not detections and ser:
                msg[1], msg[2], msg[3], msg[4] = 127, 127, 127, 127
                ser.send(msg)

            # 显示图像
            # 画面中心显示深度参考值
            if depth_frame_data is not None:
                ch, cw = depth_frame_data.shape
                cv = depth_frame_data[ch//2, cw//2] * camera.depth_scale / 1000.0
                cv2.putText(annotated, f"center depth: {cv:.2f}m",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2)
            cv2.imshow("Orbbec + YOLO", annotated)

            # 显示深度图
            if SHOW_DEPTH:
                depth_colormap = camera.get_depth_colormap(depth_image)
                cv2.imshow("Depth", depth_colormap)

            # 按 q 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断")

    finally:
        # 发送停止指令
        if ser:
            msg[1], msg[2], msg[3], msg[4] = 127, 127, 127, 127
            ser.send(msg)
            ser.close()

        camera.stop()
        cv2.destroyAllWindows()
        print("程序退出")


if __name__ == "__main__":
    main()
