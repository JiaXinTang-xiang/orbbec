# DM02 六轴机械臂 + Orbbec 视觉抓取方案讨论

日期: 2026/06/18

## 架构

```
Orbbec Astra Mini S 深度相机 (眼在手上)
        ↓
orbec_yolo_project (Python, 上位机)
        ↓ USB CDC 串口
DM02 (STM32H723) 六轴机械臂
```

## 相机数据流

```
1. 拍照 → YOLO 检测目标 → 像素坐标 (cx, cy)
2. get_3d_point(cx, cy, depth) → P_camera (x, y, z) 米, 相机坐标系
        区域采样 11x11 中位数, 避免 D2C 空洞
        使用深度内参 (关掉 D2C 对齐)
3. P_camera(m) × 1000 → P_camera(mm)
4. 通过 USB CDC 发给 DM32
```

## 坐标系转换

```
P_camera (相机坐标系, 倾斜的)
    ↓ T_hand_eye = R × P + t  (相机→末端, 固定, 写死在 DM32 固件里)
P_ee (末端坐标系)
    ↓ T_D[4][4]  (末端→基座, DM32 实时正运动学)
P_base (基座坐标系) → arm_planning_start()
```

## 抓取姿态

- 物体: 竖直放置
- 夹爪: 水平朝前, 正面抓取
- 姿态: 固定四元数, 不需要视觉推算
- 视觉只输出 3D 位置, 不管姿态

## 通信协议 (上位机 → DM32)

```
帧头   命令      X(mm)   Y(mm)   Z(mm)   qw     qx     qy     qz   Gripper  CRC
0xAA   0x01    fp32    fp32    fp32   fp32   fp32   fp32   fp32   uint8   uint8
= 34 bytes
```

## 手眼标定 T_hand_eye

- 包含: 相机相对末端法兰的平移(XYZ偏移) + 旋转(相机倾斜角度)
- 修正相机倾斜安装导致的坐标偏差
- 初期用粗略测量值, 后期可精确标定 (棋盘格 + 多姿态)

## 深度相机盲区

- Astra Mini S 有效距离 >30cm
- 机械臂初始停在目标上方 >40cm 拍照
- 靠近抓取时进入盲区不需要视觉 (机械臂自己走规划路径)

## 相机内参

- 内参 (fx,fy,cx,cy) 自动读取, 不需要手动调整
- depth_scale 通过 getValueScale() 自动获取
- 内参 = 相机光学特性, 和安装方式无关

## 需要改动

### 上位机 (orbec_yolo_project)

- 新增 `dm02_serial.py`: USB CDC 串口通信, 封装协议帧, 发送 P_camera(mm)
- main.py 改动: 检测到目标 → 坐标转 mm → 调用 dm02_serial.send()

### DM32 固件

- 新增 `vision_protocol.cpp`: USB CDC 接收解析
- 收到 P_camera → T_hand_eye 转 P_ee → T_D 转 P_base → arm_planning_start()
- 固定抓取姿态四元数写死在固件里

## 精度分析

| 环节 | 精度 | 说明 |
|------|------|------|
| 深度测量 | ±3~10mm | Astra Mini S 0.3~2m |
| 内参转换 | ±1~2mm | 出厂标定 |
| 手眼变换 | ±?mm | 粗略测量cm级, 精确标定可达±2mm |
| 区域采样 | ±5mm | 11x11 中位数, 稳定 |

## 工作流程

```
1. 机械臂停在物体上方 ~40cm
2. 停稳后拍照 (避免运动模糊)
3. YOLO 检测 → 3D 定位 → 发 DM32
4. DM32 规划路径 → 移动到目标 → 夹爪闭合
5. 抓取成功 (进入盲区, 不依赖视觉)
```
