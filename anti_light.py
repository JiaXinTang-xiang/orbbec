"""
抗灯光干扰模块
通过灰度方差区分真实目标和灯光误检
"""

import cv2
import numpy as np


def calculate_gray_variance(image, bbox):
    """计算检测框内物体的灰度方差

    参数:
        image: BGR 图像
        bbox: [x1, y1, x2, y2]

    返回:
        (variance, mean_gray)
    """
    x1, y1, x2, y2 = map(int, bbox)
    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return 0, 0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
    return np.var(gray), np.mean(gray)


def filter_detections(image, detections, min_variance=100):
    """过滤灯光误检，保留方差最大的真实目标

    灯光区域灰度均匀（方差小），真实目标纹理丰富（方差大）

    参数:
        image: BGR 图像
        detections: YOLO 检测结果列表
        min_variance: 最小方差阈值，低于此值视为灯光

    返回:
        过滤后的检测结果列表
    """
    if not detections:
        return []

    filtered = []
    for det in detections:
        variance, mean_gray = calculate_gray_variance(image, det['bbox'])
        if variance >= min_variance:
            det['variance'] = variance
            det['mean_gray'] = mean_gray
            filtered.append(det)

    # 按方差降序，方差最大的最可能是真目标
    filtered.sort(key=lambda d: d.get('variance', 0), reverse=True)
    return filtered
