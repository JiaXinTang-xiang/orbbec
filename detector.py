"""
YOLO 目标检测模块
封装模型加载和推理逻辑
"""

from ultralytics import YOLO
import torch


class YOLODetector:
    def __init__(self, model_path="best.pt", conf=0.5, iou=0.7, imgsz=640):
        """初始化 YOLO 检测器

        参数:
            model_path: 模型文件路径
            conf: 置信度阈值
            iou: IoU 阈值
            imgsz: 推理图像尺寸
        """
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

        print(f"YOLO 模型加载完成")
        print(f"  模型: {model_path}")
        print(f"  设备: {self.device}")
        print(f"  类别: {self.model.names}")

    def detect(self, frame):
        """对单帧图像进行检测

        参数:
            frame: BGR 图像

        返回:
            detections: 检测结果列表，每项为 dict:
                {
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': float,
                    'class_id': int,
                    'class_name': str
                }
            annotated_frame: 绘制了检测框的图像
        """
        results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]

                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': cls_name
                })

        # 绘制检测框
        annotated = results[0].plot()

        return detections, annotated
