"""
OpenVINO 直接推理检测器 (绕过 ultralytics 开销)
"""
import cv2
import numpy as np
import time


class OpenVINODetector:
    def __init__(self, model_path="model/best1_openvino_model/",
                 conf=0.5, iou=0.7, imgsz=640):
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.class_names = {}

        from openvino import Core
        import os

        # 找到 .xml 文件
        if os.path.isdir(model_path):
            xml_files = [f for f in os.listdir(model_path) if f.endswith('.xml')]
            if not xml_files:
                raise FileNotFoundError(f"No .xml found in {model_path}")
            xml_path = os.path.join(model_path, xml_files[0])
        else:
            xml_path = model_path

        print(f"OpenVINO 加载: {xml_path}")
        core = Core()
        model = core.read_model(xml_path)
        self.compiled_model = core.compile_model(model, "CPU",
                                                  {"PERFORMANCE_HINT": "LATENCY"})

        # 获取输入输出
        self.input_key = self.compiled_model.input(0)
        self.output_key = self.compiled_model.output(0)
        self.input_shape = self.input_key.shape
        self._h_in, self._w_in = self.input_shape[2], self.input_shape[3]
        print(f"  输入: {self.input_shape}, 设备: CPU")

        # 从 metadata.yaml 读类别名
        meta_path = os.path.join(os.path.dirname(xml_path), "metadata.yaml")
        if os.path.exists(meta_path):
            import yaml
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
            names = meta.get("names", {})
            self.class_names = {int(k): v for k, v in names.items()}

        print(f"  类别: {self.class_names}")

    def preprocess(self, frame):
        """预处理: resize + normalize + CHW + batch"""
        img = cv2.resize(frame, (self._w_in, self._h_in))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # add batch
        return img

    def postprocess(self, output, frame_shape):
        """解析输出为 detections 列表"""
        h_frame, w_frame = frame_shape[:2]
        detections = []

        for det in output[0]:
            conf = float(det[4])
            if conf < self.conf:
                continue

            class_id = int(det[5])
            x1, y1, x2, y2 = det[:4]

            # 缩放回原图坐标
            x1 = int(x1 * w_frame / self._w_in)
            y1 = int(y1 * h_frame / self._h_in)
            x2 = int(x2 * w_frame / self._w_in)
            y2 = int(y2 * h_frame / self._h_in)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = self.class_names.get(class_id, str(class_id))

            detections.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'confidence': conf,
                'class_id': class_id,
                'class_name': class_name,
            })

        return detections

    def draw(self, frame, detections):
        """绘制检测框"""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.0%}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.circle(annotated, det['center'], 4, (0, 0, 255), -1)
        return annotated

    def detect(self, frame):
        """检测单帧
        Returns:
            detections: 同 YOLODetector 格式
            annotated_frame: 标注后的图像
        """
        input_data = self.preprocess(frame)
        result = self.compiled_model([input_data])[self.output_key]
        detections = self.postprocess(result, frame.shape)
        annotated = self.draw(frame, detections)
        return detections, annotated
