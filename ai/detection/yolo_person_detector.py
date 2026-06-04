PERSON_CLASS_ID = 0


class YoloPersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.5, imgsz=640):
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(f"ultralytics is required for YOLO inference: {exc}") from exc
        self.model = YOLO(model_name)

    def detect(self, frame, frame_idx, conf=None):
        results = self.model.predict(
            frame,
            conf=self.conf if conf is None else conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )
        boxes = []
        for result in results:
            result_boxes = getattr(result, "boxes", None)
            if result_boxes is None:
                continue
            boxes.extend(filter_person_boxes(result_boxes, getattr(result, "names", {})))
        return {"frame_idx": int(frame_idx), "boxes": boxes}


class MockPersonDetector:
    def __init__(self, conf=0.9):
        self.conf = conf

    def detect(self, frame, frame_idx):
        height, width = _frame_shape(frame)
        x1, y1 = width * 0.25, height * 0.2
        x2, y2 = width * 0.65, height * 0.9
        return {
            "frame_idx": int(frame_idx),
            "boxes": [
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "score": float(self.conf),
                    "class_name": "person",
                }
            ],
        }


def filter_person_boxes(boxes, names):
    output = []
    xyxy = boxes.xyxy.detach().float().cpu().tolist() if boxes.xyxy is not None else []
    confs = boxes.conf.detach().float().cpu().tolist() if boxes.conf is not None else []
    classes = boxes.cls.detach().int().cpu().tolist() if boxes.cls is not None else []
    for idx, bbox in enumerate(xyxy):
        class_id = classes[idx] if idx < len(classes) else PERSON_CLASS_ID
        class_name = names.get(class_id, "person") if isinstance(names, dict) else "person"
        if class_id != PERSON_CLASS_ID and class_name != "person":
            continue
        output.append(
            {
                "x1": float(bbox[0]),
                "y1": float(bbox[1]),
                "x2": float(bbox[2]),
                "y2": float(bbox[3]),
                "score": float(confs[idx]) if idx < len(confs) else 0.0,
                "class_name": "person",
            }
        )
    return output


def _frame_shape(frame):
    if hasattr(frame, "shape"):
        return int(frame.shape[0]), int(frame.shape[1])
    return 480, 640
