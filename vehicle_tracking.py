import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import os

VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

def run_vehicle_tracking(input_path, output_path="outputs/output.mp4"):
    model = YOLO("yolo11n.pt")

    LINE_Y = 600
    class_counts = {name: 0 for name in VEHICLE_CLASSES.values()}
    total_count = 0
    counted_ids = set()
    last_positions = {}

    results = model.track(
        source=input_path,
        tracker="bytetrack",
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.5,
        verbose=False,
        stream=True,
        show=False
    )

    writer = None

    for result in results:
        frame = result.orig_img
        if frame is None:
            break

        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, 30, (w, h))

        start = datetime.datetime.now()
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), (0,255,255), 2)

        if result.boxes.id is None:
            continue

        ids = result.boxes.id.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()

        for tid, cls_id, box in zip(ids, classes, bboxes):
            cls_id = int(cls_id)
            track_id = int(tid)
            if cls_id not in VEHICLE_CLASSES:
                continue

            label = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box)
            center_y = (y1 + y2) // 2
            prev_y = last_positions.get(track_id, center_y)
            last_positions[track_id] = center_y

            if prev_y < LINE_Y <= center_y and track_id not in counted_ids:
                counted_ids.add(track_id)
                class_counts[label] += 1
                total_count += 1

            color = tuple(int(c) for c in np.random.RandomState(42 + cls_id).randint(0,255,3))
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{track_id}-{label}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        ypos = 30
        for v, cnt in class_counts.items():
            cv2.putText(frame, f"{v.capitalize()}: {cnt}", (10, ypos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            ypos += 30
        cv2.putText(frame, f"Total: {total_count}", (10, ypos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        elapsed = (datetime.datetime.now() - start).total_seconds()
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1]-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        writer.write(frame)

    if writer:
        writer.release()
    return output_path, class_counts
