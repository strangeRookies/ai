def draw_overlay(frame, boxes, prediction, frame_idx):
    import cv2

    output = frame.copy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        color = (0, 0, 255) if box.get("event_triggered") else (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        track_text = f" id={box['track_id']}" if box.get("track_id") is not None else ""
        draw_box_label(output, x1, y1, f"person{track_text} {box['score']:.2f}", color)
        if box.get("action_overlay"):
            draw_box_label(output, x1, y1 + 20, str(box["action_overlay"]), color)
        draw_keypoints(output, box.get("keypoints"))

    if prediction:
        label = prediction["label"]
        score = prediction["score"]
        text = f"[OK] {label} detected score={score:.2f} frame={frame_idx}"
        cv2.putText(output, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(output, f"frame={frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return output


def draw_box_label(output, x, y, text, color):
    import cv2

    y_pos = max(int(y) - 8, 15)
    cv2.putText(output, text, (int(x), y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_keypoints(output, keypoints, min_confidence=0.25):
    if not keypoints:
        return
    import cv2

    skeleton = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]
    valid = []
    for point in keypoints:
        if point is None or float(point.get("confidence", 0.0)) < min_confidence:
            valid.append(None)
            continue
        center = (int(point["x"]), int(point["y"]))
        valid.append(center)
        cv2.circle(output, center, 2, (255, 191, 0), -1)
    for start, end in skeleton:
        if start < len(valid) and end < len(valid) and valid[start] and valid[end]:
            cv2.line(output, valid[start], valid[end], (255, 191, 0), 1)
