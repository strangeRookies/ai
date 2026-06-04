def draw_overlay(frame, boxes, prediction, frame_idx):
    import cv2

    from ai.visualization.action_overlay import bbox_visual_state, format_bbox_label

    output = frame.copy()
    frame_h, frame_w = output.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
        state = bbox_visual_state(box, box.get("action_threshold", 0.3))
        color = bbox_color(state)
        thickness = bbox_thickness(frame_w, state)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        draw_filled_label(output, x1, y1, format_bbox_label(box, box.get("action_threshold", 0.3)), color)
        draw_keypoints(output, box.get("keypoints"), frame_w=frame_w)

    draw_frame_label(output, frame_idx)
    return output


def bbox_color(state):
    if state == "alert":
        return (0, 0, 255)
    if state == "warning":
        return (0, 165, 255)
    return (80, 220, 80)


def bbox_thickness(frame_width, state):
    base = max(1, int(round(float(frame_width) / 420.0)))
    if state == "alert":
        return max(base + 3, 5)
    if state == "warning":
        return max(base + 2, 4)
    return max(base, 2)


def label_scale(frame_width):
    return max(0.52, min(1.05, float(frame_width) / 1280.0))


def draw_filled_label(output, x, y, text, color):
    import cv2

    frame_h, frame_w = output.shape[:2]
    scale = label_scale(frame_w)
    thickness = max(1, int(round(scale * 2)))
    padding = max(4, int(round(frame_w / 260.0)))
    text_w, text_h = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
    label_w = text_w + padding * 2
    label_h = text_h + padding * 2
    x_pos = max(0, min(int(x), frame_w - label_w - 1))
    y_top = int(y) - label_h - 4
    if y_top < 0:
        y_top = min(frame_h - label_h - 1, int(y) + 4)
    y_top = max(0, y_top)
    cv2.rectangle(output, (x_pos, y_top), (x_pos + label_w, y_top + label_h), color, -1)
    cv2.rectangle(output, (x_pos, y_top), (x_pos + label_w, y_top + label_h), (10, 10, 10), 1)
    cv2.putText(
        output,
        text,
        (x_pos + padding, y_top + padding + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_frame_label(output, frame_idx):
    import cv2

    cv2.putText(output, f"frame={frame_idx}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 1, cv2.LINE_AA)


def draw_keypoints(output, keypoints, min_confidence=0.25, frame_w=None):
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
    width = frame_w or output.shape[1]
    point_radius = max(1, int(round(float(width) / 900.0)))
    line_thickness = max(1, int(round(float(width) / 1100.0)))
    valid = []
    for point in keypoints:
        if point is None or float(point.get("confidence", 0.0)) < min_confidence:
            valid.append(None)
            continue
        center = (int(point["x"]), int(point["y"]))
        valid.append(center)
        cv2.circle(output, center, point_radius, (255, 191, 0), -1)
    for start, end in skeleton:
        if start < len(valid) and end < len(valid) and valid[start] and valid[end]:
            cv2.line(output, valid[start], valid[end], (255, 191, 0), line_thickness)
