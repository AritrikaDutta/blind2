# overlay_utils_1.py

import cv2
from zone_utils_1 import draw_zones_on_image

def overlay_detections_and_zones(frame, class_names, zones, object_info, safety_result):
    height, width = frame.shape[:2]

    # Draw predefined zones (rectangles)
    frame = draw_zones_on_image(frame, zones)

    for obj in object_info:
        # Clamp and validate bounding box
        x1, y1, x2, y2 = map(int, obj["bbox"])
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        w = x2 - x1
        h = y2 - y1

        # Skip invalid or overly large boxes
        if w <= 0 or h <= 0 or w > 0.8 * width or h > 0.8 * height:
            print(f"⚠️  Skipping suspicious box in overlay: {w}x{h} at [{x1}, {y1}, {x2}, {y2}]")
            continue

        label_parts = [f"ID {obj['id']}", obj['cls'], obj['direction'].upper()]
        color = (0, 255, 0)

        # Mark if in crossing zone
        if "crossing_zone" in obj["zones"]:
            label_parts.append("CROSSING")
            color = (0, 255, 255)

        # Mark if stationary
        if obj.get("stationary", False):
            label_parts.append("STILL")
            color = (192, 192, 192)

        label = " ".join(label_parts)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display safety result
    safety_text = "SAFE TO CROSS" if safety_result["safe"] else "UNSAFE TO CROSS"
    reason_text = safety_result["reason"]
    safety_color = (0, 255, 0) if safety_result["safe"] else (0, 0, 255)

    cv2.putText(frame, safety_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, safety_color, 3)
    cv2.putText(frame, reason_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, safety_color, 2)

    return frame
