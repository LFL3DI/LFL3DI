import cv2
import numpy as np
import math

def compute_angle(x, z):
    return math.degrees(math.atan2(x, z))

def draw_overlay(image, text, x, y, font_scale, color, alpha=0.5):
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    overlay = image.copy()
    cv2.rectangle(overlay, (x, y - text_h - 5), (x + text_w + 5, y + baseline + 5), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

def draw_results(image, results, points_3d, width, height, class_names):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            index = min(center_y * width + center_x, len(points_3d) - 1)
            x, y, z = points_3d[index][:3]

            angle = compute_angle(x, z)
            angle_label = f"Angle: {angle:.1f}degree"
            position_label = f"X:{x:.1f}m Y:{y:.1f}m Z:{z:.1f}m"

            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            label = f"{class_names[cls]}: {conf:.2f}"

            box_color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            font_scale = max(0.3, min((x2 - x1) / 300, 0.6))

            text_y_label = y2 - 5
            text_y_position = text_y_label - 20
            text_y_angle = text_y_position - 20 
            draw_overlay(image, label, x1, text_y_label, font_scale, box_color)
            draw_overlay(image, position_label, x1, text_y_position, font_scale, box_color)
            draw_overlay(image, angle_label, x1, text_y_angle, font_scale, box_color)

    return image, label
