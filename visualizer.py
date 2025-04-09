import cv2
import numpy as np
import math
from datetime import datetime

# Function to compute the angle between the x and z coordinates
def compute_angle(x, z):
    return math.degrees(math.atan2(x, z))

# Function to draw a semi-transparent overlay with text on an image
def draw_overlay(image, text, x, y, font_scale, color, alpha=0.3):
    """
    Draws a semi-transparent rectangle with text on the image.
    - image: The image to draw on.
    - text: The text to display.
    - x, y: Coordinates for the text.
    - font_scale: Scale of the font.
    - color: Background color of the rectangle.
    - alpha: Transparency level of the rectangle.
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    overlay = image.copy()  # Create a copy of the image for the overlay
    # Draw a rectangle for the text background
    cv2.rectangle(overlay, (x, y - text_h - 5), (x + text_w + 5, y + baseline + 5), color, -1)
    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    # Draw the text on the image
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

# Function to draw detection results on the image and return detected objects
def draw_results(image, results, points_3d, width, height, class_names):
    """
    Draws bounding boxes and detection information on the image.
    - image: The image to draw on.
    - results: Detection results containing bounding boxes and class information.
    - points_3d: 3D points corresponding to the image.
    - width, height: Dimensions of the image.
    - class_names: List of class names for the detected objects.
    Returns:
    - image: The image with the detection results drawn.
    - detected_objects: A list of dictionaries containing detection information.
    """
    # Upscale the image to a higher resolution
    upscale_factor = 2  # Increase this factor for sharper overlays
    high_res_width = width * upscale_factor
    high_res_height = height * upscale_factor
    image = cv2.resize(image, (high_res_width, high_res_height), interpolation=cv2.INTER_CUBIC)

    detected_objects = []  # List to store information about detected objects

    # Iterate through the detection results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() * upscale_factor).astype(int)
            center_x = int((x1 + x2) / 2)  # Compute the center x-coordinate of the box
            center_y = int((y1 + y2) / 2)  # Compute the center y-coordinate of the box
            
            # Compute the index of the corresponding 3D point
            index = min(center_y * width + center_x, len(points_3d) - 1)
            x, y, z = points_3d[index][:3]  # Extract the 3D coordinates

            # Skip if any of the coordinates are NaN
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue

            # Compute the angle and labels for the detected object
            angle = compute_angle(x, z)
            angle_label = f"Angle: {angle:.1f}degree"
            position_label = f"X:{x:.1f}m Y:{y:.1f}m Z:{z:.1f}m"
            array_index = len(detected_objects) + 1  # 1-based index

            # Extract confidence and class label
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class index
            label_name = class_names[cls]  # Class name
            label = f"{label_name}: {conf:.2f}"  # Label with class name and confidence
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp

            # Define the color for the bounding box
            box_color = (0, 255, 0)  # Green

            # Draw the bounding box around the object
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 1)

            # Adjust font scale based on the size of the bounding box
            font_scale = max(0.3, min((x2 - x1) / 300, 0.6))

            # Compute positions for the text labels
            text_y_label = y2 - 5
            text_y_position = text_y_label - 20
            text_y_angle = text_y_position - 20

            # Draw the object index on the image (commented-out labels can be re-enabled if needed)
            # draw_overlay(image, label, x1, text_y_label, font_scale, box_color)
            # draw_overlay(image, position_label, x1, text_y_position, font_scale, box_color)
            # draw_overlay(image, angle_label, x1, text_y_angle, font_scale, box_color)
            draw_overlay(image, label, x1, text_y_label, font_scale, box_color)

            # Append the detected object information to the list
            detected_objects.append({
                "index": array_index,  # Object index in array
                "label": label_name,  # Class label
                "confidence": round(conf, 2),  # Confidence score
                "position": [round(x, 2), round(y, 2), round(z, 2)],  # 3D position
                "angle": round(angle, 1),  # Angle
                "timestamp": timestamp  # Detection timestamp
            })

    # Downscale the image back to the original resolution for display
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    return image, detected_objects  # Return the annotated image and list of detected objects