import cv2
import time
import numpy as np
import threading
from TauLidarCommon.frame import FrameType
from camera_handler import LiDARCamera
from object_detector import ObjectDetector
from visualizer import draw_results

import asyncio
import json
import websockets
from http.server import SimpleHTTPRequestHandler
import socketserver
import os
from pathlib import Path
from threading import Thread
import base64
from datetime import datetime
import csv
from collections import deque
import atexit
from custom_enum import RecordingMode


cv2.setUseOptimized(True)
cv2.setNumThreads(4)

# Global variables for shared resources
frame_lock = threading.Lock()
latest_frame = None
latest_depth = None
latest_points_3d = None
detector = None
out = None
recording_active = False  # Flag to indicate if the system is currently recording footage
recording_start_time = None
recording_mode = None
enable_recording = False  # Flag to turn recording on/off (corresponds to "Stop/Start" toggle on UI)
TRIGGERED_RECORDING_TIME = 5  # seconds

# WebSocket and HTTP server ports
HTTP_PORT = 8080
WS_PORT = 5678

# Buffer to store object detection logs
object_log_buffer = deque(maxlen=100)

# Function to handle WebSocket connections
async def websocket_handler(websocket):
    global out, recording_active, recording_start_time, recording_mode, enable_recording
    async for message in websocket:
        data = json.loads(message)  # Parse incoming message

        if data["cmd"] == "read":
            with frame_lock:  # Ensure thread-safe access to shared resources
                if latest_frame is None or latest_depth is None or latest_points_3d is None:
                    continue  # Skip if no data is available
                mat_amplitude = latest_frame.copy()
                mat_distance = latest_depth.copy()
                points_3d = latest_points_3d.copy()

            # normalize and process amplitude and distance frames
            mat_distance = np.nan_to_num(mat_distance, nan=0.0)
            mat_amplitude = cv2.normalize(mat_amplitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            colored_image = cv2.applyColorMap(mat_amplitude, cv2.COLORMAP_JET)  # Apply color map
            results = detector.detect_objects(colored_image)  # Classify object with YOLOv8s

            # Handle NaN values in 3D points
            for point in points_3d:
                if np.isnan(point).any():
                    point = np.nan_to_num(point)

            # Draw object recognition results on the image
            output_image, object_info_list = draw_results(colored_image, results, points_3d, 160, 60, detector.model.names)
            print("OBJECT INFO LIST:", object_info_list)
            resized_image = cv2.resize(output_image, (1600, 600), interpolation=cv2.INTER_CUBIC)
            _, buffer = cv2.imencode('.jpg', resized_image)  # Encode image as JPEG
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            # Save object detection data to CSV
            if object_info_list:
                for obj in object_info_list:
                    object_log_buffer.append([
                        obj["timestamp"],
                        obj["label"],
                        obj["confidence"],
                        *obj["position"],
                        obj["angle"]
                    ])

                # Only save 100 latest
                with open("detections.csv", "w", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Time", "Label", "Confidence", "X", "Y", "Z", "Angle"])  # label head
                    writer.writerows(object_log_buffer)


            # Send image and object info to client
            try:
                latest_objects = object_info_list[:5] if object_info_list else []  # Limit to 5 objects
                payload = json.dumps({
                    'depth': image_base64,
                    'objects': latest_objects
                }, default=str)
                await websocket.send(payload)
            except Exception as e:
                print(e)

            # Recording logic
            # Triggered recording based on object detection
            if recording_mode == RecordingMode.TRIGGER and enable_recording:
                person_detected = any(obj['label'] == "person" for obj in object_info_list)  # Check if the desired object was detected
                if person_detected:
                    if not recording_active:  # System is not already recording, start a new recording
                        recording_active = True
                        current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                        filename = f'recordings/Triggered-Recording-person--{current_time}.mp4'
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(filename, fourcc, 1.0, (1600, 600))
                    recording_start_time = time.time()  # Set the recording start time to the current time
                    out.write(resized_image)
                # If an object was not detected, check if we are currently recording
                elif recording_start_time is not None:
                    if time.time() - recording_start_time < TRIGGERED_RECORDING_TIME:  # Still within the recording time grace period, continue recording
                        out.write(resized_image)
                    else:  # Greater than TRIGGERED_RECORDING_TIME seconds have passed since an object was last detected, stop recording
                        recording_active = False
                        recording_start_time = None
                        out.release()
            # Continuous recording
            elif recording_mode == RecordingMode.CONTINUOUS and enable_recording:
                if not recording_active:
                    recording_active = True
                    current_time = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                    filename = f'recordings/Continuous-Recording--{current_time}.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(filename, fourcc, 1.0, (1600, 600))
                out.write(resized_image)

        elif data["cmd"] == "start":
            enable_recording = True
        elif data["cmd"] == "stop":
            enable_recording = False
            if recording_active:
                recording_active = False
                out.release()
        elif data["cmd"] == RecordingMode.CONTINUOUS:
            recording_mode = RecordingMode.CONTINUOUS
        elif data["cmd"] == RecordingMode.TRIGGER:
            recording_mode = RecordingMode.TRIGGER
        elif data["cmd"] == "toggle_recording":
            if recording_active:
                recording_active = False
                out.release()
        else:
            print(data["cmd"])

async def start_websocket_server():
    async with websockets.serve(websocket_handler, "0.0.0.0", WS_PORT):
        await asyncio.Future()

def lidar_thread(camera):
    global latest_frame, latest_depth, latest_points_3d
    while True:
        try:
            frame = camera.readFrame(FrameType.DISTANCE_AMPLITUDE)
            if frame is None or len(frame.data_amplitude) == 0:
                print("Warning: Empty frame received for DISTANCE_AMPLITUDE, skipping...")
                time.sleep(0.01)
                continue

            mat_amplitude = np.frombuffer(frame.data_amplitude, dtype=np.float32, count=-1).reshape(frame.height, frame.width)

            frame = camera.readFrame(FrameType.DISTANCE)
            if frame is None or len(frame.data_depth) == 0:
                print("Warning: Empty frame received for DISTANCE, skipping...")
                time.sleep(0.01)
                continue

            mat_distance = np.frombuffer(frame.data_depth, dtype=np.float32, count=-1).reshape(frame.height, frame.width)
            mat_distance = np.nan_to_num(mat_distance, nan=0.0)

            with frame_lock:
                latest_frame = mat_amplitude.copy()
                latest_depth = mat_distance.copy()
                latest_points_3d = frame.points_3d.copy()

            time.sleep(0.01)

        except Exception as e:
            print(f"Error in LiDAR thread: {e}")

def main():
    global detector, out, recording_active
    try:
        lidar_camera = LiDARCamera()
        atexit.register(lidar_camera.release)  # Ensure camera is released on exit
    except Exception as e:
        print(f"Error in LiDAR camera initialization: {e}")
        return
    detector = ObjectDetector()

    threading.Thread(target=lidar_thread, args=(lidar_camera.camera,), daemon=True).start()
    ws_thread = Thread(target=lambda: asyncio.run(start_websocket_server()), daemon=True)
    ws_thread.start()

    web_dir = Path(__file__).absolute().parent
    os.chdir(web_dir)

    class CustomHandler(SimpleHTTPRequestHandler):
        extensions_map = {
            ".html": "text/html",
            ".js": "application/javascript",
            ".css": "text/css",
            **SimpleHTTPRequestHandler.extensions_map,
        }

    with socketserver.TCPServer(("", HTTP_PORT), CustomHandler) as httpd:
        print(f"HTTP Server running at http://192.168.137.206:{HTTP_PORT} or use 127.0.0.1")
        print(f"WebSocket Server running at ws://127.0.0.1:{WS_PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("Shutting down servers...")
            lidar_camera.release()
            if recording_active:
                recording_active = False
                out.release()

if __name__ == "__main__":
    main()
