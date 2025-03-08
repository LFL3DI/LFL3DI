import cv2
import time
import numpy as np
import threading
from TauLidarCommon.frame import FrameType
from camera_handler import LiDARCamera
from object_detector import ObjectDetector
from visualizer import draw_results

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

frame_lock = threading.Lock()
latest_frame = None
latest_depth = None
latest_points_3d = None

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
    lidar_camera = LiDARCamera()
    detector = ObjectDetector()

    threading.Thread(target=lidar_thread, args=(lidar_camera.camera,), daemon=True).start()

    while True:
        with frame_lock:
            if latest_frame is None or latest_depth is None or latest_points_3d is None:
                continue
            mat_amplitude = latest_frame.copy()
            mat_distance = latest_depth.copy()
            points_3d = latest_points_3d.copy()

        mat_distance = np.nan_to_num(mat_distance, nan=0.0)

        mat_amplitude = cv2.normalize(mat_amplitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        colored_image = cv2.applyColorMap(mat_amplitude, cv2.COLORMAP_JET)

        results = detector.detect_objects(colored_image)

        for point in points_3d:
            if np.isnan(point).any():
                print("Warning: Detected NaN in points_3d, replacing with zeros.")
                point = np.nan_to_num(point)

        output_image = draw_results(colored_image, results, points_3d, 160, 60, detector.model.names)

        resized_image = cv2.resize(output_image, (1200, 1000), interpolation=cv2.INTER_CUBIC)

        cv2.imshow("YOLO + LiDAR Detection", resized_image)
        cv2.waitKey(1)
        time.sleep(0.01)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
