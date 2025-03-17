import numpy as np
from TauLidarCamera.camera import Camera
from TauLidarCommon.frame import FrameType

class LiDARCamera:
    def __init__(self):
        self.camera = Camera.open()
        if self.camera is None:
            print("Error: Could not connect to LiDAR Camera!")
            exit(1)

        self.camera.setModulationChannel(0)
        self.camera.setIntegrationTime3d(0, 800)
        self.camera.setMinimalAmplitude(0, 60)
        self.camera.setRange(0, 4500)
        self.camera.setIntegrationTimeGrayscale(15000)

    def get_depth_frame(self):
        """ ��ȡ LiDAR ������ݣ�X, Y, Z��"""
        frame = self.camera.readFrame(FrameType.DISTANCE)

        if frame is None:
            print("Error: No frame received from LiDAR Camera!")
            return None, None, None, None

        mat_depth = np.frombuffer(frame.data_depth, dtype=np.float32, count=-1).reshape(frame.height, frame.width)

        # ? ���� NaN ����
        if np.isnan(mat_depth).any():
            print("Warning: depth_frame contains NaN values! Fixing...")
            mat_depth = np.nan_to_num(mat_depth, nan=0.0)

        points_3d = frame.points_3d  # 3D ��������
        return mat_depth, points_3d, frame.width, frame.height

    def release(self):
        """ Release the LiDAR camera resources """
        if self.camera:
            self.camera.close()
            self.camera = None
