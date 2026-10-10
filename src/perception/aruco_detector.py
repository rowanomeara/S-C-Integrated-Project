import cv2
import numpy as np


class ArucoDetector:
    def __init__(self):
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict)
        self.focal_length = 554.0   # for 60 deg FOV, 640x480
        self.marker_real_size = 0.08  # 80 mm

    def detect(self, img_rgb):
        #range + bearing -> marker in robot frame 
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return None, None

        c = corners[0][0]
        u_center = c[:, 0].mean()
        pixel_width = np.linalg.norm(c[0] - c[1])

        # Z = (f * real_size) / pixel_width
        r = (self.focal_length * self.marker_real_size) / max(pixel_width, 1.0)
        # phi = atan2(u_center - cx, fx) <- from center
        phi = np.arctan2(320.0 - u_center, self.focal_length)

        return float(r), float(phi)
