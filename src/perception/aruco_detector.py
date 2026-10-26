import cv2
import numpy as np


class ArucoDetector:
    def __init__(self, focal_length=415.69, marker_real_size=0.1568):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # Compatibility across OpenCV versions: OpenCV 4.7+ uses ArucoDetector, <= 4.6 uses detectMarkers
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        else:
            self.detector = None
            if hasattr(cv2.aruco, "DetectorParameters_create"):
                self.params = cv2.aruco.DetectorParameters_create()
            else:
                self.params = cv2.aruco.DetectorParameters()
        # Defaults calibrated to 640x480 camera and standard 15.68cm marker
        self.focal_length = focal_length
        self.marker_real_size = marker_real_size

    def detect(self, img_rgb, annotate=False, is_docked=False):
        # range + bearing -> marker in robot frame
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if annotate else None

        if ids is None or len(ids) == 0:
            if annotate:
                cv2.drawMarker(annotated, (320, 240), (0, 165, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(annotated, "STATUS: SEARCHING FOR ARUCO DOCK", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(annotated, "Mode: LiDAR Obstacle Avoidance", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            return (None, None, annotated) if annotate else (None, None)

        # --- VISUAL SERVOING FEATURE EXTRACTION ---
        # 1. Extract 4 marker corners and image plane centroid (u_center, v_center)
        c = np.array(corners[0]).reshape(-1, 2)
        u_center = c[:, 0].mean()
        pixel_width = np.linalg.norm(c[0] - c[1])

        # 2. Estimate 3D range (depth) using pinhole camera model: Z = (f * S_real) / w_pixel
        r = (self.focal_length * self.marker_real_size) / max(pixel_width, 1.0)

        # 3. Compute bearing error (phi): horizontal angular offset from principal optical axis (cx=320)
        # Visual servoing uses this error directly: phi > 0 means marker is left of center, steer left
        phi = np.arctan2(320.0 - u_center, self.focal_length)

        if annotate:
            # Draw green polygon around detected marker
            pts = c.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

            # Center of marker
            mc = (int(c[:, 0].mean()), int(c[:, 1].mean()))
            cv2.circle(annotated, mc, 4, (0, 0, 255), -1)

            # Line from camera center to marker centroid
            cv2.line(annotated, (320, 240), mc, (0, 255, 255), 1)
            cv2.drawMarker(annotated, (320, 240), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)

            if is_docked:
                cv2.putText(annotated, "[DOCKED - TARGET REACHED]", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                marker_id = int(np.ravel(ids)[0])
                cv2.putText(annotated, f"LOCKED: ArUco ID {marker_id}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(annotated, f"Range: {r:.2f} m | Bearing: {np.rad2deg(phi):+.1f} deg", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv2.putText(annotated, "Mode: Visual Servoing to Dock", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        return (float(r), float(phi), annotated) if annotate else (float(r), float(phi))
