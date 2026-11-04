"""Perception: ArUco Marker Detection and Visual Servoing Feature Extraction.

This module detects DICT_4X4_50 fiducial markers on the docking bay and extracts
polar range and bearing features using a pinhole camera projection model.
"""

import cv2
import numpy as np


class ArucoDetector:
    """Detects ArUco markers and computes range and bearing to the docking station."""

    def __init__(self, focal_length=415.69, marker_real_size=0.1568):
        """Initialize the ArUco detector with calibrated camera intrinsics.

        Parameters:
          focal_length: Calibrated horizontal focal length fx in pixels (415.69 px for 640x480).
          marker_real_size: Physical side length of the square ArUco marker in meters (0.1568 m).
        """
        # Load the 4x4 dictionary (50 unique IDs) used on the docking station
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Maintain compatibility across OpenCV versions:
        # OpenCV >= 4.7 provides cv2.aruco.ArucoDetector class.
        # OpenCV <= 4.6 uses standalone cv2.aruco.detectMarkers function.
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        else:
            self.detector = None
            if hasattr(cv2.aruco, "DetectorParameters_create"):
                self.params = cv2.aruco.DetectorParameters_create()
            else:
                self.params = cv2.aruco.DetectorParameters()

        self.focal_length = focal_length
        self.marker_real_size = marker_real_size

    def detect(self, img_rgb, annotate=False, is_docked=False):
        """Extract range and bearing to the nearest visible ArUco marker.

        Parameters:
          img_rgb: np.ndarray RGB input image (480, 640, 3).
          annotate: bool flag to generate an annotated HUD display image.
          is_docked: bool flag indicating terminal docking completion status.

        Returns:
          If annotate is False: (range, bearing) as floats, or (None, None) if undetected.
          If annotate is True: (range, bearing, annotated_bgr_frame).
        """
        # Convert RGB input to single-channel greyscale for corner detection
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)

        annotated = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if annotate else None

        # Return None when no marker is detected in the current camera frame
        if ids is None or len(ids) == 0:
            if annotate:
                # Draw center crosshair and search status overlay
                cv2.drawMarker(annotated, (320, 240), (0, 165, 255), cv2.MARKER_CROSS, 20, 1)
                cv2.putText(annotated, "STATUS: SEARCHING FOR ARUCO DOCK", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                cv2.putText(annotated, "Mode: LiDAR Obstacle Avoidance", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            return (None, None, annotated) if annotate else (None, None)

        # 1. Extract the four corner points and compute horizontal centroid in pixels
        c = np.array(corners[0]).reshape(-1, 2)
        u_center = c[:, 0].mean()
        # Marker projected width on the image plane: Euclidean distance between top-left and top-right corners
        pixel_width = np.linalg.norm(c[0] - c[1])

        # 2. Pinhole camera depth estimation:
        # Distance Z = (focal_length * physical_size) / projected_width_pixels
        # Clamped to minimum 1.0 pixel to avoid zero-division errors
        r = (self.focal_length * self.marker_real_size) / max(pixel_width, 1.0)

        # 3. Horizontal bearing angle phi from optical axis (principal point cx = 320 px):
        # phi = arctan((cx - u_center) / fx)
        # Sign convention: phi > 0 means the target is to the left (requires positive yaw rate)
        phi = np.arctan2(320.0 - u_center, self.focal_length)

        if annotate:
            # Draw green bounding perimeter around the detected marker
            pts = c.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (0, 255, 0), 2)

            # Draw red circle at marker centroid
            mc = (int(c[:, 0].mean()), int(c[:, 1].mean()))
            cv2.circle(annotated, mc, 4, (0, 0, 255), -1)

            # Draw vector line connecting optical principal point (320, 240) to marker center
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
