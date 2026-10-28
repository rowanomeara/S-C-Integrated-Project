"""Display live camera feed and test ArUco marker detection via rosbridge."""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rosbridge_api import RosbridgeRobot
from src.perception.aruco_detector import ArucoDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Camera & ArUco Detection Viewer")
    parser.add_argument("host", nargs="?", default="192.168.1.206", help="Robot IP address (e.g. 192.168.1.209)")
    parser.add_argument("--port", type=int, default=9090, help="rosbridge port (default: 9090)")
    args = parser.parse_args()

    print(f"\n[CAMERA] Connecting to TurtleBot at {args.host}:{args.port}...")
    aruco = ArucoDetector()

    with RosbridgeRobot(args.host, port=args.port) as robot:
        print("[CAMERA] Stream active. Press 'q' or Esc in the window to quit.\n")
        try:
            while robot.is_connected:
                frame_bgr = robot.query_image(timeout=1.0, wait_for_new=True)
                if frame_bgr is None:
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                r, phi, annotated = aruco.detect(frame_rgb, annotate=True)

                if r is not None:
                    print(f"\r  🎯 ArUco Detected: Range = {r:.2f} m | Bearing = {phi:+.2f} rad ({phi*57.2958:+.1f}°)    ", end="", flush=True)

                cv2.imshow("TurtleBot3 - Camera & ArUco Feed", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()
            print("\n[CAMERA] Closed.")


if __name__ == "__main__":
    main()
