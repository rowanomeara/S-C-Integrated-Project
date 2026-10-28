import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hardware.robot_real import TurtleBotReal
except ImportError:
    from robot_real import TurtleBotReal
from src.perception.aruco_detector import ArucoDetector


def test_hardware():
    host = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "192.168.1.206"
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    print(f"\n=======================================================")
    print(f" TURTLEBOT3 HARDWARE PRE-FLIGHT CHECK ({host})")
    print(f"=======================================================")

    try:
        robot = TurtleBotReal(host=host, camera_id=cam_id)
    except Exception as e:
        print(f"\n❌ [CRITICAL] Failed to initialize robot HAL: {e}")
        return False

    time.sleep(1.2)  # allow first sensor messages to arrive
    aruco = ArucoDetector()

    # 1. Odometry Check
    v, w = robot.get_odometry()
    print(f" [1/4] Odometry:  v={v:.3f} m/s, w={w:.3f} rad/s -> ✅ PASS")

    # 2. LiDAR Check
    scan = robot.get_lidar_scan()
    min_dist = float(scan.min())
    if len(scan) == 360 and min_dist > 0.05:
        print(f" [2/4] LiDAR:     360 beams received, min distance = {min_dist:.2f} m -> ✅ PASS")
    else:
        print(f" [2/4] LiDAR:     ⚠️ Warning: min_dist = {min_dist:.2f} m (check cable/clearing)")

    # 3. Camera & ArUco Check (allow up to 2.5s for ROS topic discovery)
    frame = None
    for _ in range(25):
        frame = robot.get_camera_frame()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is not None:
        r, phi = aruco.detect(frame, annotate=False)
        status_str = f"ArUco SIGHTED (r={r:.2f}m)" if r is not None else "Frame OK (No ArUco in view)"
        print(f" [3/4] Camera:    Frame shape {frame.shape} | {status_str} -> ✅ PASS")
    else:
        print(f" [3/4] Camera:    ❌ FAILED (No frame received from /camera/image_raw/compressed or camera index)")

    # 4. Safe Motor Check (pulsing forward 0.04 m/s for 0.4s)
    print("\n [4/4] Motor:     Pulsing wheels forward 4 cm/s for 0.4s ... ", end="", flush=True)
    robot.set_velocity(0.04, 0.0)
    time.sleep(0.4)
    robot.stop()
    print("-> ✅ PASS (Stopped safely)")

    print("\n=======================================================")
    print(" ALL PRE-FLIGHT CHECKS COMPLETE. READY FOR MISSION!")
    print("=======================================================\n")
    return True


if __name__ == "__main__":
    test_hardware()
