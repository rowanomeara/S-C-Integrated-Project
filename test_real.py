import sys
import time
from src.hardware.robot_real import TurtleBotReal

def test_hardware():
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    print(f"=== Testing IRL TurtleBot3 Hardware Connections (Camera ID: {cam_id}) ===")
    robot = TurtleBotReal(camera_id=cam_id)

    time.sleep(1.0) # wait for first scans
    v, w = robot.get_odometry()
    scan = robot.get_lidar_scan()
    frame = robot.get_camera_frame()

    print(f"[1] Odometry: v={v:.3f} m/s, w={w:.3f} rad/s")
    print(f"[2] LiDAR:    {len(scan)} beams, min distance = {scan.min():.2f} m")
    print(f"[3] Camera:   frame received = {frame is not None}" + (f", shape = {frame.shape}" if frame is not None else ""))

    print("\n[4] Motor check: pulsing wheels at 0.05 m/s for 1 second...")
    robot.set_velocity(0.05, 0.0)
    time.sleep(1.0)
    robot.stop()
    print("[DONE] Hardware test complete. Robot stopped safely.")

if __name__ == "__main__":
    test_hardware()
