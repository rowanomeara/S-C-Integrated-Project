import os
import sys
import time
import numpy as np
from src.hardware.robot_real import TurtleBotReal
from src.perception.aruco_detector import ArucoDetector
from src.perception.camera_viewer import CameraWindow
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap

def main():
    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    print(f"=== Launching TurtleBot3 Real Hardware Mission (Camera ID: {cam_id}) ===")
    robot = TurtleBotReal(camera_id=cam_id, dt=0.05)
    aruco = ArucoDetector()
    lidar = LidarProcessor(danger_distance=0.38)
    ekf = RobotEKF()

    # Initial estimated pose (x=1.0m, y=0.0m, facing west theta=pi)
    # Adjust this to match where you place the robot in the lab arena
    ekf.x = np.array([1.0, 0.0, np.pi], dtype=float)
    ekf.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])

    target_dock = np.array([-1.52, 0.0])
    marker_pos = np.array([-1.69, 0.0])

    # Show live camera window if a display is available
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name == "darwin")
    cam_window = CameraWindow() if has_display else None

    docked = False
    dodge_dir = 0.0
    dodge_timer = 0
    step = 0

    print("[READY] Starting real-time control loop (20 Hz). Press Ctrl+C to stop.")

    try:
        while not docked:
            step += 1
            t_start = time.time()

            # 1. EKF prediction from real wheel odometry
            v_odom, w_odom = robot.get_odometry()
            ekf.predict(v_odom, w_odom, dt=robot.dt)

            # 2. Camera POV & ArUco detection
            frame = robot.get_camera_frame()
            r, phi = None, None
            if frame is not None:
                r, phi, annotated = aruco.detect(frame, annotate=True, is_docked=docked)
                if cam_window is not None:
                    cam_window.show(annotated)
                if r is not None:
                    ekf.update_aruco(r, phi, dock_pos=marker_pos)

            # 3. Docking terminal condition
            dist_to_dock = np.hypot(target_dock[0] - ekf.x[0], target_dock[1] - ekf.x[1])
            if dist_to_dock < 0.10 or (r is not None and r < 0.20):
                print(f"\n[DOCK SUCCESS] Target reached! Stopping robot.")
                docked = True
                robot.set_velocity(0.0, 0.0)
                break

            # 4. LiDAR obstacle processing
            ranges = robot.get_lidar_scan()
            d_front = float(np.min(ranges[150:210]))
            d_left = float(np.mean(ranges[195:255]))
            d_right = float(np.mean(ranges[105:165]))

            angle_to_dock = np.arctan2(target_dock[1] - ekf.x[1], target_dock[0] - ekf.x[0])
            heading_err = wrap(angle_to_dock - ekf.x[2])

            # 5. Control laws (Visual Servoing vs Obstacle Avoidance)
            if r is not None and dist_to_dock < 1.0:
                # Visual servoing: directly align camera optical axis with marker
                cmd_w = float(np.clip(2.5 * phi, -0.6, 0.6))
                cmd_v = float(np.clip(0.12 * np.cos(phi), 0.03, 0.12))
            elif d_front < 0.38 and dist_to_dock > 0.4:
                # Obstacle directly ahead -> turn toward clearer side
                if dodge_timer <= 0:
                    dodge_dir = 1.0 if d_left >= d_right else -1.0
                    dodge_timer = 20
                cmd_v = 0.03
                cmd_w = dodge_dir * 0.7
            elif dodge_timer > 0:
                dodge_timer -= 1
                cmd_v = 0.12
                cmd_w = dodge_dir * 0.3
            elif abs(heading_err) > np.deg2rad(45):
                cmd_v = 0.0
                cmd_w = float(np.clip(2.0 * heading_err, -0.7, 0.7))
            else:
                cmd_v = float(np.clip(0.14 * np.cos(heading_err), 0.05, 0.14))
                cmd_w = float(np.clip(1.5 * heading_err, -0.5, 0.5))

            # Send actuation
            robot.set_velocity(cmd_v, cmd_w)

            # Telemetry logging every 20 steps (1 sec)
            if step % 20 == 0:
                status = f"LOCKED r={r:.2f}m" if r is not None else f"SEEKING (front={d_front:.2f}m)"
                print(f"[{step:04d}] Est: [{ekf.x[0]:.2f}, {ekf.x[1]:.2f}, {np.rad2deg(ekf.x[2]):.0f}°] | Dist: {dist_to_dock:.2f}m | {status}")

            # Maintain 20 Hz loop rate
            elapsed = time.time() - t_start
            sleep_time = max(0.0, robot.dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[STOP] User interrupted mission.")
    finally:
        # Crucial for safety: ensure physical motors are stopped
        robot.stop()
        if cam_window is not None:
            cam_window.close()
        print("[SHUTDOWN] Real robot safely stopped.")

if __name__ == "__main__":
    main()
