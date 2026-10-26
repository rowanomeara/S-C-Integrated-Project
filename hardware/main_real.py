import os
import sys
import time
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hardware.robot_real import TurtleBotReal
except ImportError:
    from robot_real import TurtleBotReal
from src.perception.aruco_detector import ArucoDetector
from src.perception.camera_viewer import CameraWindow
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap


def main():
    parser = argparse.ArgumentParser(description="TurtleBot3 Real Hardware Mission Runner")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--x0", type=float, default=1.0, help="Initial X position in meters (default: 1.0)")
    parser.add_argument("--y0", type=float, default=0.0, help="Initial Y position in meters (default: 0.0)")
    parser.add_argument("--th0", type=float, default=180.0, help="Initial heading in degrees (default: 180.0)")
    parser.add_argument("--no-video", action="store_true", help="Disable automatic MP4 video recording")
    parser.add_argument("--video-out", type=str, default="real_mission_camera_pov.mp4", help="Video output path")
    args = parser.parse_args()

    print(f"=== Initializing TurtleBot3 Real Hardware Mission (Camera ID: {args.cam}) ===")
    robot = TurtleBotReal(camera_id=args.cam, dt=0.05)
    aruco = ArucoDetector()
    lidar = LidarProcessor(danger_distance=0.40)
    ekf = RobotEKF()

    # Initial estimated pose from CLI / default spawn
    th0_rad = np.deg2rad(args.th0)
    ekf.x = np.array([args.x0, args.y0, th0_rad], dtype=float)
    ekf.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])

    target_dock = np.array([-1.52, 0.0])
    marker_pos = np.array([-1.69, 0.0])

    # Show live camera window if a graphical display is connected
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or os.name == "darwin")
    cam_window = CameraWindow() if has_display else None

    # Automatic MP4 Video Recorder (records robot POV with annotations)
    video_writer = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.video_out, fourcc, 20.0, (640, 480))
        print(f"[RECORDER] POV video will be recorded to: {args.video_out}")

    docked = False
    dodge_dir = 0.0
    dodge_timer = 0
    step = 0
    flight_log = []
    os.makedirs("results", exist_ok=True)

    print("[READY] Real-time control loop running at 20 Hz. Press Ctrl+C to emergency stop.")

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
                if video_writer is not None and annotated is not None:
                    video_writer.write(annotated)
                if r is not None:
                    ekf.update_aruco(r, phi, dock_pos=marker_pos)

            # 3. Docking terminal condition
            dist_to_dock = np.hypot(target_dock[0] - ekf.x[0], target_dock[1] - ekf.x[1])
            if dist_to_dock < 0.10 or (r is not None and r < 0.20):
                print(f"\n[DOCK SUCCESS] Target docking plate reached! Stopping robot.")
                docked = True
                robot.set_velocity(0.0, 0.0)
                # Write final frames with docked overlay
                if video_writer is not None and frame is not None:
                    _, _, final_annotated = aruco.detect(frame, annotate=True, is_docked=True)
                    for _ in range(30):  # 1.5s freeze frame on docked status
                        video_writer.write(final_annotated)
                break

            # 4. LiDAR obstacle processing
            ranges = robot.get_lidar_scan()
            d_front = float(np.min(ranges[150:210]))
            d_left = float(np.mean(ranges[195:255]))
            d_right = float(np.mean(ranges[105:165]))

            angle_to_dock = np.arctan2(target_dock[1] - ekf.x[1], target_dock[0] - ekf.x[0])
            heading_err = wrap(angle_to_dock - ekf.x[2])

            # 5. Two-phase Control Law (Visual Servoing vs Obstacle Avoidance at 20 Hz)
            if r is not None and dist_to_dock < 1.0:
                dodge_timer = 0
                cmd_w = float(np.clip(2.5 * phi, -0.6, 0.6))
                cmd_v = float(np.clip(0.12 * np.cos(phi), 0.03, 0.12))
            elif dodge_timer <= 0 and d_front < 0.45 and dist_to_dock > 0.4:
                dodge_dir = 1.0 if d_left >= d_right else -1.0
                dodge_timer = 24  # 1.2s total at 20 Hz
                cmd_v = 0.02
                cmd_w = dodge_dir * 0.75
            elif dodge_timer > 0:
                dodge_timer -= 1
                if dodge_timer > 12:
                    # Phase 1 (0.6s): rotate away with creeping forward speed
                    cmd_v = 0.03
                    cmd_w = dodge_dir * 0.75
                else:
                    # Phase 2 (0.6s): drive forward past the flank
                    cmd_v = 0.14
                    cmd_w = dodge_dir * 0.15
            elif abs(heading_err) > np.deg2rad(45):
                cmd_v = 0.0
                cmd_w = float(np.clip(2.0 * heading_err, -0.7, 0.7))
            else:
                cmd_v = float(np.clip(0.15 * np.cos(heading_err), 0.05, 0.15))
                cmd_w = float(np.clip(1.5 * heading_err, -0.5, 0.5))

            # Send actuation
            robot.set_velocity(cmd_v, cmd_w)

            # Record telemetry for quantitative comparison against simulation
            flight_log.append([
                step * robot.dt, ekf.x[0], ekf.x[1], ekf.x[2],
                np.sqrt(ekf.P[0, 0]), np.sqrt(ekf.P[1, 1]), np.sqrt(ekf.P[2, 2]),
                cmd_v, cmd_w, 1 if r is not None else 0,
                r if r is not None else -1.0, phi if phi is not None else 0.0
            ])

            # Telemetry logging every 20 steps (1.0 sec)
            if step % 20 == 0:
                status = f"LOCKED r={r:.2f}m" if r is not None else f"SEEKING (front={d_front:.2f}m)"
                print(f"[{step:04d}] Est: [{ekf.x[0]:.2f}, {ekf.x[1]:.2f}, {np.rad2deg(ekf.x[2]):.0f}°] | Dist: {dist_to_dock:.2f}m | {status}")

            # Enforce 20 Hz rate
            elapsed = time.time() - t_start
            sleep_time = max(0.0, robot.dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[STOP] User interrupted mission.")
    finally:
        robot.stop()
        if flight_log:
            csv_path = "results/real_docking_log.csv"
            header = "time,x_est,y_est,theta_est,sigma_x,sigma_y,sigma_theta,cmd_v,cmd_w,aruco_visible,aruco_range,aruco_bearing"
            np.savetxt(csv_path, np.array(flight_log), delimiter=",", header=header, comments="")
            print(f"[DATA] Hardware flight log saved to: {csv_path}")
        if video_writer is not None:
            video_writer.release()
            print(f"[RECORDER] Mission video successfully saved to: {args.video_out}")
        if cam_window is not None:
            cam_window.close()
        print("[SHUTDOWN] TurtleBot hardware safely stopped.")


if __name__ == "__main__":
    main()
