import numpy as np
import mujoco
import mujoco.viewer
from src.simulation.robot_interface import TurtleBotSim
from src.perception.aruco_detector import ArucoDetector
from src.perception.camera_viewer import CameraWindow
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap


def spawn_robot(robot, ekf, rng):
    """Spawns robot at random location with heading variation."""
    x_rand = float(rng.uniform(0.85, 1.25))
    y_rand = float(rng.uniform(-0.50, 0.50))
    th_rand = float(np.pi + rng.uniform(-np.deg2rad(45), np.deg2rad(45)))

    robot.data.qpos[0] = x_rand
    robot.data.qpos[1] = y_rand
    robot.data.qpos[2] = 0.033
    robot.data.qpos[3] = np.cos(th_rand / 2.0)
    robot.data.qpos[4] = 0.0
    robot.data.qpos[5] = 0.0
    robot.data.qpos[6] = np.sin(th_rand / 2.0)
    robot.data.qvel[:] = 0.0
    mujoco.mj_forward(robot.model, robot.data)

    ekf.x = np.array([x_rand, y_rand, th_rand], dtype=float)
    ekf.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
    print(f"\n[SPAWN] Robot spawned at [{x_rand:.2f}, {y_rand:.2f}], heading={np.rad2deg(th_rand):.0f}°")


def main():
    print(r"=== Launching MuJoCo TurtleBot3 Simulation with Live Camera POV ===")
    robot = TurtleBotSim()
    aruco = ArucoDetector()
    lidar = LidarProcessor(danger_distance=0.42)
    ekf = RobotEKF()
    cam_window = CameraWindow()

    rng = np.random.default_rng()
    spawn_robot(robot, ekf, rng)

    target_dock = np.array([-1.52, 0.0])
    marker_pos = np.array([-1.69, 0.0])

    docked = False
    dodge_dir = 0.0
    dodge_timer = 0
    prev_time = robot.data.time

    try:
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            step = 0
            while viewer.is_running():
                step += 1

                # Detect viewer Reset button press (time resets to ~0)
                if robot.data.time < prev_time - 0.05:
                    spawn_robot(robot, ekf, rng)
                    docked = False
                    dodge_timer = 0
                    step = 0
                prev_time = robot.data.time

                # 1. Odometry prediction in EKF
                v_odom, w_odom = robot.get_odometry()
                ekf.predict(v_odom, w_odom, dt=robot.dt)

                # 2. Camera POV stream & ArUco detection
                frame = robot.get_camera_frame() if step % 2 == 0 else None
                if frame is not None:
                    r, phi, annotated = aruco.detect(frame, annotate=True, is_docked=docked)
                    cam_window.show(annotated)
                    if r is not None:
                        ekf.update_aruco(r, phi, dock_pos=marker_pos)
                else:
                    r, phi = None, None

                # 3. Check docking condition
                dist_to_dock = np.hypot(target_dock[0] - ekf.x[0], target_dock[1] - ekf.x[1])
                if dist_to_dock < 0.08 or (r is not None and r < 0.18):
                    if not docked:
                        print(f"[DOCK SUCCESS] Robot reached ArUco station and stopped!")
                        docked = True
                    robot.set_velocity(0.0, 0.0)
                    robot.step()
                    viewer.sync()
                    continue

                # 4. LiDAR scan & navigation
                ranges = robot.get_lidar_scan()
                d_front = float(np.min(ranges[150:210]))
                d_left = float(np.mean(ranges[195:255]))
                d_right = float(np.mean(ranges[105:165]))

                angle_to_dock = np.arctan2(target_dock[1] - ekf.x[1], target_dock[0] - ekf.x[0])
                heading_err = wrap(angle_to_dock - ekf.x[2])

                # State-feedback control on estimated state
                if r is not None and dist_to_dock < 1.0:
                    # --- VISUAL SERVOING CONTROLLER (PBVS / Direct Optical Feedback) ---
                    # 1. Angular rate: proportional feedback on image bearing error (phi -> 0).
                    #    Zeroes the error between marker centroid and camera optical center (cx=320).
                    dodge_timer = 0
                    cmd_w = float(np.clip(2.5 * phi, -0.6, 0.6))
                    # 2. Linear velocity: scaled by cos(phi) to slow down during turns,
                    #    preventing the marker from exiting the camera's Field of View (FOV).
                    cmd_v = float(np.clip(0.14 * np.cos(phi), 0.04, 0.14))
                elif dodge_timer <= 0 and d_front < 0.48 and dist_to_dock > 0.4:
                    # Obstacle directly ahead -> steer around it toward clearer side
                    dodge_dir = 1.0 if d_left >= d_right else -1.0
                    dodge_timer = 120
                    cmd_v = 0.02
                    cmd_w = dodge_dir * 0.8
                elif dodge_timer > 0:
                    # Two-phase dodge: orient first, then drive past obstacle
                    dodge_timer -= 1
                    if dodge_timer > 60:
                        cmd_v = 0.03
                        cmd_w = dodge_dir * 0.75
                    else:
                        cmd_v = 0.15
                        cmd_w = dodge_dir * 0.15
                elif abs(heading_err) > np.deg2rad(45):
                    # Orient toward dock
                    cmd_v = 0.0
                    cmd_w = float(np.clip(2.0 * heading_err, -0.8, 0.8))
                else:
                    # Drive forward towards dock
                    cmd_v = float(np.clip(0.18 * np.cos(heading_err), 0.06, 0.18))
                    cmd_w = float(np.clip(1.5 * heading_err, -0.6, 0.6))

                robot.set_velocity(cmd_v, cmd_w)
                robot.step()
                viewer.sync()

                if step % 100 == 0:
                    true_pose = robot.get_ground_truth_pose()
                    err = np.linalg.norm(ekf.x[:2] - true_pose[:2])
                    print(f"Step {step:04d} | Est: [{ekf.x[0]:.2f}, {ekf.x[1]:.2f}] | True: [{true_pose[0]:.2f}, {true_pose[1]:.2f}] | Pos Err: {err*100:.1f} cm | Front: {d_front:.2f}m")
    finally:
        cam_window.close()

    print("\n[Done] Simulation session closed.")


if __name__ == "__main__":
    main()
