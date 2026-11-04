"""Main: Interactive Closed-Loop Simulation with MuJoCo Viewer and Camera POV.

Launches the complete integrated pipeline in simulation:
1. Differential drive unicycle physics at 100 Hz.
2. Extended Kalman Filter (EKF) state estimation fusing odometry and ArUco sightings.
3. Multi-mode docking controller with PBVS visual servoing and LiDAR obstacle evasion.
4. Real-time 3D passive viewer and live camera HUD display.
"""

import os
import sys

# On macOS, MuJoCo passive viewer requires mjpython trampoline for AppKit event loop integration
if sys.platform == "darwin" and "MJPYTHON_LIBPYTHON" not in os.environ and "mjpython" not in sys.executable:
    mjpy = os.path.join(os.path.dirname(sys.executable), "mjpython")
    if os.path.exists(mjpy):
        os.execv(mjpy, [mjpy] + sys.argv)

import numpy as np
import mujoco
import mujoco.viewer
from src.simulation.robot_interface import TurtleBotSim
from src.perception.aruco_detector import ArucoDetector
from src.perception.camera_viewer import CameraWindow
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap
from src.control import DockingController


def spawn_robot(robot, ekf, rng):
    """Spawn the robot at a randomized starting position with heading variations.

    Initial bounds:
      x: Uniform in [0.85, 1.25] meters.
      y: Uniform in [-0.50, 0.50] meters.
      theta: Uniform in [pi - 45 deg, pi + 45 deg] radians (facing toward the docking bay).

    Parameters:
      robot: TurtleBotSim instance.
      ekf: RobotEKF instance to re-initialize with initial pose and prior covariance.
      rng: NumPy random generator instance.
    """
    x_rand = float(rng.uniform(0.85, 1.25))
    y_rand = float(rng.uniform(-0.50, 0.50))
    th_rand = float(np.pi + rng.uniform(-np.deg2rad(45), np.deg2rad(45)))

    # Set MuJoCo free-joint generalized coordinates [x, y, z, qw, qx, qy, qz]
    robot.data.qpos[0] = x_rand
    robot.data.qpos[1] = y_rand
    robot.data.qpos[2] = 0.033  # Elevation matching wheel radius
    robot.data.qpos[3] = np.cos(th_rand / 2.0)  # Quaternion qw
    robot.data.qpos[4] = 0.0                     # Quaternion qx
    robot.data.qpos[5] = 0.0                     # Quaternion qy
    robot.data.qpos[6] = np.sin(th_rand / 2.0)  # Quaternion qz
    robot.data.qvel[:] = 0.0                    # Zero initial joint velocities
    mujoco.mj_forward(robot.model, robot.data)

    # Initialize EKF state estimate and reset initial covariance
    ekf.x = np.array([x_rand, y_rand, th_rand], dtype=float)
    ekf.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])
    print(f"\n[SPAWN] Robot spawned at [{x_rand:.2f}, {y_rand:.2f}], heading={np.rad2deg(th_rand):.0f}°")


def main():
    """Execute main interactive simulation loop."""
    print(r"=== Launching MuJoCo TurtleBot3 Simulation with Live Camera POV ===")
    robot = TurtleBotSim()
    aruco = ArucoDetector()
    lidar = LidarProcessor(danger_distance=0.42)
    ekf = RobotEKF()
    cam_window = CameraWindow()

    rng = np.random.default_rng()
    spawn_robot(robot, ekf, rng)

    # Target docking position and physical landmark coordinates
    target_dock = np.array([-1.52, 0.0])  # Desired robot stopping position inside corridor
    marker_pos = np.array([-1.69, 0.0])   # Physical location of ArUco marker mounted on dock wall

    docked = False
    controller = DockingController(target_dock=target_dock)
    prev_time = robot.data.time

    try:
        # Launch non-blocking interactive 3D scene viewer
        with mujoco.viewer.launch_passive(robot.model, robot.data) as viewer:
            step = 0
            while viewer.is_running():
                step += 1

                # Detect GUI reset button press: restarts mission from a new random initial pose
                if robot.data.time < prev_time - 0.05:
                    spawn_robot(robot, ekf, rng)
                    docked = False
                    controller = DockingController(target_dock=target_dock)
                    step = 0
                prev_time = robot.data.time

                # ---------------------------------------------------------
                # 1. EKF State Prediction (Proprioceptive Odometry)
                # ---------------------------------------------------------
                v_odom, w_odom = robot.get_odometry()
                ekf.predict(v_odom, w_odom, dt=robot.dt)

                # ---------------------------------------------------------
                # 2. Camera Capture, Marker Detection & EKF Update (Exteroceptive)
                # ---------------------------------------------------------
                # Render camera frame every 2 steps (50 Hz equivalent rendering rate)
                frame = robot.get_camera_frame() if step % 2 == 0 else None
                if frame is not None:
                    r, phi, annotated = aruco.detect(frame, annotate=True, is_docked=docked)
                    cam_window.show(annotated)
                    if r is not None:
                        # Update EKF with visual observation to bound odometry drift
                        ekf.update_aruco(r, phi, dock_pos=marker_pos)
                else:
                    r, phi = None, None

                # ---------------------------------------------------------
                # 3. LiDAR Clearance Evaluation
                # ---------------------------------------------------------
                ranges = robot.get_lidar_scan()
                d_front, _ = lidar.get_front_clearance(ranges)
                d_left = float(np.mean(ranges[195:255]))
                d_right = float(np.mean(ranges[105:165]))

                # ---------------------------------------------------------
                # 4. Docking Completion Check
                # ---------------------------------------------------------
                dist_to_dock = np.hypot(target_dock[0] - ekf.x[0], target_dock[1] - ekf.x[1])
                if dist_to_dock < 0.18 or (r is not None and r < 0.25) or (dist_to_dock < 0.35 and d_front < 0.25):
                    if not docked:
                        print(f"[DOCK SUCCESS] Robot reached ArUco station and stopped!")
                        docked = True
                    # Zero wheel commands and hold position
                    robot.set_velocity(0.0, 0.0)
                    robot.step()
                    viewer.sync()
                    continue

                # ---------------------------------------------------------
                # 5. Closed-Loop Control Execution
                # ---------------------------------------------------------
                # Command computation uses estimated state ekf.x strictly (never ground truth)
                cmd_v, cmd_w, _ = controller.compute(ekf.x, r, phi, d_front, d_left, d_right)

                robot.set_velocity(cmd_v, cmd_w)
                robot.step()
                viewer.sync()

                # Print telemetry every 100 steps (1 second) comparing estimated pose against ground truth
                if step % 100 == 0:
                    true_pose = robot.get_ground_truth_pose()
                    err = np.linalg.norm(ekf.x[:2] - true_pose[:2])
                    print(f"Step {step:04d} | Est: [{ekf.x[0]:.2f}, {ekf.x[1]:.2f}] | True: [{true_pose[0]:.2f}, {true_pose[1]:.2f}] | Pos Err: {err*100:.1f} cm | Front: {d_front:.2f}m")
    finally:
        cam_window.close()

    print("\n[Done] Simulation session closed.")


if __name__ == "__main__":
    main()
