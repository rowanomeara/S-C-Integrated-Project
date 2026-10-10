import numpy as np
from src.simulation.robot_interface import TurtleBotSim
from src.perception.aruco_detector import ArucoDetector
from src.perception.lidar import LidarProcessor
from src.estimation.ekf import RobotEKF, wrap


def main():
    print("\\\ Launching Nominal Simulation Demo ///")
    robot = TurtleBotSim()
    aruco = ArucoDetector()
    lidar = LidarProcessor(danger_distance=0.45)
    ekf = RobotEKF()

    # Charging dock
    target_pos = np.array([-1.2, 0.1])

    for step in range(300):  # (300 steps, dt=0.01)
        # read odometry + EKF
        v_odom, w_odom = robot.get_odometry()
        ekf.predict(v_odom, w_odom, dt=robot.dt)

        #Camera + ArUco measurement
        cam_frame = robot.get_camera_frame()
        r_marker, phi_marker = aruco.detect(cam_frame)
        if r_marker is not None:
            ekf.update_aruco(r_marker, phi_marker, dock_pos=target_pos)

        # LiDAR obstacle clearance
        ranges = robot.get_lidar_scan()
        min_front, is_clear = lidar.get_front_clearance(ranges)

        # Controller
        x_est, y_est, th_est = ekf.x
        angle_to_dock = np.arctan2(target_pos[1] - y_est, target_pos[0] - x_est)
        heading_err = wrap(angle_to_dock - th_est)

        if not is_clear:
            cmd_v, cmd_w = 0.05, 0.5
        else:
            cmd_v = 0.15
            cmd_w = np.clip(1.5 * heading_err, -0.6, 0.6)

        robot.set_velocity(cmd_v, cmd_w)
        robot.step()

        if step % 50 == 0:
            true_pose = robot.get_ground_truth_pose()
            err_pos = np.linalg.norm(ekf.x[:2] - true_pose[:2])
            print(f"Step {step:03d} | Est: [{x_est:.2f}, {y_est:.2f}] | True: [{true_pose[0]:.2f}, {true_pose[1]:.2f}] | Error: {err_pos*100:.1f} cm | LiDAR Front: {min_front:.2f}m")

    print("\n[Yay] Simulation completed task nominally on estimated state.")


if __name__ == "__main__":
    main()
