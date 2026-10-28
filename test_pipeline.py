import unittest
import numpy as np

from src.estimation.ekf import RobotEKF, wrap
from src.control.docking_controller import DockingController
from src.control.model_analysis import analyze_controllability, analyze_observability
from src.perception.lidar import LidarProcessor


class TestPipeline(unittest.TestCase):

    def test_angle_wrap(self):
        """Angle wrapping strictly stays within [-pi, pi]."""
        self.assertAlmostEqual(wrap(0.5), 0.5)
        self.assertAlmostEqual(wrap(2 * np.pi), 0.0)
        self.assertAlmostEqual(wrap(3 * np.pi), -np.pi)  # wrap(pi) wraps into [-pi, pi)
        self.assertAlmostEqual(wrap(-3 * np.pi), -np.pi)
        self.assertAlmostEqual(np.cos(wrap(np.pi)), -1.0)
        self.assertAlmostEqual(np.sin(wrap(np.pi)), 0.0)

    def test_ekf_prediction_and_contraction(self):
        """EKF covariance grows under odometry prediction and contracts on visual update."""
        ekf = RobotEKF(x0=np.array([1.0, 0.0, np.pi]))
        p_initial_trace = np.trace(ekf.P)

        # 1. Prediction steps grow uncertainty
        for _ in range(10):
            ekf.predict(v=0.1, omega=0.0, dt=0.05)
        p_pred_trace = np.trace(ekf.P)
        self.assertGreater(p_pred_trace, p_initial_trace, "Covariance must grow during dead-reckoning")

        # 2. Visual measurement update contracts uncertainty
        ekf.update_aruco(r_meas=2.5, phi_meas=0.0, dock_pos=np.array([-1.52, 0.0]))
        p_post_trace = np.trace(ekf.P)
        self.assertLess(p_post_trace, p_pred_trace, "Covariance must contract upon sighting visual landmark")

        # 3. Covariance must remain strictly symmetric
        self.assertTrue(np.allclose(ekf.P, ekf.P.T), "Covariance matrix must remain symmetric")

    def test_docking_controller_saturation(self):
        """Controller strictly enforces velocity and yaw rate saturation limits."""
        controller = DockingController(target_dock=np.array([-1.52, 0.0]))

        # High bearing error command test
        v, w, dist = controller.compute(
            x_est=np.array([0.0, 0.0, 0.0]),
            r=2.0, phi=1.5,
            d_front=2.0, d_left=2.0, d_right=2.0
        )
        self.assertLessEqual(v, 0.18, "Linear velocity exceeded 0.18 m/s limit")
        self.assertGreaterEqual(v, 0.0, "Linear velocity must be non-negative")
        self.assertLessEqual(abs(w), 0.8, "Angular velocity exceeded 0.8 rad/s limit")

    def test_lidar_avoidance_turn(self):
        """LiDAR processor steers toward whichever flank has greater clearance."""
        lidar = LidarProcessor(danger_distance=0.45)
        ranges = np.ones(360) * 2.0
        ranges[150:210] = 0.30  # Obstacle ahead

        min_dist, is_clear = lidar.get_front_clearance(ranges)
        self.assertFalse(is_clear)
        self.assertAlmostEqual(min_dist, 0.30)

        ranges[195:255] = 3.0  # Left open
        ranges[105:165] = 0.5  # Right blocked
        turn = lidar.get_avoidance_turn(ranges)
        self.assertEqual(turn, 1.0, "Should steer left when left is open")

    def test_model_controllability_and_observability(self):
        """Part A model analysis satisfies full state controllability and observability."""
        self.assertTrue(analyze_controllability(), "Unicycle kinematics must satisfy LARC rank = 3")
        self.assertTrue(analyze_observability(), "Visual docking must achieve observability rank = 3")


if __name__ == "__main__":
    unittest.main()
