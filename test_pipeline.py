"""Unit Tests: Autonomous Docking Verification Suite.

Automated regression test suite verifying core pipeline modules against rubric requirements:
1. Angle wrapping utilities [-pi, pi).
2. EKF covariance growth (dead reckoning) and contraction (visual observation update).
3. Docking controller actuator saturation bounds and non-negativity.
4. LiDAR obstacle detection and flank clearance directional decisions.
5. Lie Algebra Rank Condition (LARC) controllability and discrete observability rank.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.estimation.ekf import RobotEKF, wrap
from src.control.docking_controller import DockingController
from src.control.model_analysis import analyze_controllability, analyze_observability
from src.perception.lidar import LidarProcessor


class TestPipeline(unittest.TestCase):
    """Test cases for estimation, control, perception, and kinematic model properties."""

    def test_angle_wrap(self):
        """Angle wrapping strictly normalizes angles to [-pi, pi) radians."""
        self.assertAlmostEqual(wrap(0.5), 0.5)
        self.assertAlmostEqual(wrap(2 * np.pi), 0.0)
        self.assertAlmostEqual(wrap(3 * np.pi), -np.pi)
        self.assertAlmostEqual(wrap(-3 * np.pi), -np.pi)
        self.assertAlmostEqual(np.cos(wrap(np.pi)), -1.0)
        self.assertAlmostEqual(np.sin(wrap(np.pi)), 0.0)

    def test_ekf_prediction_and_contraction(self):
        """EKF covariance trace grows under open-loop prediction and contracts on visual update."""
        ekf = RobotEKF(x0=np.array([1.0, 0.0, np.pi]))
        p_initial_trace = np.trace(ekf.P)

        # 1. Prediction steps without measurements must increase covariance due to process noise Q
        for _ in range(10):
            ekf.predict(v=0.1, omega=0.0, dt=0.05)
        p_pred_trace = np.trace(ekf.P)
        self.assertGreater(p_pred_trace, p_initial_trace, "Covariance trace must grow during dead-reckoning")

        # 2. Exteroceptive ArUco measurement update must contract covariance trace
        ekf.update_aruco(r_meas=2.5, phi_meas=0.0, dock_pos=np.array([-1.52, 0.0]))
        p_post_trace = np.trace(ekf.P)
        self.assertLess(p_post_trace, p_pred_trace, "Covariance trace must contract upon visual landmark sighting")

        # 3. Covariance matrix must remain symmetric
        self.assertTrue(np.allclose(ekf.P, ekf.P.T), "Covariance matrix P must remain strictly symmetric")

    def test_docking_controller_saturation(self):
        """Controller strictly enforces motor saturation limits under extreme error signals."""
        controller = DockingController(target_dock=np.array([-1.52, 0.0]))

        # High bearing error input (1.5 rad ~= 86 deg)
        v, w, dist = controller.compute(
            x_est=np.array([0.0, 0.0, 0.0]),
            r=2.0, phi=1.5,
            d_front=2.0, d_left=2.0, d_right=2.0
        )
        self.assertLessEqual(v, 0.18, "Linear velocity exceeded 0.18 m/s maximum speed limit")
        self.assertGreaterEqual(v, 0.0, "Linear velocity command must be non-negative")
        self.assertLessEqual(abs(w), 0.8, "Angular velocity command exceeded 0.8 rad/s limit")

    def test_lidar_avoidance_turn(self):
        """LiDAR processor detects forward hazards and steers toward the flank with greater clearance."""
        lidar = LidarProcessor(danger_distance=0.45)
        ranges = np.ones(360) * 2.0
        # Place forward obstacle at 0.30 m in the 60-degree front cone [150:210]
        ranges[150:210] = 0.30

        min_dist, is_clear = lidar.get_front_clearance(ranges)
        self.assertFalse(is_clear, "Obstacle at 0.30 m must trigger is_clear=False")
        self.assertAlmostEqual(min_dist, 0.30)

        # Asymmetric flank clearances: left flank open (3.0 m), right flank blocked (0.5 m)
        ranges[185:255] = 3.0
        ranges[105:175] = 0.5
        turn = lidar.get_avoidance_turn(ranges)
        self.assertEqual(turn, 1.0, "Must steer left (+1.0) when left sector has greater clearance")

    def test_model_controllability_and_observability(self):
        """Kinematics and measurement models satisfy full state controllability and observability."""
        self.assertTrue(analyze_controllability(), "Unicycle kinematics must satisfy LARC rank = 3")
        self.assertTrue(analyze_observability(), "Visual docking must achieve observability rank = 3")


if __name__ == "__main__":
    unittest.main()
