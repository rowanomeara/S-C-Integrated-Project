"""Control: Linear Quadratic Regulator (LQR) Trajectory Tracking Benchmark.

Implements an optimal state-feedback controller for lateral unicycle error dynamics,
used to compare performance against the primary non-linear PBVS controller.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from src.estimation.ekf import wrap


class UnicycleLQR:
    """Optimal error-state LQR tracker for planar unicycle waypoints."""

    def __init__(self, v_nom=0.18, q_c=4.0, q_th=1.0, r=0.2):
        """Initialize LQR matrices and solve the Continuous Algebraic Riccati Equation.

        Linearised lateral error state: x = [e_c, e_th]^T (cross-track error, heading error)
        Control input: u = omega (yaw rate command)

        Continuous dynamics:
          dot(e_c)  = v_nom * e_th
          dot(e_th) = omega

        Parameters:
          v_nom: Nominal forward velocity for Jacobian linearization in m/s (0.18 m/s).
          q_c: State penalty on lateral cross-track error (4.0).
          q_th: State penalty on heading error (1.0).
          r: Control penalty on angular steering effort (0.2).
        """
        self.v_nom = v_nom

        # Continuous-time state transition matrix A and input matrix B
        A = np.array([[0.0, v_nom], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])

        # Cost weighting matrices: Q (state error penalty), R (actuator effort penalty)
        Q = np.diag([q_c, q_th])
        R = np.array([[r]])

        # Solve Continuous Algebraic Riccati Equation (CARE):
        # A^T * P + P * A - P * B * R^-1 * B^T * P + Q = 0
        P = solve_continuous_are(A, B, Q, R)

        # Compute optimal steady-state feedback gain matrix K = R^-1 * B^T * P
        self.K = (np.linalg.inv(R) @ B.T @ P).flatten()  # [k_cross_track, k_heading]

    def compute_cmd(self, x_est, waypoint, max_w=0.6):
        """Compute optimal control signals to steer toward the target waypoint.

        Parameters:
          x_est: Estimated robot state [x, y, theta] from EKF.
          waypoint: Target coordinates [x_target, y_target].
          max_w: Saturation threshold for angular velocity (rad/s).

        Returns:
          v: Linear forward speed command (m/s).
          omega: Angular yaw rate command (rad/s).
          dist: Euclidean distance to target waypoint (m).
        """
        # Vector from estimated robot position to target waypoint
        dx = waypoint[0] - x_est[0]
        dy = waypoint[1] - x_est[1]
        dist = np.hypot(dx, dy)

        # Desired heading pointing directly toward the waypoint
        theta_d = np.arctan2(dy, dx)
        e_th = wrap(theta_d - x_est[2])

        # Perpendicular cross-track error to the line-of-sight vector
        e_c = dist * np.sin(e_th)

        # Optimal state feedback: omega = K[0] * e_c + K[1] * e_th
        omega = float(np.clip(self.K[0] * e_c + self.K[1] * e_th, -max_w, max_w))

        # Modulate forward speed by cosine of heading error to slow down during turns
        v = float(np.clip(self.v_nom * np.cos(e_th), 0.0, self.v_nom))

        return v, omega, dist
