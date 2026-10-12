import numpy as np
from scipy.linalg import solve_continuous_are
from src.estimation.ekf import wrap


class UnicycleLQR:
    """Error-state LQR tracker"""
    def __init__(self, v_nom=0.18, q_c=4.0, q_th=1.0, r=0.2):
        self.v_nom = v_nom
        # Linearised lateral unicycle error dynamics: [e_c, e_th]^T
        A = np.array([[0.0, v_nom], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        Q = np.diag([q_c, q_th])
        R = np.array([[r]])

        P = solve_continuous_are(A, B, Q, R)
        self.K = (np.linalg.inv(R) @ B.T @ P).flatten()  # [k_cross_track, k_heading]

    def compute_cmd(self, x_est, waypoint, max_w=0.6):
        dx = waypoint[0] - x_est[0]
        dy = waypoint[1] - x_est[1]
        dist = np.hypot(dx, dy)
        theta_d = np.arctan2(dy, dx)
        e_th = wrap(theta_d - x_est[2])
        e_c = dist * np.sin(e_th)

        omega = float(np.clip(self.K[0] * e_c + self.K[1] * e_th, -max_w, max_w))
        v = float(np.clip(self.v_nom * np.cos(e_th), 0.0, self.v_nom))
        return v, omega, dist
