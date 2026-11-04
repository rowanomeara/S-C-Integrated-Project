"""Estimation: Extended Kalman Filter (EKF) for Unicycle State Estimation.

Fuses wheel odometry (prediction) with camera-based ArUco range and bearing
observations (correction) to track the 3-DoF robot pose [x, y, theta].
"""

import numpy as np


def wrap(a):
    """Wrap angle to the interval [-pi, pi) radians."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotEKF:
    """Extended Kalman Filter estimating planar unicycle pose [x, y, theta]."""

    def __init__(self, x0=np.array([1.2, 0.0, np.pi])):
        """Initialize filter state, error covariance, and noise matrices.

        Parameters:
          x0: Initial state vector [x, y, theta] in world coordinates (default: [1.2m, 0.0m, pi rad]).
        """
        # State vector: [x (m), y (m), theta (rad)]
        self.x = x0.astype(float)

        # Initial estimation error covariance P0:
        # Standard deviations: 10 cm in x, 10 cm in y, 5 degrees in heading
        self.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])

        # Process noise covariance Q:
        # Accounts for wheel slip and unmodelled kinematics per step (2 cm position, 2 deg heading)
        self.Q = np.diag([0.02**2, 0.02**2, np.deg2rad(2)**2])

        # Measurement noise covariance R:
        # Calibrated sensor noise for ArUco observations (5 cm range, 3 deg bearing)
        self.R = np.diag([0.05**2, np.deg2rad(3)**2])

    def predict(self, v, omega, dt=0.01):
        """Propagate state and covariance forward using unicycle kinematics.

        Kinematic equations:
          x_{k+1}     = x_k + v * cos(theta_k) * dt
          y_{k+1}     = y_k + v * sin(theta_k) * dt
          theta_{k+1} = wrap(theta_k + omega * dt)

        Parameters:
          v: Linear forward velocity command (m/s).
          omega: Rotational angular velocity command (rad/s).
          dt: Integration time step (s).
        """
        th = self.x[2]

        # 1. State propagation f(x, u)
        self.x[0] += v * np.cos(th) * dt
        self.x[1] += v * np.sin(th) * dt
        self.x[2] = wrap(th + omega * dt)

        # 2. State transition Jacobian F = df/dx evaluated at current state:
        # [ 1,  0, -v*sin(th)*dt ]
        # [ 0,  1,  v*cos(th)*dt ]
        # [ 0,  0,        1      ]
        F = np.eye(3)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] =  v * np.cos(th) * dt

        # 3. Covariance propagation: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update_aruco(self, r_meas, phi_meas, dock_pos=np.array([-1.69, 0.0])):
        """Update state estimate with exteroceptive ArUco range and bearing observations.

        Measurement model:
          dx = x_dock - x_robot
          dy = y_dock - y_robot
          r_pred   = sqrt(dx^2 + dy^2)
          phi_pred = wrap(arctan2(dy, dx) - theta_robot)

        Parameters:
          r_meas: Observed Euclidean distance to the marker in meters.
          phi_meas: Observed bearing angle to the marker in radians.
          dock_pos: Known world coordinates of the docking landmark [x, y].
        """
        # 1. Predicted measurement vector h(x)
        dx = dock_pos[0] - self.x[0]
        dy = dock_pos[1] - self.x[1]
        r_pred = np.hypot(dx, dy)
        phi_pred = wrap(np.arctan2(dy, dx) - self.x[2])

        # 2. Measurement Jacobian H = dh/dx evaluated at predicted state:
        # row 0: dr/dx     = -dx / r,          dr/dy     = -dy / r,          dr/dtheta     = 0
        # row 1: dphi/dx   =  dy / r^2,        dphi/dy   = -dx / r^2,        dphi/dtheta   = -1
        H = np.zeros((2, 3))
        H[0, 0] = -dx / r_pred
        H[0, 1] = -dy / r_pred
        H[1, 0] =  dy / (r_pred**2)
        H[1, 1] = -dx / (r_pred**2)
        H[1, 2] = -1.0

        # 3. Measurement innovation (residual) with wrapped bearing error
        y = np.array([r_meas - r_pred, wrap(phi_meas - phi_pred)])

        # 4. Innovation covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # 5. Optimal Kalman gain K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # 6. State update: x = x + K * y
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])

        # 7. Covariance update: P = (I - K * H) * P
        self.P = (np.eye(3) - K @ H) @ self.P

        # Enforce numerical symmetry: P = 0.5 * (P + P^T)
        self.P = 0.5 * (self.P + self.P.T)
