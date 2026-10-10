#Unicycle Range & Bearing EKF

import numpy as np


def wrap(a):
    #Angle wrap [-pi, pi] 
    return (a + np.pi) % (2 * np.pi) - np.pi


class RobotEKF:
    def __init__(self, x0=np.array([-1.7, 0.5, -np.pi/2])):
        self.x = x0.astype(float)                # State estimate [x, y, theta]
        self.P = np.diag([0.1**2, 0.1**2, np.deg2rad(5)**2])  # Initial covariance

        # Process noise Q + measurement noise R
        self.Q = np.diag([0.02**2, 0.02**2, np.deg2rad(2)**2])
        self.R = np.diag([0.05**2, np.deg2rad(3)**2])  # -> 5cm range sigma, 3 deg bearing sigma

    def predict(self, v, omega, dt=0.01):
        # f_uni and Jacobian F_uni from unicycle
        th = self.x[2]

        # State propagation f(x, u)
        self.x[0] += v * np.cos(th) * dt
        self.x[1] += v * np.sin(th) * dt
        self.x[2] = wrap(th + omega * dt)

        # Jacobian F_t = df/dx
        F = np.eye(3)
        F[0, 2] = -v * np.sin(th) * dt
        F[1, 2] =  v * np.cos(th) * dt

        # Covariance propagation
        self.P = F @ self.P @ F.T + self.Q

    def update_aruco(self, r_meas, phi_meas, dock_pos=np.array([-1.2, 0.1])):
        #landmark range & bearing to update measurement 
        dx = dock_pos[0] - self.x[0]
        dy = dock_pos[1] - self.x[1]
        r_pred = np.hypot(dx, dy)
        phi_pred = wrap(np.arctan2(dy, dx) - self.x[2])

        # Measurement Jacobian H_t = dh/dx
        H = np.zeros((2, 3))
        H[0, 0] = -dx / r_pred
        H[0, 1] = -dy / r_pred
        H[1, 0] =  dy / (r_pred**2)
        H[1, 1] = -dx / (r_pred**2)
        H[1, 2] = -1.0

        # Innovation
        y = np.array([r_meas - r_pred, wrap(phi_meas - phi_pred)])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(3) - K @ H) @ self.P
