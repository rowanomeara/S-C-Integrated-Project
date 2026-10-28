import numpy as np


class DockingController:
    """State-feedback docking controller with PBVS visual servoing, corridor tracking, and obstacle avoidance."""

    def __init__(self, target_dock=np.array([-1.88, 0.0])):
        self.target_dock = np.asarray(target_dock, dtype=float)
        self.dodge_timer = 0
        self.dodge_dir = 0.0
        self.prev_w = 0.0

    def compute(self, x_est, r, phi, d_front, d_left, d_right, is_real=False):
        """
        Computes (cmd_v, cmd_w, dist_to_dock) from estimated state, visual features, and LiDAR clearances.
        Uses estimated state only (never ground truth).
        """
        dist_to_dock = float(np.hypot(self.target_dock[0] - x_est[0], self.target_dock[1] - x_est[1]))
        max_v = 0.10 if is_real else 0.18
        dodge_steps = 20 if is_real else 100

        # 1. Obstacle avoidance (Emergency brake if d_front < 0.32m, reactive dodge if d_front < 0.55m)
        if d_front < 0.32 and dist_to_dock > 0.25:
            # Emergency close obstacle (e.g. foot right in front): STOP forward drive, rotate away
            self.dodge_dir = 1.0 if d_left >= d_right else -1.0
            self.dodge_timer = dodge_steps
            raw_v = 0.0
            raw_w = self.dodge_dir * (0.32 if is_real else 0.70)

        elif self.dodge_timer <= 0 and d_front < 0.55 and dist_to_dock > 0.30:
            # Obstacle detected ahead in safety buffer: crawl and turn towards open space
            self.dodge_dir = 1.0 if d_left >= d_right else -1.0
            self.dodge_timer = dodge_steps
            raw_v = 0.02
            raw_w = self.dodge_dir * (0.28 if is_real else 0.65)

        elif self.dodge_timer > 0:
            self.dodge_timer -= 1
            if d_front < 0.38:
                # Still blocked in front: do NOT drive forward into obstacle
                raw_v = 0.0
                raw_w = self.dodge_dir * (0.30 if is_real else 0.65)
            elif self.dodge_timer > (dodge_steps // 2):
                raw_v = 0.03
                raw_w = self.dodge_dir * (0.25 if is_real else 0.60)
            else:
                raw_v = 0.08 if is_real else 0.14
                raw_w = self.dodge_dir * (0.08 if is_real else 0.15)

        # 2. Visual Servoing (PBVS): lock directly on ArUco optical line-of-sight
        elif r is not None and dist_to_dock < 2.5:
            self.dodge_timer = 0
            # Damped proportional gain to prevent hunting/overshoot with Wi-Fi latency
            k_phi = 0.65 if is_real else 2.5
            w_max = 0.22 if is_real else 0.60
            raw_w = float(np.clip(k_phi * phi, -w_max, w_max))
            v_gain = 0.08 if is_real else 0.14
            raw_v = float(np.clip(v_gain * np.cos(phi), 0.03, max_v))

        # 3. Final Corridor Docking (when close to dock, eliminates pure-pursuit waypoint singularity)
        elif dist_to_dock < 0.90:
            self.dodge_timer = 0
            # Target heading points straight into dock (theta = pi)
            th_err = (np.pi - x_est[2] + np.pi) % (2 * np.pi) - np.pi
            # Cross-track error towards dock centerline (y = target_dock[1])
            y_err = self.target_dock[1] - x_est[1]
            raw_w = float(np.clip(0.60 * th_err + 0.50 * y_err, -0.22, 0.22))
            raw_v = float(np.clip(0.08 * np.cos(th_err), 0.03, max_v))

        # 4. Long-range Navigation towards Dock
        else:
            angle_to_dock = np.arctan2(self.target_dock[1] - x_est[1], self.target_dock[0] - x_est[0])
            heading_err = (angle_to_dock - x_est[2] + np.pi) % (2 * np.pi) - np.pi
            if abs(heading_err) > np.deg2rad(45):
                raw_v = 0.0
                raw_w = float(np.clip(0.80 * heading_err, -0.30, 0.30))
            else:
                raw_v = float(np.clip(max_v * np.cos(heading_err), 0.04, max_v))
                raw_w = float(np.clip(0.60 * heading_err, -0.25, 0.25))

        # Slew-rate limiting on angular velocity for hardware stability (prevents network-delay oscillations)
        if is_real:
            max_dw = 0.04  # max angular acceleration: 0.8 rad/s^2 at 20 Hz
            dw = float(np.clip(raw_w - self.prev_w, -max_dw, max_dw))
            cmd_w = self.prev_w + dw
        else:
            cmd_w = raw_w

        self.prev_w = cmd_w
        return raw_v, cmd_w, dist_to_dock
