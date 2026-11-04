"""Control: Multi-Mode Docking Controller.

Implements state-feedback control for autonomous docking:
1. Reactive obstacle avoidance via LiDAR sector monitoring.
2. Position-Based Visual Servoing (PBVS) locking onto ArUco visual line-of-sight.
3. Terminal corridor line tracking regulating heading and lateral offset.
4. Far-field pursuit for initial arena approach.

Uses estimated state x_est exclusively (no access to ground truth).
"""

import numpy as np
from src.estimation.ekf import wrap


class DockingController:
    """State-feedback docking controller with PBVS visual servoing and collision avoidance."""

    def __init__(self, target_dock=np.array([-1.88, 0.0])):
        """Initialize controller parameters and evasion state.

        Parameters:
          target_dock: World coordinates of the docking station target point [x, y].
        """
        self.target_dock = np.asarray(target_dock, dtype=float)
        self.dodge_timer = 0
        self.dodge_dir = 0.0
        self.prev_w = 0.0

    def compute(self, x_est, r, phi, d_front, d_left, d_right, is_real=False):
        """Compute linear velocity v and angular velocity omega commands.

        Parameters:
          x_est: Estimated robot pose [x, y, theta] from EKF.
          r: Observed ArUco range in meters (or None if unobserved).
          phi: Observed ArUco bearing in radians (or None if unobserved).
          d_front: Minimum LiDAR distance in the forward cone (meters).
          d_left: Mean LiDAR distance in the left flank sector (meters).
          d_right: Mean LiDAR distance in the right flank sector (meters).
          is_real: Boolean flag selecting hardware gains and slew limiters.

        Returns:
          cmd_v: Commanded linear velocity (m/s).
          cmd_w: Commanded angular velocity (rad/s).
          dist_to_dock: Estimated Euclidean distance to the docking station (m).
        """
        # Distance to dock target computed strictly from estimated state
        dist_to_dock = float(np.hypot(self.target_dock[0] - x_est[0], self.target_dock[1] - x_est[1]))
        max_v = 0.10 if is_real else 0.18
        dodge_steps = 20 if is_real else 120
        vs_dist = 2.5 if is_real else 1.0

        # -------------------------------------------------------------
        # Mode 1: Reactive Obstacle Avoidance
        # -------------------------------------------------------------
        # Thresholds: emergency brake stops forward motion; dodge threshold triggers crawl and turn
        emergency_dist = 0.32 if is_real else 0.26
        dodge_dist = 0.55 if is_real else 0.48

        if d_front < emergency_dist and dist_to_dock > 0.25:
            # Emergency stop: halt forward drive, spin toward the flank with greater clearance
            self.dodge_dir = 1.0 if d_left >= d_right else -1.0
            self.dodge_timer = dodge_steps
            raw_v = 0.0
            raw_w = self.dodge_dir * (0.32 if is_real else 0.75)

        elif self.dodge_timer <= 0 and d_front < dodge_dist and dist_to_dock > 0.30:
            # Proactive evasion: reduce speed and turn away from the detected hazard
            self.dodge_dir = 1.0 if d_left >= d_right else -1.0
            self.dodge_timer = dodge_steps
            raw_v = 0.02
            raw_w = self.dodge_dir * (0.28 if is_real else 0.75)

        elif self.dodge_timer > 0:
            # Active dodge sequence: counts down steps before restoring nominal navigation
            self.dodge_timer -= 1
            if is_real and d_front < emergency_dist:
                raw_v = 0.0
                raw_w = self.dodge_dir * 0.30
            elif self.dodge_timer > (dodge_steps // 2):
                raw_v = 0.03
                raw_w = self.dodge_dir * (0.25 if is_real else 0.75)
            else:
                raw_v = 0.08 if is_real else 0.15
                raw_w = self.dodge_dir * (0.08 if is_real else 0.15)

        # -------------------------------------------------------------
        # Mode 2: Position-Based Visual Servoing (PBVS)
        # -------------------------------------------------------------
        # Engages when ArUco marker is sighted within approach range
        elif r is not None and dist_to_dock < vs_dist:
            self.dodge_timer = 0
            # Proportional bearing gain: reduced on physical robot to prevent oscillation over Wi-Fi
            k_phi = 0.65 if is_real else 2.5
            w_max = 0.22 if is_real else 0.60
            raw_w = float(np.clip(k_phi * phi, -w_max, w_max))

            # Forward speed modulated by cos(phi): slows down when bearing error is large
            v_gain = 0.08 if is_real else 0.14
            raw_v = float(np.clip(v_gain * np.cos(phi), 0.03, max_v))

        # -------------------------------------------------------------
        # Mode 3: Terminal Corridor Line Tracking
        # -------------------------------------------------------------
        # Engages inside the narrow docking corridor to eliminate line-of-sight singularities
        elif dist_to_dock < (0.40 if not is_real else 0.90):
            self.dodge_timer = 0
            # Regulate heading toward dock centerline orientation (theta = pi rad)
            th_err = wrap(np.pi - x_est[2])
            # Regulate lateral cross-track error toward y = target_dock[1] (centerline y = 0)
            y_err = self.target_dock[1] - x_est[1]
            w_limit = 0.22 if is_real else 0.60
            raw_w = float(np.clip(0.60 * th_err + 0.50 * y_err, -w_limit, w_limit))
            raw_v = float(np.clip(0.08 * np.cos(th_err), 0.03, max_v))

        # -------------------------------------------------------------
        # Mode 4: Far-Field Pursuit
        # -------------------------------------------------------------
        # Long-range navigation toward the docking station entry point
        else:
            angle_to_dock = np.arctan2(self.target_dock[1] - x_est[1], self.target_dock[0] - x_est[0])
            heading_err = wrap(angle_to_dock - x_est[2])
            k_turn = 0.80 if is_real else 2.0
            turn_limit = 0.30 if is_real else 0.80
            k_track = 0.60 if is_real else 1.5
            track_limit = 0.25 if is_real else 0.60

            # Turn in place if heading error exceeds 45 degrees
            if abs(heading_err) > np.deg2rad(45):
                raw_v = 0.0
                raw_w = float(np.clip(k_turn * heading_err, -turn_limit, turn_limit))
            else:
                raw_v = float(np.clip(max_v * np.cos(heading_err), 0.04, max_v))
                raw_w = float(np.clip(k_track * heading_err, -track_limit, track_limit))

        # -------------------------------------------------------------
        # Angular Acceleration Slew Limiter (Hardware Stability)
        # -------------------------------------------------------------
        # Clamps delta omega to 0.04 rad/step (0.8 rad/s^2 at 20 Hz) to eliminate network lag hunting
        if is_real:
            max_dw = 0.04
            dw = float(np.clip(raw_w - self.prev_w, -max_dw, max_dw))
            cmd_w = self.prev_w + dw
        else:
            cmd_w = raw_w

        self.prev_w = cmd_w
        return raw_v, cmd_w, dist_to_dock
