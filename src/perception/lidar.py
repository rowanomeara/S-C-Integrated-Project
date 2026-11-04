"""Perception: 2D LiDAR Scan Processing and Clearance Extraction.

This module processes raw 360-beam LDS-01 laser scans to evaluate front collision
hazards and determine reactive steering direction toward open space.
"""

import numpy as np


class LidarProcessor:
    """Extracts directional clearance sectors from 360-element laser scans."""

    def __init__(self, danger_distance=0.45, cone_deg=60):
        """Initialize the LiDAR processor.

        Parameters:
          danger_distance: Distance threshold in meters below which an obstacle triggers evasion (0.45 m).
          cone_deg: Angular width of the forward monitoring cone in degrees (default: 60 deg, +/-30 deg).
        """
        self.danger_distance = danger_distance
        self.cone_deg = cone_deg

    def get_front_clearance(self, ranges, cone_deg=None):
        """Compute the minimum distance in the forward sector.

        The 360-beam array maps index 180 to straight ahead (0 rad),
        with indices 0 and 359 corresponding to the rear (-pi and +pi rad).
        A 60-degree cone spans indices 150 to 210 (+/-30 deg from center).
        This angular aperture clears the robot chassis width without falsely
        detecting parallel docking corridor walls (corridor width = 0.70 m).

        Parameters:
          ranges: np.ndarray of 360 laser distance measurements in meters.
          cone_deg: Optional override for cone aperture in degrees.

        Returns:
          min_dist: float minimum distance detected within the forward cone.
          is_clear: bool indicating whether min_dist exceeds danger_distance.
        """
        deg = cone_deg if cone_deg is not None else self.cone_deg
        half = deg // 2
        # Slice forward monitoring sector centered at beam 180
        front_cone = ranges[180 - half : 180 + half]
        min_dist = float(np.min(front_cone))
        return min_dist, min_dist > self.danger_distance

    def get_avoidance_turn(self, ranges):
        """Determine reactive evasion direction based on flank space.

        Compares mean range in the left quadrant (indices 185 to 255)
        against the right quadrant (indices 105 to 175).

        Returns:
          +1.0 to steer counter-clockwise (left) if the left flank offers greater clearance.
          -1.0 to steer clockwise (right) if the right flank offers greater clearance.
        """
        left_clear = float(np.mean(ranges[185:255]))
        right_clear = float(np.mean(ranges[105:175]))
        return 1.0 if left_clear >= right_clear else -1.0
