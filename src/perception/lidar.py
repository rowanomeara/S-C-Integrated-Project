import numpy as np


class LidarProcessor:
    def __init__(self, danger_distance=0.45, cone_deg=60):
        self.danger_distance = danger_distance
        self.cone_deg = cone_deg

    def get_front_clearance(self, ranges, cone_deg=None):
        """
        Returns (min_distance, is_clear) in front cone (indices centered at 180 = straight ahead).
        Default 60-degree cone (ranges[150:210], ±30°) provides safety clearance
        for the robot body without falsely detecting side corridor blocks.
        """
        deg = cone_deg if cone_deg is not None else self.cone_deg
        half = deg // 2
        front_cone = ranges[180 - half : 180 + half]
        min_dist = float(np.min(front_cone))
        return min_dist, min_dist > self.danger_distance

    def get_avoidance_turn(self, ranges):
        """Returns +1 to steer left, -1 to steer right based on open space."""
        left_clear = float(np.mean(ranges[185:255]))
        right_clear = float(np.mean(ranges[105:175]))
        return 1.0 if left_clear >= right_clear else -1.0
