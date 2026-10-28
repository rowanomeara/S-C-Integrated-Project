import numpy as np


class LidarProcessor:
    def __init__(self, danger_distance=0.55):
        self.danger_distance = danger_distance

    def get_front_clearance(self, ranges):
        """
        Returns (min_distance, is_clear) in front 100-degree cone (indices 130:230, where 180 = straight ahead).
        Widened cone ensures foot/obstacle detection across the entire robot frontal profile.
        """
        front_cone = ranges[130:230]
        min_dist = float(np.min(front_cone))
        return min_dist, min_dist > self.danger_distance

    def get_avoidance_turn(self, ranges):
        """Returns +1 to steer left, -1 to steer right based on open space."""
        left_clear = float(np.mean(ranges[185:255]))
        right_clear = float(np.mean(ranges[105:175]))
        return 1.0 if left_clear >= right_clear else -1.0
