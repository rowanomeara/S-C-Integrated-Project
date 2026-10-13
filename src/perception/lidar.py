import numpy as np


class LidarProcessor:
    def __init__(self, danger_distance=0.45):
        self.danger_distance = danger_distance

    def get_front_clearance(self, ranges):
        # returns (min_distance, is_clear) in front 60-degree cone (180 = straight ahead)
        front_cone = ranges[45:360]
        min_dist = float(np.min(front_cone))
        return min_dist, min_dist > self.danger_distance

    def get_avoidance_turn(self, ranges):
        # returns +1 to steer left, -1 to steer right based on open space
        left_clear = float(np.mean(ranges[210:270]))
        right_clear = float(np.mean(ranges[90:150]))
        return 1.0 if left_clear >= right_clear else -1.0

