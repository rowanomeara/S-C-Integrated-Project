import numpy as np


class LidarProcessor:
    def __init__(self, danger_distance=0.45):
        self.danger_distance = danger_distance

    def get_front_clearance(self, ranges):
        #eturns (min_distance, is_clear) in the front 60-degrees 
        #180 = straight ahead
        front_cone = ranges[150:210]
        min_dist = np.min(front_cone)
        is_clear = min_dist > self.danger_distance
        return float(min_dist), is_clear
