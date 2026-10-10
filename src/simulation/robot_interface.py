

import mujoco
import numpy as np


class TurtleBotSim:
    def __init__(self, scene_path="assets/scenes/arena.xml", dt=0.01):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.dt = dt

        self.r = 0.033   # Wheel radius (m)
        self.W = 0.160   # Wheelbase (m)

        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def set_velocity(self, v, omega):
        # Differential drive inverse kinematics
        w_l = (v - 0.5 * omega * self.W) / self.r
        w_r = (v + 0.5 * omega * self.W) / self.r
        self.data.ctrl[:] = [w_l, w_r]

    def get_odometry(self):
        # Degrees of freedom 6 and 7 (heh) are the wheels
        w_l, w_r = self.data.qvel[6], self.data.qvel[7]
        v = self.r * (w_r + w_l) / 2.0
        omega = self.r * (w_r - w_l) / self.W
        return v, omega

    def get_lidar_scan(self, num_beams=360):
        pos = self.data.site("lidar_site").xpos
        R = self.data.body("base").xmat.reshape(3, 3)
        angles = np.linspace(-np.pi, np.pi, num_beams, endpoint=False)
        geom_id = np.zeros(1, dtype=np.int32)

        ranges = []
        for a in angles:
            ray_dir = R @ np.array([np.cos(a), np.sin(a), 0.0])
            d = mujoco.mj_ray(self.model, self.data, pos, ray_dir, None, 1, 1, geom_id)
            ranges.append(d if d >= 0 else 5.0)

        return np.clip(ranges, 0.0, 5.0)

    def get_camera_frame(self):
        self.renderer.update_scene(self.data, camera="front_camera")
        return self.renderer.render()

    def get_ground_truth_pose(self):
        x, y = self.data.qpos[:2]
        qw, qx, qy, qz = self.data.qpos[3:7]
        theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        return np.array([x, y, theta])