"""Simulation: MuJoCo Physics Interface for TurtleBot3 Burger.

Provides the Hardware Abstraction Layer (HAL) in simulation:
1. Differential drive forward and inverse kinematics.
2. 360-beam 2D laser rangefinder raycasting.
3. Offscreen RGB camera frame rendering.
4. Ground truth state extraction for benchmarking and validation.
"""

import mujoco
import numpy as np


class TurtleBotSim:
    """MuJoCo simulation wrapper providing sensor queries and actuator controls."""

    def __init__(self, scene_path="assets/scenes/arena.xml", dt=0.01):
        """Load MuJoCo arena model and configure physics parameters.

        Parameters:
          scene_path: Filepath to the XML scene definition.
          dt: Physics integration timestep in seconds (0.01 s = 100 Hz).
        """
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = dt
        self.dt = dt

        # TurtleBot3 Burger physical dimensions:
        # r = 0.033 m: wheel radius (33 mm)
        # W = 0.160 m: wheelbase track width between left and right wheels (160 mm)
        self.r = 0.033   # Wheel radius (m)
        self.W = 0.160   # Wheelbase (m)

        # Offscreen OpenGL renderer matching physical camera resolution (640x480)
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.reset()

    def reset(self):
        """Reset simulation state to keyframe 0 and propagate forward kinematics."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Advance physics by one integration timestep (0.01 s)."""
        mujoco.mj_step(self.model, self.data)

    def set_velocity(self, v, omega):
        """Convert chassis velocities to wheel rotational speeds (inverse kinematics).

        Equations:
          w_left  = (v - 0.5 * omega * W) / r
          w_right = (v + 0.5 * omega * W) / r
        """
        w_l = (v - 0.5 * omega * self.W) / self.r
        w_r = (v + 0.5 * omega * self.W) / self.r
        self.data.ctrl[:] = [w_l, w_r]

    def get_odometry(self):
        """Compute chassis linear and angular velocity from wheel speeds (forward kinematics).

        DoFs 0 to 5 represent the 6-DoF floating base.
        DoFs 6 and 7 represent the left and right wheel hinge joints.

        Equations:
          v     = r * (w_right + w_left) / 2
          omega = r * (w_right - w_left) / W
        """
        w_l, w_r = self.data.qvel[6], self.data.qvel[7]
        v = self.r * (w_r + w_l) / 2.0
        omega = self.r * (w_r - w_l) / self.W
        return v, omega

    def get_lidar_scan(self, num_beams=360):
        """Simulate a 360-degree laser scan using MuJoCo geometric raycasting.

        Casts 360 rays in the horizontal plane from the sensor mount site.
        Rotates beam directions from the body frame to the world frame using body rotation matrix R.
        Clamps distances to [0.0, 5.0] meters matching physical LDS-01 limits.
        """
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
        """Render and return the current RGB camera frame from the front camera mount."""
        self.renderer.update_scene(self.data, camera="front_camera")
        return self.renderer.render()

    def get_ground_truth_pose(self):
        """Extract exact 2D pose [x, y, theta] from physics state for validation.

        Converts unit quaternion [qw, qx, qy, qz] to planar yaw angle theta.
        Never passed to the controller (ensures state estimation rigor).
        """
        x, y = self.data.qpos[:2]
        qw, qx, qy, qz = self.data.qpos[3:7]
        theta = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
        return np.array([x, y, theta])
