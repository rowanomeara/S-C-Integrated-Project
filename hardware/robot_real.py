"""Hardware: Physical TurtleBot3 Hardware Abstraction Layer (HAL).

Connects to the physical TurtleBot3 Burger onboard computer via the rosbridge
WebSocket protocol (port 9090), providing an identical API to TurtleBotSim:
1. Subscribes to wheel encoder odometry on /odom.
2. Queries 360-beam LDS-01 laser scans on /scan.
3. Decodes compressed camera frames on /camera/image_raw/compressed.
4. Publishes twist velocity commands to /cmd_vel.
"""

import time
import threading
import numpy as np
import cv2


class TurtleBotReal:
    """Hardware interface communicating with TurtleBot3 via rosbridge WebSocket."""

    def __init__(self, host=None, port=9090, camera_id=0, dt=0.05, namespace=""):
        """Initialize rosbridge connection, subscribers, and thread locks.

        Parameters:
          host: IP address of the TurtleBot (e.g. '192.168.1.206').
          port: rosbridge WebSocket port (default: 9090).
          camera_id: Camera index fallback.
          dt: Sampling period in seconds (0.05 s = 20 Hz).
          namespace: Optional ROS namespace prefix.
        """
        self.dt = dt
        self.v_odom = 0.0
        self.w_odom = 0.0
        # Initialize LiDAR buffer with 360 beams at 5.0 m clearance
        self.ranges = np.ones(360, dtype=float) * 5.0
        # Mutex protecting concurrent read/write access to sensor buffers
        self.lock = threading.Lock()
        self.bridge = None

        prefix = f"/{namespace.strip('/')}" if namespace else ""
        odom_topic = f"{prefix}/odom"

        if host:
            try:
                from rosbridge_api import RosbridgeRobot
                import roslibpy

                self.bridge = RosbridgeRobot(host, port=port)
                self.bridge.setup()

                # Asynchronous callback invoked when new odometry messages arrive
                def _on_odom(msg):
                    try:
                        self.v_odom = float(msg["twist"]["twist"]["linear"]["x"])
                        self.w_odom = float(msg["twist"]["twist"]["angular"]["z"])
                    except Exception:
                        pass

                self.odom_sub = roslibpy.Topic(self.bridge._client, odom_topic, "nav_msgs/msg/Odometry")
                self.odom_sub.subscribe(_on_odom)
                print(f"[HW] Connected to TurtleBot via rosbridge WebSocket at {host}:{port}.")
                time.sleep(0.5)
            except Exception as e:
                print(f"[HW] rosbridge connection to {host}:{port} failed ({e}).")

    def _process_scan(self, raw_ranges, angle_min):
        """Clean, resample, and orient raw laser scan readings.

        1. Replaces non-finite readings (NaN, inf) with 5.0 m (clear space).
        2. Replaces near-blind zone readings (< 0.05 m) with 5.0 m to reject chassis reflections.
        3. Resamples beam count to exactly 360 elements.
        4. Rolls array by 180 indices if angle_min starts near 0 rad, aligning beam 180 to forward.
        """
        arr = np.array(raw_ranges, dtype=float)
        arr[~np.isfinite(arr)] = 5.0
        arr[arr < 0.05] = 5.0

        if len(arr) != 360:
            indices = np.linspace(0, len(arr) - 1, 360).astype(int)
            arr = arr[indices]

        # Sensor frame alignment: shifts forward-pointing beams to index 180
        if abs(angle_min) < 0.1:
            arr = np.roll(arr, 180)

        with self.lock:
            self.ranges = arr

    def set_velocity(self, v, omega):
        """Publish linear and angular velocity commands to /cmd_vel."""
        if self.bridge is not None:
            self.bridge.send_command(linear=float(v), angular=float(omega))

    def get_odometry(self):
        """Return latest linear forward speed (m/s) and angular yaw rate (rad/s)."""
        return self.v_odom, self.w_odom

    def get_lidar_scan(self):
        """Query laser scanner and return cleaned 360-beam range array in meters."""
        if self.bridge is not None:
            scan = self.bridge.query_laser_scan(timeout=0)
            if scan is not None:
                self._process_scan(scan.ranges, scan.angle_min)
        with self.lock:
            return self.ranges.copy()

    def get_camera_frame(self):
        """Query latest camera frame and convert BGR color channels to RGB."""
        if self.bridge is not None:
            frame_bgr = self.bridge.query_image(timeout=0)
            if frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def stop(self):
        """Halt motors and cleanly terminate network sockets and threads."""
        try:
            self.set_velocity(0.0, 0.0)
        except Exception:
            pass
        if hasattr(self, "odom_sub") and self.odom_sub is not None:
            try:
                self.odom_sub.unsubscribe()
            except Exception:
                pass
        if self.bridge is not None:
            try:
                self.bridge.stop()
                self.bridge.cleanup()
            except Exception:
                pass
