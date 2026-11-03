import time
import threading
import numpy as np
import cv2

# Hardware Abstraction Layer (HAL) for physical TurtleBot3
# Connects directly to the robot via rosbridge WebSocket API (turtlebot_rosbridge / roslibpy).


class TurtleBotReal:
    def __init__(self, host=None, port=9090, camera_id=0, dt=0.05, namespace=""):
        self.dt = dt
        self.v_odom = 0.0
        self.w_odom = 0.0
        self.ranges = np.ones(360, dtype=float) * 5.0
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
        arr = np.array(raw_ranges, dtype=float)
        arr[~np.isfinite(arr)] = 5.0
        arr[arr < 0.05] = 5.0

        if len(arr) != 360:
            indices = np.linspace(0, len(arr) - 1, 360).astype(int)
            arr = arr[indices]

        if abs(angle_min) < 0.1:
            arr = np.roll(arr, 180)

        with self.lock:
            self.ranges = arr

    def set_velocity(self, v, omega):
        if self.bridge is not None:
            self.bridge.send_command(linear=float(v), angular=float(omega))

    def get_odometry(self):
        return self.v_odom, self.w_odom

    def get_lidar_scan(self):
        if self.bridge is not None:
            scan = self.bridge.query_laser_scan(timeout=0)
            if scan is not None:
                self._process_scan(scan.ranges, scan.angle_min)
        with self.lock:
            return self.ranges.copy()

    def get_camera_frame(self):
        if self.bridge is not None:
            frame_bgr = self.bridge.query_image(timeout=0)
            if frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def stop(self):
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
