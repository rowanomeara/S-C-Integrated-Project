import time
import threading
import numpy as np
import cv2

# Hardware Abstraction Layer (HAL) for physical TurtleBot3
# Supports:
#  1. rosbridge WebSocket API (running directly from student laptop via UTS-RI/turtlebot_rosbridge)
#  2. Onboard ROS 2 (rclpy) / ROS 1 (rospy) if executed directly on the robot
#  3. Local V4L2 device fallback


class TurtleBotReal:
    def __init__(self, host=None, port=9090, camera_id=0, dt=0.05, namespace=""):
        self.dt = dt
        self.v_odom = 0.0
        self.w_odom = 0.0
        self.ranges = np.ones(360, dtype=float) * 5.0
        self.ros_frame = None
        self.lock = threading.Lock()
        self.backend = None
        self.bridge = None
        self.cap = None
        self.ros_version = None

        prefix = f"/{namespace.strip('/')}" if namespace else ""
        cmd_topic = f"{prefix}/cmd_vel"
        scan_topic = f"{prefix}/scan"
        odom_topic = f"{prefix}/odom"

        # 1. Connect via rosbridge_api if host provided
        if host:
            try:
                from rosbridge_api import RosbridgeRobot
                import roslibpy

                self.bridge = RosbridgeRobot(host, port=port)
                self.bridge.setup()
                self.backend = "rosbridge"

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
                return
            except Exception as e:
                print(f"[HW] rosbridge connection to {host}:{port} failed ({e}). Checking onboard ROS...")

        # 2. Connect via ROS 2 (rclpy) or ROS 1 (rospy)
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan

            rclpy.init(args=None)
            self.ros_version = "ros2"
            self.node = Node("turtlebot_hal")

            self.cmd_pub = self.node.create_publisher(Twist, cmd_topic, 10)
            self.node.create_subscription(LaserScan, scan_topic, self._ros2_scan_cb, qos_profile_sensor_data)
            self.node.create_subscription(Odometry, odom_topic, self._ros2_odom_cb, qos_profile_sensor_data)

            try:
                from sensor_msgs.msg import CompressedImage
                self.node.create_subscription(CompressedImage, "/camera/image_raw/compressed", self._ros2_compressed_img_cb, qos_profile_sensor_data)
                self.node.create_subscription(CompressedImage, "/image_raw/compressed", self._ros2_compressed_img_cb, qos_profile_sensor_data)
            except Exception as e:
                print(f"[HW] CompressedImage subscription note: {e}")
            try:
                from sensor_msgs.msg import Image
                self.node.create_subscription(Image, "/camera/image_raw", self._ros2_raw_img_cb, qos_profile_sensor_data)
                self.node.create_subscription(Image, "/image_raw", self._ros2_raw_img_cb, qos_profile_sensor_data)
            except Exception:
                pass

            self.executor = rclpy.executors.SingleThreadedExecutor()
            self.executor.add_node(self.node)
            self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
            self.spin_thread.start()
            print("[HW] Connected via ROS 2 (rclpy). Subscribed to camera topics.")

        except ImportError:
            try:
                import rospy
                from geometry_msgs.msg import Twist
                from nav_msgs.msg import Odometry
                from sensor_msgs.msg import LaserScan

                rospy.init_node("turtlebot_hal", anonymous=True, disable_signals=True)
                self.ros_version = "ros1"

                self.cmd_pub = rospy.Publisher(cmd_topic, Twist, queue_size=10)
                rospy.Subscriber(scan_topic, LaserScan, self._ros1_scan_cb)
                rospy.Subscriber(odom_topic, Odometry, self._ros1_odom_cb)
                print("[HW] Connected via ROS 1 (rospy).")

            except ImportError:
                print("[HW] No ROS environment detected. Falling back to local V4L2 device.")
                self.cap = cv2.VideoCapture(camera_id)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"[HW] Local camera index {camera_id} opened.")

        time.sleep(0.5)

    def _ros2_scan_cb(self, msg):
        self._process_scan(msg.ranges, msg.angle_min)

    def _ros1_scan_cb(self, msg):
        self._process_scan(msg.ranges, msg.angle_min)

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

    def _ros2_odom_cb(self, msg):
        self.v_odom = float(msg.twist.twist.linear.x)
        self.w_odom = float(msg.twist.twist.angular.z)

    def _ros1_odom_cb(self, msg):
        self.v_odom = float(msg.twist.twist.linear.x)
        self.w_odom = float(msg.twist.twist.angular.z)

    def _ros2_compressed_img_cb(self, msg):
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is not None:
            with self.lock:
                self.ros_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _ros2_raw_img_cb(self, msg):
        h, w = msg.height, msg.width
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        if len(arr) == h * w * 3:
            frame = arr.reshape((h, w, 3))
            if "bgr" in msg.encoding.lower():
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.ros_frame = frame

    def set_velocity(self, v, omega):
        if self.backend == "rosbridge" and self.bridge is not None:
            self.bridge.send_command(linear=float(v), angular=float(omega))
        elif self.ros_version == "ros2":
            from geometry_msgs.msg import Twist
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(omega)
            self.cmd_pub.publish(msg)
        elif self.ros_version == "ros1":
            from geometry_msgs.msg import Twist
            msg = Twist()
            msg.linear.x = float(v)
            msg.angular.z = float(omega)
            self.cmd_pub.publish(msg)

    def get_odometry(self):
        return self.v_odom, self.w_odom

    def get_lidar_scan(self):
        if self.backend == "rosbridge" and self.bridge is not None:
            scan = self.bridge.query_laser_scan(timeout=0)
            if scan is not None:
                self._process_scan(scan.ranges, scan.angle_min)
            with self.lock:
                return self.ranges.copy()
        with self.lock:
            return self.ranges.copy()

    def get_camera_frame(self):
        if self.backend == "rosbridge" and self.bridge is not None:
            frame_bgr = self.bridge.query_image(timeout=0)
            if frame_bgr is not None:
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return None
        with self.lock:
            if self.ros_frame is not None:
                return self.ros_frame.copy()
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def stop(self):
        try:
            self.set_velocity(0.0, 0.0)
        except Exception:
            pass
        if self.backend == "rosbridge" and self.bridge is not None:
            try:
                self.bridge.stop()
                self.bridge.cleanup()
            except Exception:
                pass
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()
        if self.ros_version == "ros2":
            try:
                import rclpy
                if hasattr(self, 'executor') and self.executor is not None:
                    self.executor.shutdown()
                if hasattr(self, 'spin_thread') and self.spin_thread.is_alive():
                    self.spin_thread.join(timeout=0.3)
                if hasattr(self, 'node') and self.node is not None:
                    self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
