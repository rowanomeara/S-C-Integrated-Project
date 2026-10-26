import time
import threading
import numpy as np
import cv2

# Hardware Abstraction Layer (HAL) for physical TurtleBot3
# Auto-detects ROS 2 (rclpy) or ROS 1 (rospy)

class TurtleBotReal:
    def __init__(self, camera_id=0, dt=0.05, namespace=""):
        self.dt = dt
        self.v_odom = 0.0
        self.w_odom = 0.0
        self.ranges = np.ones(360, dtype=float) * 5.0
        self.ros_frame = None
        self.lock = threading.Lock()
        self.ros_version = None

        prefix = f"/{namespace.strip('/')}" if namespace else ""
        cmd_topic = f"{prefix}/cmd_vel"
        scan_topic = f"{prefix}/scan"
        odom_topic = f"{prefix}/odom"

        self.cap = None

        # 1. Connect via ROS 2 (rclpy) or ROS 1 (rospy)
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import LaserScan

            rclpy.init(args=None)
            self.ros_version = "ros2"
            self.node = Node("turtlebot_hal")

            self.cmd_pub = self.node.create_publisher(Twist, cmd_topic, 10)
            self.node.create_subscription(LaserScan, scan_topic, self._ros2_scan_cb, qos_profile_sensor_data)
            self.node.create_subscription(Odometry, odom_topic, self._ros2_odom_cb, qos_profile_sensor_data)

            # Camera subscriptions: Best-effort accepts both Best Effort and Reliable publishers
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

            # Controlled executor prevents shutdown core dumps
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

        # Allow initial sensor messages to arrive
        time.sleep(0.5)

    def _ros2_scan_cb(self, msg):
        self._process_scan(msg.ranges, msg.angle_min)

    def _ros1_scan_cb(self, msg):
        self._process_scan(msg.ranges, msg.angle_min)

    def _process_scan(self, raw_ranges, angle_min):
        arr = np.array(raw_ranges, dtype=float)
        # Clean invalid measurements (inf, nan, <=0.05m noise) -> 5.0m max
        arr[~np.isfinite(arr)] = 5.0
        arr[arr < 0.05] = 5.0

        # Resample to exactly 360 beams if needed
        if len(arr) != 360:
            indices = np.linspace(0, len(arr) - 1, 360).astype(int)
            arr = arr[indices]

        # In TurtleBotSim, index 0 is -pi (rear) and 180 is 0 (front).
        # LDS-01 standard scan starts at 0 (front) if angle_min == 0.
        # np.roll shifts front (index 0) to index 180 to match simulation coordinate conventions.
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
        if self.ros_version == "ros2":
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
        with self.lock:
            return self.ranges.copy()

    def get_camera_frame(self):
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
