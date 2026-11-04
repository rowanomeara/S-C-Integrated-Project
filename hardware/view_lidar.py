"""Hardware: Live 2D LiDAR Polar Radar Visualizer.

Streams sensor_msgs/LaserScan from the TurtleBot3 over rosbridge to render
a real-time top-down polar radar display using OpenCV:
1. Filters out non-finite (NaN, inf) returns and hardware blind spots.
2. Projects polar laser coordinates (r, theta) into Cartesian screen pixels.
3. Renders concentric distance rings (0.25, 0.50, 0.75, 1.00 of display range).
4. Displays robot origin, heading orientation arrow, and clearance telemetry.
"""

import argparse
import math
import cv2
import numpy as np
from rosbridge_api import LaserScan, RosbridgeRobot


def render_laser_scan(
    scan: LaserScan, image_size: int = 700, display_range: float | None = None
) -> np.ndarray:
    """Project laser scan points onto a 2D top-down BGR radar canvas.

    Parameters:
      scan: LaserScan data object containing ranges and beam angles.
      image_size: Canvas width and height in pixels (default: 700).
      display_range: Maximum radar distance in meters (default: 3.5 m).

    Returns:
      np.ndarray: uint8 BGR image canvas.
    """
    if image_size < 100:
        raise ValueError("image_size must be at least 100 pixels")

    # Filter valid laser returns within sensor range limits
    valid = np.isfinite(scan.ranges)
    valid &= scan.ranges >= scan.range_min
    valid &= scan.ranges <= scan.range_max

    # Set default display range to 3.5 m (LDS-01 operational range)
    if display_range is None:
        display_range = scan.range_max
        if not math.isfinite(display_range) and np.any(valid):
            display_range = float(np.max(scan.ranges[valid]))
    if (
        display_range is None
        or not math.isfinite(display_range)
        or display_range <= 0
    ):
        display_range = 3.5

    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    centre = image_size // 2
    scale = (image_size * 0.45) / display_range  # Meter to pixel scale factor

    # Draw concentric distance rings (25%, 50%, 75%, 100% of display range)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        radius = round(display_range * fraction * scale)
        cv2.circle(canvas, (centre, centre), radius, (50, 50, 50), 1)

    visible = valid & (scan.ranges <= display_range)
    ranges = scan.ranges[visible]
    angles = scan.angles[visible]

    # Polar to Cartesian screen coordinates:
    # Forward (+x in robot frame) maps toward the top of the canvas (-y in pixel space)
    pixel_x = np.rint(centre - ranges * np.sin(angles) * scale).astype(int)
    pixel_y = np.rint(centre - ranges * np.cos(angles) * scale).astype(int)

    # Discard points outside image boundary
    inside = (
        (pixel_x >= 0)
        & (pixel_x < image_size)
        & (pixel_y >= 0)
        & (pixel_y < image_size)
    )
    # Paint laser obstacle returns in yellow
    canvas[pixel_y[inside], pixel_x[inside]] = (0, 255, 255)

    # Draw green circle at robot center
    cv2.circle(canvas, (centre, centre), 6, (0, 255, 0), -1)
    # Draw directional arrow pointing forward
    cv2.arrowedLine(
        canvas,
        (centre, centre),
        (centre, centre - 40),
        (0, 255, 0),
        2,
        tipLength=0.25,
    )

    # Render telemetry HUD text
    min_dist = float(np.min(scan.ranges[valid])) if np.any(valid) else 0.0
    cv2.putText(
        canvas,
        f"range: {display_range:.1f} m | points: {int(np.count_nonzero(visible))} | min_dist: {min_dist:.2f} m",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    """Connect to TurtleBot3 laser scan stream and display live radar window."""
    parser = argparse.ArgumentParser(description="Live 2D LiDAR Scan Viewer")
    parser.add_argument("host", nargs="?", default="192.168.1.206", help="Robot IP address (e.g. 192.168.1.209)")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--topic", default="/scan", help="LaserScan topic")
    parser.add_argument("--size", type=int, default=700, help="Window size")
    parser.add_argument("--range", dest="display_range", type=float, default=3.5, help="Max display range (m)")
    args = parser.parse_args()

    print(f"\n[LIDAR] Connecting to TurtleBot at {args.host}:{args.port}...")
    with RosbridgeRobot(args.host, port=args.port, laser_scan_topic=args.topic) as robot:
        print("[LIDAR] Laser scan active. Press 'q' or Esc in the window to quit.\n")
        try:
            while robot.is_connected:
                scan = robot.query_laser_scan(timeout=1.0, wait_for_new=True)
                if scan is None:
                    continue
                image = render_laser_scan(scan, args.size, args.display_range)
                cv2.imshow("TurtleBot3 - Live LiDAR Scan", image)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            cv2.destroyAllWindows()
            print("\n[LIDAR] Closed.")


if __name__ == "__main__":
    main()
