"""Control robot velocity from the keyboard using rosbridge_api."""

import argparse
import threading
import time
from rosbridge_api import RosbridgeRobot


def teleop(robot: RosbridgeRobot, publish_rate: float = 10.0) -> None:
    """Control the robot using W/S/A/D, Space to stop, and Escape to exit."""
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError('Teleop requires pynput: pip install pynput') from exc

    command_lock = threading.Lock()
    command = {"linear": 0.0, "angular": 0.0}

    def on_press(key) -> None:
        should_stop = False
        with command_lock:
            try:
                if key.char == "w":
                    command["linear"] = 0.12
                elif key.char == "s":
                    command["linear"] = -0.12
                elif key.char == "a":
                    command["angular"] = 0.8
                elif key.char == "d":
                    command["angular"] = -0.8
            except AttributeError:
                if key == keyboard.Key.space:
                    command["linear"] = 0.0
                    command["angular"] = 0.0
                    should_stop = True
        if should_stop:
            robot.stop()

    def on_release(key):
        with command_lock:
            try:
                if key.char in ("w", "s"):
                    command["linear"] = 0.0
                elif key.char in ("a", "d"):
                    command["angular"] = 0.0
            except AttributeError:
                pass
        if key == keyboard.Key.esc:
            return False
        return None

    print("\n" + "=" * 50)
    print(" TURTLEBOT3 KEYBOARD TELEOP ACTIVE")
    print("=" * 50)
    print(" Controls:")
    print("   [W] / [S] : Forward / Backward (0.12 m/s)")
    print("   [A] / [D] : Turn Left / Right  (0.80 rad/s)")
    print("   [Space]   : Emergency Stop")
    print("   [Esc]     : Exit teleop")
    print("=" * 50 + "\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    try:
        while listener.running and robot.is_connected:
            with command_lock:
                linear = command["linear"]
                angular = command["angular"]
            robot.send_command(linear, angular)
            time.sleep(1.0 / publish_rate)
    finally:
        listener.stop()
        listener.join()
        if robot.is_connected:
            robot.stop()
        print("\n[TELEOP] Robot stopped and teleop finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyboard Teleoperation for TurtleBot3")
    parser.add_argument("host", nargs="?", default="192.168.1.206", help="Robot IP address (e.g. 192.168.1.209)")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--rate", type=float, default=10.0, help="Publish rate (Hz)")
    args = parser.parse_args()

    print(f"\n[TELEOP] Connecting to TurtleBot at {args.host}:{args.port}...")
    with RosbridgeRobot(args.host, port=args.port) as robot:
        teleop(robot, publish_rate=args.rate)


if __name__ == "__main__":
    main()
