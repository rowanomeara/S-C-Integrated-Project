"""Perception: Multiprocessing Camera Window Viewer.

Isolates OpenCV GUI rendering inside a dedicated subprocess using the 'spawn'
context, preventing Cocoa/AppKit thread collisions and window freezes on macOS.
"""

import multiprocessing as mp
import cv2


def _display_loop(queue, title):
    """Worker process loop consuming frames from the IPC queue and displaying them."""
    while True:
        try:
            frame = queue.get()
            # Poison pill pattern: None signals window closure
            if frame is None:
                break
            cv2.imshow(title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        except Exception:
            break
    cv2.destroyAllWindows()


class CameraWindow:
    """Non-blocking OpenCV camera window running in an independent OS process."""

    def __init__(self, title="TurtleBot3 - Camera Feed"):
        """Spawn the background display process.

        Parameters:
          title: OS window title string.
        """
        self.title = title
        # 'spawn' start method creates a fresh Python interpreter process (mandatory on macOS for GUI safety)
        ctx = mp.get_context("spawn")
        # Bounded queue of maxsize 2 drops stale frames when display rendering lags
        self.queue = ctx.Queue(maxsize=2)
        self.proc = ctx.Process(target=_display_loop, args=(self.queue, self.title), daemon=True)
        self.proc.start()

    def show(self, frame):
        """Send a new video frame to the display queue without blocking execution.

        Parameters:
          frame: np.ndarray image frame (BGR or RGB).

        Returns:
          bool: True if the display process remains running.
        """
        if frame is not None and not self.queue.full():
            try:
                self.queue.put_nowait(frame)
            except Exception:
                pass
        return self.proc.is_alive()

    def close(self):
        """Send termination poison pill and join worker process."""
        try:
            self.queue.put_nowait(None)
            self.proc.join(timeout=0.5)
        except Exception:
            pass
        cv2.destroyAllWindows()
