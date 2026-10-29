import multiprocessing as mp
import cv2


def _display_loop(queue, title):
    while True:
        try:
            frame = queue.get()
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
    def __init__(self, title="TurtleBot3 - Camera Feed"):
        self.title = title
        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue(maxsize=2)
        self.proc = ctx.Process(target=_display_loop, args=(self.queue, self.title), daemon=True)
        self.proc.start()

    def show(self, frame):
        if frame is not None and not self.queue.full():
            try:
                self.queue.put_nowait(frame)
            except Exception:
                pass
        return self.proc.is_alive()

    def close(self):
        try:
            self.queue.put_nowait(None)
            self.proc.join(timeout=0.5)
        except Exception:
            pass
        cv2.destroyAllWindows()
