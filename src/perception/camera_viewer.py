
import multiprocessing as mp
import cv2

def _display_loop(queue):
    while True:
        try:
            frame = queue.get()
            if frame is None:
                break
            cv2.imshow('TurtleBot3 - Onboard Camera POV', frame)
            cv2.waitKey(1)
        except Exception:
            break
    cv2.destroyAllWindows()

class CameraWindow:
    def __init__(self):
        ctx = mp.get_context('spawn')
        self.queue = ctx.Queue(maxsize=2)
        self.proc = ctx.Process(target=_display_loop, args=(self.queue,), daemon=True)
        self.proc.start()

    def show(self, frame):
        if not self.queue.full():
            try:
                self.queue.put_nowait(frame)
            except Exception:
                pass

    def close(self):
        try:
            self.queue.put_nowait(None)
            self.proc.join(timeout=0.5)
        except Exception:
            pass
