import cv2


class CameraWindow:
    def __init__(self, title="TurtleBot3 - Camera Feed"):
        self.title = title

    def show(self, frame):
        if frame is not None:
            cv2.imshow(self.title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return False
        return True

    def close(self):
        cv2.destroyAllWindows()
