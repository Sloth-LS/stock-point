import cv2


def is_picam(source):
    return isinstance(source, str) and source.startswith("picam")


def picam_number(source):
    parts = source.split(":")
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return 0


class PiCamera:
    def __init__(self, source, width=640, height=480):
        self.camera = None
        self.width = width
        self.height = height
        try:
            from picamera2 import Picamera2
        except ImportError:
            print("picamera2 mangler. Installer den med: sudo apt install python3-picamera2")
            return
        try:
            camera = Picamera2(picam_number(source))
            config = camera.create_video_configuration(main={"size": (width, height), "format": "RGB888"})
            camera.configure(config)
            camera.start()
        except Exception as e:
            print(f"Kan ikke starte Pi-kameraet {source}: {e}")
            return
        self.camera = camera

    def isOpened(self):
        return self.camera is not None

    def read(self):
        if self.camera is None:
            return False, None
        try:
            frame = self.camera.capture_array()
        except Exception:
            return False, None
        return True, frame

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0

    def release(self):
        if self.camera is None:
            return
        try:
            self.camera.stop()
            self.camera.close()
        except Exception:
            pass
        self.camera = None
