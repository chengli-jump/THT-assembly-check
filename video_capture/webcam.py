import cv2
from PyQt5.QtWidgets import QWidget
import logging


class WebCam(QWidget):
    """Config/control web camera. 
    """

    def __init__(self, x_size, y_size, mode='online', path=None):
        super().__init__()
        self.my_logger = logging.getLogger('logger_main')
        self.mirror_needed = False
        self.frame_ndarray = None
        self.fps = None
        self.mode = None
        self.rval_video = False
        self.x_size = x_size
        self.y_size = y_size
        self._select_capturer(mode, path)

    def _select_capturer(self, mode, path):
        self.mode = mode
        if self.mode == 'online':
            self.my_logger.debug('Real time online camera is selected.')
            self.videocapture = cv2.VideoCapture(1)
            if self.videocapture.read()[1] is None:
                self.videocapture = cv2.VideoCapture(0)
            self._init_cam()
        elif self.mode == 'offline_video':
            self.my_logger.debug('Offline video mode is selected.')
            self.videocapture = cv2.VideoCapture(path)
            self._init_cam()
        else:
            self.my_logger.error('Select wrong mode.')
            raise Exception

    def _init_cam(self):
        if self.videocapture.isOpened():
            self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.x_size)
            self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.y_size)
            self.fps = self.videocapture.get(cv2.CAP_PROP_FPS) % 100

    def next_frame(self):
        self.rval_video, capture_frame_raw = self.videocapture.read()
        if self.rval_video:
            if self.mirror_needed:
                capture_frame = cv2.flip(capture_frame_raw, -1)
                #capture_frame = cv2.flip(capture_frame_raw, 1)
            else:
                capture_frame = capture_frame_raw
            self.frame_ndarray = capture_frame
        else:
            if self.mode == 'online':
                self.my_logger.debug('Videocapture.read() return false. stop capture.')
                raise Exception
            # If it's offline mode, it means the vidoe is finished.
            pass

    def close(self):
        self.videocapture.release()
