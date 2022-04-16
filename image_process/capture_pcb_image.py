import cv2
import logging
import numpy as np
import mediapipe as mp
import threading
from PyQt5.QtCore import pyqtSignal, QThread
from threading import Timer
from video_capture.webcam import WebCam  # noqa :E402


class Capture_PCB_Image_Thread(QThread):
    capture_a_frame_with_gesture_done = pyqtSignal(np.ndarray)  # real time frame from camera.
    capture_a_frame_after_timeout_done = pyqtSignal(np.ndarray)  # manual take picture from camera.

    def __init__(self, x_size=1280, y_size=720, mode='online', path=None):
        super().__init__()
        self.my_logger = logging.getLogger('logger_main')
        self.mode = mode
        self.path = path
        self.my_cam = WebCam(x_size, y_size, mode, path)
        self.stop_capture_flag = False
        self.detect_speical_gesture = False
        self.allow_emit_static_image_after_timeout = True
        self.move_hand_response_time = 1.5
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5)
        # Restrict the number of frames that mediapipe recognizes hands, to decrease CPU load.
        self.mediapipe_capture_counter = 0

    def reload_camera(self, x_size, y_size, mode, path):
        self.my_cam.close()
        self.my_cam.x_size = x_size
        self.my_cam.y_size = y_size
        self.my_cam._select_capturer(mode, path)

    def getStructuredLandmarks(self, landmarks):
        structuredLandmarks = []
        for i in range(21):
            structuredLandmarks.append({'x': landmarks[i].x, 'y': landmarks[i].y})
        return structuredLandmarks

    def parseFiveFingers(self, thumb, index, middle, ring, little):
        """parse result based on 5 fingers opening/closing
        O (not 0) means 'open'. C means 'close'.

        Returns:
            [str]: parse result.
        """
        connect_str = '_'
        combileFingerState = connect_str.join((thumb, index, middle, ring, little))
        parse_dict = {
            'O_O_O_O_O': 'FIVE', 'C_O_O_O_O': 'FOUR', 'O_O_O_C_C': 'THREE',
            'C_O_O_C_C': 'TWO', 'C_O_C_C_C': 'ONE', 'C_C_C_C_C': 'ZERO',
            'O_C_C_C_O': 'CAPTURE'
        }
        if combileFingerState in parse_dict.keys():
            gesture = parse_dict[combileFingerState]
        else:
            gesture = 'UNKNOW'
        return gesture

    def recognizeHandGesture(self, landmarks):
        """ recognzie gesture based on 21 key points of palm.
        Refer to google's mediapipe website.
        """
        thumbState = 'UNKNOW'
        indexFingerState = 'UNKNOW'
        middleFingerState = 'UNKNOW'
        ringFingerState = 'UNKNOW'
        littleFingerState = 'UNKNOW'
        recognizedHandGesture = None
        thumb_offset = 0.01
        other_offset = 0.005

        pseudoFixKeyPoint = landmarks[2]['x']
        if (landmarks[3]['x'] < pseudoFixKeyPoint and landmarks[4]['x'] < landmarks[3]['x']):
            thumbState = 'C'  # close
        elif (pseudoFixKeyPoint < landmarks[3]['x'] - thumb_offset and landmarks[3]['x'] < landmarks[4]['x'] - thumb_offset):
            thumbState = 'O'  # open

        pseudoFixKeyPoint = landmarks[6]['y']
        if (landmarks[7]['y'] < pseudoFixKeyPoint and landmarks[8]['y'] < landmarks[7]['y']):
            indexFingerState = 'O'
        elif (pseudoFixKeyPoint < landmarks[7]['y'] - other_offset and landmarks[7]['y'] < landmarks[8]['y'] - other_offset):
            indexFingerState = 'C'  # it's capitalized c. not 0.

        pseudoFixKeyPoint = landmarks[10]['y']
        if (landmarks[11]['y'] < pseudoFixKeyPoint and landmarks[12]['y'] < landmarks[11]['y']):
            middleFingerState = 'O'
        elif (pseudoFixKeyPoint < landmarks[11]['y'] - other_offset and landmarks[11]['y'] < landmarks[12]['y'] - other_offset):
            middleFingerState = 'C'

        pseudoFixKeyPoint = landmarks[14]['y']
        if (landmarks[15]['y'] < pseudoFixKeyPoint and landmarks[16]['y'] < landmarks[15]['y']):
            ringFingerState = 'O'
        elif (pseudoFixKeyPoint < landmarks[15]['y'] - other_offset and landmarks[15]['y'] < landmarks[16]['y'] - other_offset):
            ringFingerState = 'C'

        pseudoFixKeyPoint = landmarks[18]['y']
        if (landmarks[19]['y'] < pseudoFixKeyPoint and landmarks[20]['y'] < landmarks[19]['y']):
            littleFingerState = 'O'
        elif (pseudoFixKeyPoint < landmarks[19]['y'] - other_offset and landmarks[19]['y'] < landmarks[20]['y'] - other_offset):
            littleFingerState = 'C'

        recognizedHandGesture = self.parseFiveFingers(
            thumbState, indexFingerState, middleFingerState, ringFingerState, littleFingerState)
        return recognizedHandGesture

    # run() for (QThread). start() to call it. no parameter!
    def run(self):
        """
        provide camera's real time image to UI. 
        provide static image if special gesture was detected.
        """
        def emit_static_image_after_timeout():
            print('triggered.')
            self.capture_a_frame_after_timeout_done.emit(self.image_with_gesture)
            self.allow_emit_static_image_after_timeout = True
            self.stop_capture_flag = True

        # print('threading start')
        # print('Thread id : %d'%threading.currentThread().ident)
        # for index in threading.enumerate():
        #     print('capture picture stage Thread',index)
        self.my_logger.debug('Call capture image by sepcial gesture start.')
        self.stop_capture_flag = False
        while not self.stop_capture_flag:
            self.detect_speical_gesture = False
            self.my_cam.next_frame()
            self.mediapipe_capture_counter += 1
            image = self.my_cam.frame_ndarray
            # image = cv2.cvtColor(self.my_cam.frame_ndarray, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            if self.mediapipe_capture_counter == 1:
                self.mediapipe_capture_counter = 0
                image = cv2.cvtColor(self.my_cam.frame_ndarray, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                if results.multi_hand_landmarks:
                    # Note! handedness detection is not reliable
                    # handedness_info = results.multi_handedness[0].__str__()
                    # hand_label = handedness_info.split('\n')[3].split("\"")[1]
                    for hand_landmarks in results.multi_hand_landmarks:
                        # hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]  is hand_landmarks.landmark[0].
                        # refer to https://google.github.io/mediapipe/solutions/hands.html
                        recognizedResult = self.recognizeHandGesture(self.getStructuredLandmarks(
                            hand_landmarks.landmark))
                        mp.solutions.drawing_utils.draw_landmarks(
                            image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        if recognizedResult == 'CAPTURE':
                            self.detect_speical_gesture = True
            self.image_with_gesture = image
            # must add waitKey for highGUI.Send self.image_with_gesture to controller for updating UI.
            cv2.waitKey(1)
            self.capture_a_frame_with_gesture_done.emit(self.image_with_gesture)
            # if the required gesture is detcted, send static image to controller for defect detecting and updating UI.
            if self.detect_speical_gesture and self.allow_emit_static_image_after_timeout:
                self.allow_emit_static_image_after_timeout = False  # prevent emitting multiple times in this loop
                self.my_logger.debug('trigger Qtimer : emit_static_image_after_timeout')
                # another thread to tigger timeout.
                #检测到特殊手势后需要延时检测
                t1 = Timer(self.move_hand_response_time, emit_static_image_after_timeout)
                t1.start()
        self.my_logger.debug('Stop capture flag set to true.I am out of loop.')
        # self.my_cam.close()
        # self.hands.close()

    def set_timeout_for_move_hand(self, timeout):
        self.move_hand_response_time = timeout

    def resume_capture_image_by_special_gesture(self):
        # Give system some time to quit the while-loop. And restart the while-loop.
        # Set stop_capture_flag = True in other thread in advance.
        t2 = Timer(1.0, self.run)
        t2.start()

    def capture_image_by_button(self):
        """Manally take an image.
        """
        #self.stop_capture_flag = False
        self.my_cam.next_frame()
        self.capture_a_frame_after_timeout_done.emit(self.my_cam.frame_ndarray)

    def stop_capturing(self):
        self.stop_capture_flag = True
