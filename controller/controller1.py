# from posixpath import sep
import sys
import os
import logging
import configparser
import numpy as np
import cv2
import yaml
import glob
import traceback
import threading
from threading import Timer
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
from playsound import playsound


# If we want to run controller.py as main, then we need set import as below:
currentdir = os.path.dirname(os.path.abspath(__file__))  # noqa :E402
parentdir = os.path.dirname(currentdir)  # noqa :E402
sys.path.append(parentdir)  # noqa :E402
from miscellaneous.myFormatConvertor import FormatConvertor  # noqa :E402
from image_process.opencv_processing import ImageProcessing_Thread  # noqa :E402
from image_process.capture_pcb_image import Capture_PCB_Image_Thread  # noqa :E402
from view.ui_main import Ui_main_frame  # noqa :E402
from miscellaneous.ziwenLog.myLogConfig import ConfigMyLog  # noqa : E402
from config_manage.config_manage_controller import ConfigManageMainWindow


class MainWindow(QtWidgets.QMainWindow):
    """This is mainframe UI thread"""

    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.my_logger = logging.getLogger('logger_main')
        self.image_capturer = None
        self.pcb_checker = ImageProcessing_Thread()
        self._load_default_config()
        self._init_ui()
        self._set_shortcut_key()
        self._connect_signal_to_slot_function()

    def _load_default_config(self):
        self.config = configparser.ConfigParser()
        config_file_path = os.path.join(parentdir, 'config', 'config.ini')
        self.my_logger.debug('Load default config settings from :')
        self.my_logger.debug(config_file_path)
        self.config.read(config_file_path)
        self.my_logger.debug('Config file was loaded successfully.')

    def _load_mlfb(self):
        try:
            folder_contains_yaml = os.path.join(parentdir, 'config')
            yaml_files_full_path = glob.glob(folder_contains_yaml+'/*.yaml')
            mlfb_names = [x.split(os.path.sep)[-1][:-5] for x in yaml_files_full_path]
            if len(mlfb_names) == 0:
                raise Exception
            self.my_logger.debug('Load MLFB from yaml files done.')
            return mlfb_names
        except:
            self.my_logger.error('Failed to parse MLFB from yaml files.')
            os._exit(1)

    def _init_ui(self):
        self.my_logger.debug('Start to initialize main frame UI.')
        self.ui = Ui_main_frame()
        self.ui.setupUi(self)
        self.ui.label_video.setPixmap(QPixmap(""))
        self.ui.label_comment.setText('Welcome to use this tool.')
        # make 'offline radiobutton' as default.
        self.ui.radioButton_offline_image.setChecked(True)
        # MLFBs
        for mlfb in self._load_mlfb():
            self.ui.comboBox_MLFB_config.addItem(mlfb)
        # load config of default MLFB
        self._load_config_from_yaml()
        raw_images_pathlist = os.listdir(os.path.join(parentdir, 'raw_images'))
        # Add paths to offline video path select comboBox
        for path in raw_images_pathlist:
            if path.endswith(".mp4"):
                self.ui.comboBox_offline_video_select.addItem(f'raw_images/{path}')
        # Add paths to offline image path select comboBox
        for path in raw_images_pathlist:
            if path.endswith(".png") or path.endswith(".jpg") or path.endswith(".gif"):
                self.ui.comboBox_MLFB_ipath_select.addItem(f'raw_images/{path}')
        # Video size
        self.ui.textEdit_video_size.setText(
            str(self.config['CAMERA']['Video_Width']) + 'x' + str(self.config['CAMERA']['Video_Height']))
        # timeout after special gesture was detected
        self.ui.textEdit_response_time.setText(str(self.config['WORKER']['Response_Time']))
        # minimum size of pcb area
        self.ui.textEdit_min_size_pcb_area.setText(str(self.config_of_current_MLFB['minimum_size_of_pcb_area']))
        # golden sample Image path
        self.ui.textEdit_golden_sample_path.setText(str(self.config_of_current_MLFB['golden_sample_path']))

    def _set_shortcut_key(self):
        self.shortcut_idle_cv = QtWidgets.QShortcut(QKeySequence(Qt.Key_Escape), self)
        self.shortcut_oneshot_by_space = QtWidgets.QShortcut(QKeySequence(Qt.Key_Space), self)
        self.shortcut_oneshot_by_enter = QtWidgets.QShortcut(QKeySequence(Qt.Key_Enter), self)
        self.shortcut_oneshot_by_return = QtWidgets.QShortcut(QKeySequence(Qt.Key_Return), self)

    def _connect_signal_to_slot_function(self):
        self.shortcut_idle_cv.activated.connect(self.idle)
        self.shortcut_oneshot_by_space.activated.connect(self.capture_image_by_button_click)
        self.shortcut_oneshot_by_enter.activated.connect(self.capture_image_by_button_click)
        self.shortcut_oneshot_by_return.activated.connect(self.capture_image_by_button_click)
        self.ui.comboBox_MLFB_config.currentIndexChanged.connect(
            self._reload_config_from_yaml)  # switch to another MLFB
        #程序入口
        self.ui.pushButton_start.clicked.connect(self.capture_image_by_special_gesture)
        self.ui.pushButton_oneshot.clicked.connect(self.capture_image_by_button_click)
        self.ui.pushButton_idle.clicked.connect(self.idle)
        self.pcb_checker.checking_done.connect(self.on_check_image_done)
        self.ui.pushButton_config.clicked.connect(self.open_config_window)
        self.ui.label_result.mouseDoubleClickEvent = self.open_log_file
        self.ui.pushButton_viewlog.clicked.connect(self.open_log_dir)
        self.ui.pushButton_takepicture.clicked.connect(self.take_picture)

    @pyqtSlot()
    def _load_config_from_yaml(self):
        current_MLFB = self.ui.comboBox_MLFB_config.currentText()
        path_of_yaml = os.path.join(parentdir, 'config', current_MLFB+'.yaml')
        with open(path_of_yaml, 'r') as f:
            self.config_of_current_MLFB = yaml.load(f, Loader=yaml.SafeLoader)

    @pyqtSlot()
    def _reload_config_from_yaml(self):
        self._load_config_from_yaml()
        self.ui.textEdit_min_size_pcb_area.setText(str(self.config_of_current_MLFB['minimum_size_of_pcb_area']))
        self.ui.textEdit_golden_sample_path.setText(str(self.config_of_current_MLFB['golden_sample_path']))

    @ pyqtSlot()
    def capture_image_by_button_click(self):
        if self.image_capturer:
            self.image_capturer.stop_capturing()
            self.image_capturer.capture_image_by_button()

    @ pyqtSlot()
    def capture_image_by_special_gesture(self):
        """Use another thread to capture video/image."""
        self.my_logger.debug('Start capture images thru video/image/camera.')
        x_size = int(self.config['CAMERA']['Video_Width'])
        y_size = int(self.config['CAMERA']['Video_Height'])
        move_hand_response_time = float(self.config['WORKER']['Response_Time'])
        offline_video_path = self.ui.comboBox_offline_video_select.currentText()
        offline_image_path = self.ui.comboBox_MLFB_ipath_select.currentText()
        self.ui.label_result.setText('RUNNING')
        self.ui.label_result.setStyleSheet("background-color: rgb(0, 255, 0);")
        self.ui.label_comment.setText('Capturing image for analysis.')
        self.MAX_RETRY_COUNT = 2
        try:
            if self.ui.radioButton_offline_image.isChecked():  # offline static image mode
                self.process_image_to_check_THT_process(cv2.imread(offline_image_path))
            else:  # streaming (video or camera)
                if self.ui.radioButton_online_cam.isChecked():  # online camera mode
                    current_mode = 'online'
                    current_path = None
                elif self.ui.radioButton_offline_video.isChecked():  # offline video mode
                    current_mode = 'offline_video'
                    current_path = offline_video_path
                # prevent creating multiple threads which will increase memory.
                #第二次以后点击start按钮时，防止启动多个相机实例(单例模式?)
                if isinstance(self.image_capturer, Capture_PCB_Image_Thread):
                    self.my_logger.debug('The object of Capture_PCB_Image_Thread exised.')
                    self.image_capturer.stop_capturing()  
                    self.image_capturer.reload_camera(x_size, y_size, mode=current_mode, path=current_path)
                    #新建线程
                    self.image_capturer.resume_capture_image_by_special_gesture()
                    #通过标志位重启循环，而不是新建线程
                    # self.image_capturer.stop_capture_flag = False
                    self.image_capturer.capture_a_frame_with_gesture_done.connect(self.update_ui_image_frame)
                # First time create image_capturer object.(分摄像机捕获帧和缺陷检测model两个线程)
                else:
                    self.image_capturer = Capture_PCB_Image_Thread(
                        x_size, y_size, mode=current_mode, path=current_path)
                    self.image_capturer.set_timeout_for_move_hand(move_hand_response_time)
                    #逐帧更新ui
                    self.image_capturer.capture_a_frame_with_gesture_done.connect(self.update_ui_image_frame)
                    #model线程在相机线程捕获到静态帧时启动(相机模式捕获有两种,一是手势识别，二是manual take picture)
                    self.image_capturer.capture_a_frame_after_timeout_done.connect(
                        self.process_image_to_check_THT_process)
                    # self.image_capturer.run()
                    self.image_capturer.start()
        except Exception:
            self.my_logger.error(traceback.format_exc())
            self.ui.label_comment.setText('Failed to process video/image. Please check log.')

    @ pyqtSlot()
    def stop_capturing(self):
        try:
            self.image_capturer.stop_capturing()
            self.image_capturer.capture_a_frame_with_gesture_done.disconnect()
            self.image_capturer.capture_a_frame_after_timeout_done.disconnect()
        except:
            pass
        self.ui.label_video.setPixmap(QPixmap(""))
        self.ui.label_comment.setText('Stop capture')
        self.ui.label_result.setStyleSheet('background-color:rgb(255,255,0);')
        self.ui.label_result.setText('STOP')

    @ pyqtSlot()
    def idle(self):
        try:
            self.image_capturer.stop_capturing()
        except:
            pass
        # need some time to stop UI updating
        QtCore.QTimer.singleShot(500, lambda: {self.ui.label_video.setPixmap(QPixmap(""))})
        self.ui.label_comment.setText('Set to idle.')
        self.ui.label_result.setStyleSheet('background-color:rgb(255,255,0);')
        self.ui.label_result.setText('IDLE')

    @ pyqtSlot(np.ndarray)
    def process_image_to_check_THT_process(self, frame_ndarray):
        if self.image_capturer and self.image_capturer.receivers(self.image_capturer.capture_a_frame_with_gesture_done) > 0:
            #如果捕获到待进行检测得帧，需要断开实时刷新ui得pysingal(注释掉未见影响,发送静态帧时已经暂停了摄像头，不会有下一帧更新ui？)
            self.image_capturer.capture_a_frame_with_gesture_done.disconnect()
            pass
        #加载图片和配置到类变量
        self.pcb_checker.set_check_info(frame_ndarray, self.config_of_current_MLFB)
        #启动pcb_checker线程，自定义槽信号self.pcb_checker.checking_done发送(str,img)信号
        self.pcb_checker.start()
        # for index in threading.enumerate():
        #     print('model stage Thread',index)
        print("当前线程数{}".format(threading.active_count()))

    @ pyqtSlot(str, np.ndarray)
    def on_check_image_done(self, check_result, frame_ndarray):
        # print the check_image on GUI first.
        frame_QPixmap = FormatConvertor.convert_to_QPixmap(frame_ndarray)
        self.ui.label_video.setPixmap(frame_QPixmap.scaled(
            self.ui.label_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.ui.label_result.setText(str.capitalize(check_result))
        if check_result == 'pass':
            self._take_actions_if_check_pass()
        if check_result == 'fail':
            self._take_actions_if_check_fail()
        if check_result == 'error':
            self._take_actions_if_check_error()

    @ pyqtSlot()
    def open_config_window(self):
        config_manage_window = ConfigManageMainWindow()
        config_manage_window.show()


    def _take_actions_if_check_pass(self):
        def resume_capture_later():
            self.ui.label_result.setText('RUNNING')
            self.ui.label_comment.setText('Capturing image for analysis.')
            self.image_capturer.capture_a_frame_with_gesture_done.connect(self.update_ui_image_frame)
            self.image_capturer.resume_capture_image_by_special_gesture()
        self.my_logger.info('Check pass.')
        self.ui.label_result.setText('PASS')
        self.ui.label_comment.setText('Passed')
        self.ui.label_result.setStyleSheet('background-color: rgb(0, 255, 0);')
        # Continue to check when online camera is selected
        if self.ui.radioButton_online_cam.isChecked():
            QtCore.QTimer.singleShot(1000, resume_capture_later)
        playsound_timer = Timer(0.05, self.playsound_after_check, args=['pass'])
        playsound_timer.start()
        self.MAX_RETRY_COUNT = 2

    def _take_actions_if_check_fail(self):
        self.ui.label_result.setText('FAIL')
        self.ui.label_comment.setText('Failed')
        self.ui.label_result.setStyleSheet('background-color: rgb(255, 0, 0);')
        self.ui.label_comment.setText('Check failed. Please refer to the red box.')
        playsound_timer = Timer(0.05, self.playsound_after_check, args=['fail'])
        playsound_timer.start()
        self.MAX_RETRY_COUNT = 2

    def _take_actions_if_check_error(self):
        if self.MAX_RETRY_COUNT <= 0:
            self.ui.label_result.setStyleSheet('background-color: rgb(255, 0, 0);')
            self.ui.label_comment.setText('Image processing return error . Please check log.')
            playsound_timer = Timer(0.05, self.playsound_after_check, args=['error'])
            playsound_timer.start()
            self.MAX_RETRY_COUNT = 2
        else:
            self.MAX_RETRY_COUNT -= 1
            retry_timer = Timer(0.05, self.image_capturer.capture_image_by_button)
            retry_timer.start()

    @ pyqtSlot(np.ndarray)
    def update_ui_image_frame(self, frame_ndarray):
        frame_QPixmap = FormatConvertor.convert_to_QPixmap(frame_ndarray)
        self.ui.label_video.setPixmap(frame_QPixmap.scaled(
            self.ui.label_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def open_log_file(self, event):
        logs_dir = os.path.join(parentdir, 'logs')
        logfile_dir = os.path.join(logs_dir, 'THT_Check.log')
        os.system(f"start /b notepad \".{logfile_dir}\"")

    def open_log_dir(self):
        logs_dir = os.path.join(parentdir, 'logs')
        # os.system(f"start \"{logs_dir}\"")
        os.startfile(logs_dir)
        
    def take_picture(self):
        logs_dir = os.path.join(parentdir, 'logs')
        if self.image_capturer:
            self.image_capturer.my_cam.next_frame()
            captured_img = self.image_capturer.my_cam.frame_ndarray
            cv2.imwrite(os.path.join(logs_dir, 'lastCaptured.png'), captured_img)
            QMessageBox.information(self, 'Image Taken', 'Successfully took an image. The image is saved to /logs/lastCaptured.png.', QMessageBox.Yes)
        else:
            QMessageBox.warning(self, 'No Capturer', 'No capturer found. Please start camera and try again.', QMessageBox.Yes)

    def playsound_after_check(self, check_status):
        sound_dir = os.path.join(parentdir, 'assets')
        sound_pass_dir = os.path.join(sound_dir, f'check_{check_status}.mp3')
        playsound(sound_pass_dir)

    # This function is called when app exits. Stop camera when user quits the app to prevent camera running in background.
    @ pyqtSlot()
    def closeEvent(self, event):
        print("About to quit the app. Closing camera streaming.")
        try:
            self.image_capturer.stop_capturing()
        except:
            pass
        event.accept()


if __name__ == '__main__':
    my_logger = ConfigMyLog('logger_main', logFileName='THT_Check.log', logFolderPath=r'D:\Projects\THT_Assembly\source\logs', withFolder=False,
                            maxBytes=30*1024, backupCount=3).give_me_a_logger()
    my_logger.info('THT assembly component checking program started...')
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    application.show()
    sys.exit(app.exec())
