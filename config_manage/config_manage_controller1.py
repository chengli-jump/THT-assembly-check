# This Python file uses the following encoding: utf-8

import os
import sys
currentdir = os.path.dirname(os.path.abspath(__file__))  # noqa :E402
parentdir = os.path.dirname(currentdir)  # noqa :E402
sys.path.append(parentdir)  # noqa :E402

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidgetItem
from PyQt5.QtGui import QPixmap, QKeySequence
from PyQt5.QtCore import QSize, Qt, QRect
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QShortcut
import cv2
import yaml
import numpy as np
from os.path import basename
from config_manage.config_manage_ui import Ui_Dialog
from miscellaneous.myFormatConvertor import FormatConvertor
from image_process import check_methods




class ConfigManageMainWindow(QMainWindow, Ui_Dialog):
    
    flag_drawing = False
    flag_dragging_singlebox = False
    flag_dragging_multibox = False
    flag_selecting_multibox = False
    flag_selected_multibox = False
    dragging_index = -1
    start_pos = None
    end_pos = None
    opened_file_directory = None
    mouse_pos = (0, 0)
    check_box_corner1 = (0, 0)
    check_box_corner2 = (0, 0)
    label_size = (1000, 1400)
    label_y_offset = 0
    golden_sample_img = None
    config_object = None
    current_checkbox_index = 0

    def __init__(self, parent=None):
        super(ConfigManageMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.init_ui()

        self.button_open_config.clicked.connect(self.open_config_file)
        self.button_open_sample_image.clicked.connect(self.open_sample_image)
        self.button_create_checkbox.clicked.connect(self.create_checkbox)
        self.button_draw_checkbox.clicked.connect(self.draw_checkbox)
        self.button_delete_checkbox.clicked.connect(self.delete_checkbox)
        self.button_save_changes.clicked.connect(self.save_changes)
        self.button_create_config.clicked.connect(self.create_config)
        self.button_save_config.clicked.connect(self.save_file)
        self.button_save_config_as.clicked.connect(self.save_file_as)
        self.listWidget.currentRowChanged.connect(self.select_check_box)
        self.comboBox_threshold_method.currentIndexChanged.connect(self.check_threshold_method)

        self.target_image_frame.mousePressEvent = self.frame_mousePress
        self.target_image_frame.mouseDoubleClickEvent = self.frame_mouseDoubleClick
        self.target_image_frame.mouseMoveEvent = self.frame_mouseMove
        self.target_image_frame.mouseReleaseEvent = self.frame_mouseRelease

        self.shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        self.shortcut.activated.connect(self.copy_checkbox)

    def init_ui(self):
        self.comboBox_threshold_method.clear()
        self.comboBox_threshold_method.addItem("by_percent")
        for method in check_methods.threshold_method_functions:
            self.comboBox_threshold_method.addItem(method)
        self.comboBox_threshold_method.setCurrentIndex(0)

    def update_list_widget(self):
        # Delete all items in existing list
        for i in range(self.listWidget.count()):
            self.listWidget.takeItem(0)
        # Update checkboxes to listview
        box_list_item = QListWidgetItem("显示全部", self.listWidget)
        box_list_item.setSizeHint(QSize(20,50))
        box_list_item.setTextAlignment(Qt.AlignCenter)
        for box in self.config_object["check_box"]:
            box_list_item = QListWidgetItem(box["name"], self.listWidget)
            box_list_item.setSizeHint(QSize(20,50)) 
            box_list_item.setTextAlignment(Qt.AlignCenter)
        self.draw_all_checkboxes()
        self.listWidget.setCurrentRow(0)


    def open_config_file(self):
        # Open a dialog for user to open a config file
        file_directory, _ = QFileDialog.getOpenFileName(self, "选择文件", r"config", "configs (*.yaml)")
        if file_directory and file_directory[-5:] == '.yaml':
            self.label_config_filename.setText(file_directory)
            with open(file_directory, 'r', encoding='utf-8') as f:
                self.opened_file_directory = file_directory
                self.config_object = yaml.load(f, Loader=yaml.FullLoader)

                # Replace Golden_sample_picture with picture in config file
                if self.config_object["golden_sample_path"]:
                    golden_sample_path = self.config_object["golden_sample_path"]
                    self.golden_sample_img = cv2.imread(golden_sample_path)
                    self.label_y_offset = (self.label_size[0] - self.golden_sample_img.shape[0]) // 2
                self.text_sample_image_path.setText(self.config_object["golden_sample_path"])

                if self.config_object["minimum_size_of_pcb_area"]:
                    self.text_MIN_AREA.setText(str(self.config_object["minimum_size_of_pcb_area"]))

                if self.config_object["pcb_area_hue_range"]:
                    self.text_pcb_hue_range.setText(str(self.config_object["pcb_area_hue_range"]))

                if self.config_object["pcb_area_saturation_range"]:
                    self.text_pcb_saturation_range.setText(str(self.config_object["pcb_area_saturation_range"]))

                if self.config_object["pcb_area_value_range"]:
                    self.text_pcb_value_range.setText(str(self.config_object["pcb_area_value_range"]))

                if self.config_object["binary_val"]:
                    self.text_IMG_BINARY_VAL.setText(str(self.config_object["binary_val"]))

                if self.config_object["saturation_value"]:
                    self.text_saturation_value.setText(str(self.config_object["saturation_value"]))

                self.update_list_widget()

                self.button_save_config.setEnabled(True)
                self.button_save_config_as.setEnabled(True)
                self.button_open_sample_image.setEnabled(True)

    def open_sample_image(self):
        from image_process.opencv_processing import ImageProcessing_Thread

        file_directory, _ = QFileDialog.getOpenFileName(self, "选择图片", r"raw_images\\", 'Image files (*.jpg *.gif *.png *.jpeg)')
        if file_directory:
            # Read Image from file
            image_raw = cv2.imread(file_directory)
            # Find PCB from image and generate golden sample image
            image_thread = ImageProcessing_Thread()
            if self.text_MIN_AREA.toPlainText():
                self.config_object['minimum_size_of_pcb_area'] = int(self.text_MIN_AREA.toPlainText())
            image_thread.set_check_info(image_raw, self.config_object)
            # golden_sample_img = image_thread._find_pcb_area_of_image(image_raw, is_golden_sample=True)

            pcb_width = image_raw.shape[1]
            pcb_heigth = image_raw.shape[0]
            canvas_width, canvas_height = int(pcb_width*1.1), int(pcb_heigth*1.1)
            canvas = 0 * np.ones(shape=[canvas_height, canvas_width, 3], dtype=np.uint8)
            canvas[0:pcb_heigth, 0:pcb_width, :] = image_raw[0:pcb_heigth, 0:pcb_width, :]
            golden_sample_img = canvas
            self.golden_sample_img = golden_sample_img
            # Save golden sample image to folder
            base_name = basename(file_directory)
            cv2.imwrite(f"golden_images/golden_sample_{base_name}", self.golden_sample_img)
            self.text_sample_image_path.setText(f"golden_images/golden_sample_{base_name}")
            # Update golden_sample_path in config object
            self.config_object["golden_sample_path"] = f"golden_images/golden_sample_{base_name}"
            self.label_y_offset = (self.label_size[0] - self.golden_sample_img.shape[0]) // 2
            self.update_picture_label(self.target_image_frame, self.golden_sample_img)

            self.button_save_config.setEnabled(True)
            self.button_save_config_as.setEnabled(True)
            self.button_create_checkbox.setEnabled(True)


    def draw_all_checkboxes(self):
        img_with_boxes = np.copy(self.golden_sample_img)
        for box in self.config_object["check_box"]:
            cv2.rectangle(img_with_boxes, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 255, 0), 2)
        self.update_picture_label(self.target_image_frame, img_with_boxes)

    def select_check_box(self, item_index):
        if item_index < 0:
            item_index = 0
        self.current_checkbox_index = item_index
        # When selecting all checkboxes, draw all rectangles on image
        if item_index == 0:
            self.text_name.setText("N/A")
            self.text_threshold_value.setText("N/A")
            self.text_box_upper_left.setText("N/A")
            self.text_box_lower_right.setText("N/A")
            self.draw_all_checkboxes()

            # Set buttons 
            self.button_create_checkbox.setEnabled(True)
            self.button_save_changes.setEnabled(False)
            self.button_draw_checkbox.setEnabled(False)
            self.button_delete_checkbox.setEnabled(False)
            
        # When selecting one checkbox, update information
        else:
            if len(self.config_object["check_box"]) == 0:
                return
            self.current_checkbox_index -= 1
            current_checkbox = self.config_object["check_box"][self.current_checkbox_index]
            self.text_name.setText(current_checkbox["name"])
            self.comboBox_threshold_method.setCurrentIndex(self.comboBox_threshold_method.findText(str(current_checkbox["threshold_method"])))
            self.text_threshold_value.setText(str(current_checkbox["threshold_value"]))
            self.text_box_upper_left.setText(str(current_checkbox["box_upper_left"]))
            self.text_box_lower_right.setText(str(current_checkbox["box_lower_right"]))
            self.comboBox_check_channel.setCurrentIndex(self.comboBox_check_channel.findText(str(current_checkbox["check_channel"])))
            img_with_boxes = np.copy(self.golden_sample_img)
            cv2.rectangle(img_with_boxes, parse_tuple(current_checkbox["box_upper_left"]), parse_tuple(current_checkbox["box_lower_right"]), (0, 255, 0), 2)
            self.update_picture_label(self.target_image_frame, img_with_boxes, save_display='golden')

            # Set buttons 
            self.button_create_checkbox.setEnabled(True)
            self.button_save_changes.setEnabled(True)
            self.button_draw_checkbox.setEnabled(True)
            self.button_delete_checkbox.setEnabled(True)


    def create_checkbox(self):
        if self.config_object is None:
            return
        if self.listWidget.count() == 0:
            box_list_item = QListWidgetItem("显示全部", self.listWidget)
            box_list_item.setSizeHint(QSize(20,50))
            box_list_item.setTextAlignment(Qt.AlignCenter)
        new_checkbox = {
            "name": "新建元件" + str(len(self.config_object["check_box"]) + 1), 
            "box_upper_left": "(0, 0)",
            "box_lower_right": "(100, 100)",
            "check_channel": "green",
            "threshold_method": "by_percent",
            "threshold_value": 20,
        }
        box_list_item = QListWidgetItem(new_checkbox["name"], self.listWidget)
        box_list_item.setSizeHint(QSize(20,50))
        box_list_item.setTextAlignment(Qt.AlignCenter)
        self.config_object["check_box"].append(new_checkbox)
        self.listWidget.setCurrentRow(len(self.config_object["check_box"]))
        self.flag_drawing = True
        self.draw_checkbox()

    def delete_checkbox(self):
        reply = QMessageBox.question(self, '删除检测区块', '确定要删除选中的检测区块吗?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.config_object["check_box"].pop(self.current_checkbox_index)
            self.update_list_widget()
        else:
            return

    def save_changes(self):
        self.config_object["check_box"][self.current_checkbox_index]["name"] = self.text_name.toPlainText()
        self.config_object["check_box"][self.current_checkbox_index]["threshold_method"] = self.comboBox_threshold_method.currentText()
        self.config_object["check_box"][self.current_checkbox_index]["threshold_value"] = int(self.text_threshold_value.toPlainText())
        self.config_object["check_box"][self.current_checkbox_index]["check_channel"] = self.comboBox_check_channel.currentText()
        self.config_object["check_box"][self.current_checkbox_index]["box_upper_left"] = self.text_box_upper_left.toPlainText()
        self.config_object["check_box"][self.current_checkbox_index]["box_lower_right"] = self.text_box_lower_right.toPlainText()
        self.listWidget.item(self.current_checkbox_index + 1).setText(self.text_name.toPlainText())

    def draw_checkbox(self):
        if self.config_object is None:
            return
        self.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))
        self.flag_drawing = True

    def frame_mouseMove(self, event):
        if self.golden_sample_img is None:
            return
        mouse_pos = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
        drawing_img = np.copy(self.displaying_image)
        if self.flag_drawing:
            cv2.rectangle(drawing_img, self.check_box_corner1, mouse_pos, (0, 255, 0), 2)
            self.update_picture_label(self.target_image_frame, drawing_img, save_display=False)
        
        elif self.flag_selecting_multibox:
            cv2.rectangle(drawing_img, self.start_pos, mouse_pos, (188, 188, 188), 1)
            self.update_picture_label(self.target_image_frame, drawing_img, save_display=False)

        elif self.flag_dragging_singlebox:
            pos_delta = (mouse_pos[0] - self.start_pos[0], mouse_pos[1] - self.start_pos[1])
            for box in self.config_object["check_box"]:
                # Draw a red box indicating the new copy
                if box == self.config_object["check_box"][self.dragging_index]: 
                    box_upper_left = parse_tuple(box["box_upper_left"])
                    box_lower_right = parse_tuple(box["box_lower_right"])
                    box_upper_left = (box_upper_left[0] + pos_delta[0], box_upper_left[1] + pos_delta[1])
                    box_lower_right = (box_lower_right[0] + pos_delta[0], box_lower_right[1] + pos_delta[1])
                    cv2.rectangle(drawing_img, box_upper_left, box_lower_right, (0, 30, 235), 2)
            self.update_picture_label(self.target_image_frame, drawing_img, save_display=False)

        elif self.flag_selected_multibox:
            pos_delta = (mouse_pos[0] - self.start_pos[0], mouse_pos[1] - self.start_pos[1])
            for box in self.config_object["check_box"]:
                if box in self.selected_boxes:
                    continue
                cv2.rectangle(drawing_img, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 255, 0), 2)
            # Draw all selected boxes after moving
            for box in self.selected_boxes:
                box_upper_left = parse_tuple(box["box_upper_left"])
                box_lower_right = parse_tuple(box["box_lower_right"])
                box_upper_left = (box_upper_left[0] + pos_delta[0], box_upper_left[1] + pos_delta[1])
                box_lower_right = (box_lower_right[0] + pos_delta[0], box_lower_right[1] + pos_delta[1])
                cv2.rectangle(drawing_img, box_upper_left, box_lower_right, (0, 30, 235), 2)
            self.update_picture_label(self.target_image_frame, drawing_img, save_display=False)


    def frame_mousePress(self,event):
        if self.golden_sample_img is None:
            return
        if self.flag_drawing:
            self.check_box_corner1 = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
        # elif: self.flag_selecting_multibox
        elif self.flag_selected_multibox:
            self.start_pos = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
        elif self.listWidget.currentRow() == 0:
            self.start_pos = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
            self.flag_selecting_multibox = True


    def frame_mouseDoubleClick(self, event):
        if self.golden_sample_img is None:
            return
        # Only allow to double click when user is not drawing new box
        if self.flag_drawing:
            return
        # Only allow to double click when displaying all boxes
        if self.listWidget.currentRow() != 0:
            return

        def find_box_by_click_pos(boxes, pos):
            for i, box in enumerate(boxes):
                pos_upper_left = box['box_upper_left'][1:-1].split(',')
                pos_lower_right = box ['box_lower_right'][1:-1].split(',')
                if int(pos_upper_left[0]) < pos[0] < int(pos_lower_right[0]) and int(pos_upper_left[1]) < pos[1] < int(pos_lower_right[1]):
                    print(f'found! Box is {box["name"]}')
                    return i
            return -1

        self.start_pos = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
        if not self.flag_selected_multibox:
            box_index = find_box_by_click_pos(self.config_object['check_box'], self.start_pos)        
            if box_index > -1:
                self.flag_dragging_singlebox = True
                self.dragging_index = box_index

                drawing_img = np.copy(self.golden_sample_img)
                for box in self.config_object["check_box"]:
                    if box == self.config_object["check_box"][box_index]:
                        continue
                    cv2.rectangle(drawing_img, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 255, 0), 2)\
                # Draw all box except for the box we are moving, and save the image as displaying image
                self.displaying_image = drawing_img.copy()
                box = self.config_object["check_box"][box_index]
                # Then, draw the selected box in another color
                cv2.rectangle(drawing_img, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 30, 235), 2)
                self.update_picture_label(self.target_image_frame, drawing_img, save_display=False)

    def frame_mouseRelease(self, event):
        if self.golden_sample_img is None:
            return
        self.check_box_corner2 = self.restrain_points_in_image(event.x(), event.y() - self.label_y_offset)
        self.end_pos = self.check_box_corner2

        if self.flag_drawing:
            check_box_corner_upper_left, check_box_corner_lower_right = [0, 0], [0, 0]
            check_box_corner_upper_left[0]  = min(self.check_box_corner1[0], self.check_box_corner2[0])
            check_box_corner_upper_left[1]  = min(self.check_box_corner1[1], self.check_box_corner2[1])
            check_box_corner_lower_right[0] = max(self.check_box_corner1[0], self.check_box_corner2[0])
            check_box_corner_lower_right[1] = max(self.check_box_corner1[1], self.check_box_corner2[1])
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            self.flag_drawing = False
            self.text_box_upper_left.setText(str(tuple(check_box_corner_upper_left)))
            self.text_box_lower_right.setText(str(tuple(check_box_corner_lower_right)))
            return
        
        if self.flag_dragging_singlebox:
            self.flag_dragging_singlebox = False
            self.move_singlebox_copy()
            return
        if self.flag_selecting_multibox:
            self.select_boxes(self.start_pos, self.end_pos)
            return
        if self.flag_selected_multibox:
            self.flag_selected_multibox = False
            reply = QMessageBox.question(self, 'Copy Multiple Checkboxes', 'Are you sure to copy selected boxes to this position?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.create_multibox_copy()
            else:
                self.draw_all_checkboxes()
                return

    def select_boxes(self, start_pos, end_pos):
        self.selected_boxes = []
        img_with_boxes = np.copy(self.golden_sample_img)
        for box in self.config_object["check_box"]:
            cv2.rectangle(img_with_boxes, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 255, 0), 2)
            upper_left = parse_tuple(box["box_upper_left"])
            lower_right = parse_tuple(box["box_lower_right"])
            select_box_corners = [min(start_pos[0], end_pos[0]), min(start_pos[1], end_pos[1]), max(start_pos[0], end_pos[0]), max(start_pos[1], end_pos[1])]
            if (upper_left[0] > select_box_corners[0] and upper_left[1] > select_box_corners[1] and lower_right[0] < select_box_corners[2] and lower_right[1] < select_box_corners[3]):
                self.selected_boxes.append(box)
                cv2.rectangle(img_with_boxes, parse_tuple(box["box_upper_left"]), parse_tuple(box["box_lower_right"]), (0, 0, 255), 2)

        self.update_picture_label(self.target_image_frame, img_with_boxes)
        self.flag_selecting_multibox = False

        # Set flag_selected_multibox to True only when some box is selected    
        self.flag_selected_multibox = True if len(self.selected_boxes) > 0 else False

    def move_singlebox_copy(self):
        pos_delta = (self.end_pos[0] - self.start_pos[0], self.end_pos[1] - self.start_pos[1])
        dragging_box = self.config_object['check_box'][self.dragging_index]
        box_upper_left = parse_tuple(dragging_box["box_upper_left"])
        box_lower_right = parse_tuple(dragging_box["box_lower_right"])
        box_upper_left = (box_upper_left[0] + pos_delta[0], box_upper_left[1] + pos_delta[1])
        box_lower_right = (box_lower_right[0] + pos_delta[0], box_lower_right[1] + pos_delta[1])
        dragging_box['box_upper_left'] = str(box_upper_left)
        dragging_box['box_lower_right'] = str(box_lower_right)
        self.update_list_widget()

    def create_multibox_copy(self):
        pos_delta = (self.end_pos[0] - self.start_pos[0], self.end_pos[1] - self.start_pos[1])
        for box in self.selected_boxes:
            box_upper_left = parse_tuple(box["box_upper_left"])
            box_lower_right = parse_tuple(box["box_lower_right"])
            box_upper_left = (box_upper_left[0] + pos_delta[0], box_upper_left[1] + pos_delta[1])
            box_lower_right = (box_lower_right[0] + pos_delta[0], box_lower_right[1] + pos_delta[1])
            new_box_copy = {
                'box_upper_left'  : str(box_upper_left),
                'box_lower_right' : str(box_lower_right),
                'check_channel'   : box['check_channel'],
                'name'            : box['name'] + "_multicopy",
                'threshold_method': box['threshold_method'],
                'threshold_value' : box['threshold_value'],
            }
            self.config_object['check_box'].append(new_box_copy)
        self.update_list_widget()


    # Restrain cursor positions inside the image frame
    def restrain_points_in_image(self, pos_x, pos_y):
        mouse_pos = [pos_x, pos_y]
        if mouse_pos[0] < 0:
            mouse_pos[0] = 0
        if mouse_pos[1] < 0:
            mouse_pos[1] = 0
        if mouse_pos[0] >= self.golden_sample_img.shape[1]:
            mouse_pos[0] = self.golden_sample_img.shape[1] - 1
        if mouse_pos[1] >= self.golden_sample_img.shape[0]:
            mouse_pos[1] = self.golden_sample_img.shape[0] - 1
        mouse_pos = tuple(mouse_pos)
        return mouse_pos

    # Update target image label with image data
    def update_picture_label(self, label_obj, image, save_display='current'):
        frame_QPixmap = FormatConvertor.convert_to_QPixmap(image)
        label_obj.setPixmap(frame_QPixmap.scaled(
            image.shape[1], image.shape[0], QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        if save_display == 'current':
            self.displaying_image = image
        elif save_display == 'golden':
            self.displaying_image = self.golden_sample_img

    def create_config(self):
        self.config_object = {
            "golden_sample_path": "",
            "minimum_size_of_pcb_area": 400000,
            "pcb_area_hue_range": "45-95",
            "pcb_area_saturation_range": "10-150",
            "pcb_area_value_range": "0-255",
            "check_box": [],
            "binary_val": 60,
            "saturation_value": 0,
        }
        self.golden_sample_img = None
        # Delete all items in existing list
        for i in range(self.listWidget.count()):
            self.listWidget.takeItem(0)
        # Update checkboxes to listview
        self.target_image_frame.setText("                                                     Target Image")
        self.label_config_filename.setText("新建config文件 - 未保存")

        # Update default config values
        self.text_MIN_AREA.setText(str(self.config_object["minimum_size_of_pcb_area"]))
        self.text_IMG_BINARY_VAL.setText(str(self.config_object["binary_val"]))
        self.text_saturation_value.setText(str(self.config_object["saturation_value"]))
        self.text_pcb_hue_range.setText(str(self.config_object["pcb_area_hue_range"]))
        self.text_pcb_saturation_range.setText(str(self.config_object["pcb_area_saturation_range"]))
        self.text_pcb_value_range.setText(str(self.config_object["pcb_area_value_range"]))

        # Reset text areas
        self.text_name.setText(" ")
        self.text_threshold_value.setText(" ")
        self.text_box_upper_left.setText(" ")
        self.text_box_lower_right.setText(" ")

        # Set buttons 
        self.button_create_checkbox.setEnabled(False)
        self.button_save_changes.setEnabled(False)
        self.button_draw_checkbox.setEnabled(False)
        self.button_delete_checkbox.setEnabled(False)
        self.button_open_sample_image.setEnabled(True)

    def save_file(self):
        file_directory = self.opened_file_directory
        if self.text_MIN_AREA.toPlainText():
            self.config_object['minimum_size_of_pcb_area'] = int(self.text_MIN_AREA.toPlainText())
        if self.text_IMG_BINARY_VAL.toPlainText():
            self.config_object['binary_val'] = int(self.text_IMG_BINARY_VAL.toPlainText())
        if self.text_saturation_value.toPlainText():
            self.config_object['saturation_value'] = int(self.text_saturation_value.toPlainText())
        if self.text_pcb_hue_range.toPlainText():
            self.config_object["pcb_area_hue_range"] = self.text_pcb_hue_range.toPlainText()
        if self.text_pcb_saturation_range.toPlainText():
            self.config_object["pcb_area_saturation_range"] = self.text_pcb_saturation_range.toPlainText()
        if self.text_pcb_value_range.toPlainText():
            self.config_object["pcb_area_value_range"] = self.text_pcb_value_range.toPlainText()

        if file_directory and file_directory[-5:] == '.yaml':
            self.label_config_filename.setText(file_directory)
            with open(file_directory, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_object, f, allow_unicode=True)
            f.close()

    def save_file_as(self):
        file_directory, _ = QFileDialog.getSaveFileName(self, "保存文件", r"C:\Users\ziwen\Desktop\config_manage\config_manage", "configs (*.yaml)")
        if self.text_MIN_AREA.toPlainText():
            self.config_object['minimum_size_of_pcb_area'] = int(self.text_MIN_AREA.toPlainText())
        if self.text_IMG_BINARY_VAL.toPlainText():
            self.config_object['binary_val'] = int(self.text_IMG_BINARY_VAL.toPlainText())
        if self.text_saturation_value.toPlainText():
            self.config_object['saturation_value'] = int(self.text_saturation_value.toPlainText())
        if file_directory and file_directory[-5:] == '.yaml':
            self.label_config_filename.setText(file_directory)
            with open(file_directory, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_object, f, allow_unicode=True)
            f.close()

    def copy_checkbox(self):
        print("ctrl+c pressed.")

    def check_threshold_method(self):
        if self.comboBox_threshold_method.currentText() == "by_percent":
            self.comboBox_check_channel.setEnabled(True)
        else:
            self.comboBox_check_channel.setEnabled(False)

def parse_tuple(str_tuple):
    return tuple(map(int, str_tuple[1:-1].split(',')))


if __name__ == "__main__":
    app = QApplication([])
    window = ConfigManageMainWindow()
    window.show()
    sys.exit(app.exec_())
