import sys
from PyQt5.QtWidgets import QApplication, QAction, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, \
    QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QGridLayout, QCheckBox, QGroupBox, QLineEdit, \
    QProgressBar, QRadioButton, QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import QObject, Qt, QTimer, QThread, pyqtSignal, pyqtSlot
# from PyQt5.QtCore import AspectRatioMode
from PyQt5 import QtGui
import time
import random
import numpy as np
import cv2
import keyboard
import os
from pathlib import Path
from queue import Queue
from queue import Empty
from collections import namedtuple
import datetime
import json
import socket
import subprocess
import platform
import threading
import configparser

PYTHON_DIR = Path(__file__).resolve().parent
REPO_ROOT = PYTHON_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Synchro3 import PreviewFrame
from gui_workers import PostProcessingWorker, SimulationWorker, WorkerOutcome
from tools import (
    copyImageData,
    max_projected_length,
    is_visible,
    expand_bb,
    sensor_callback,
    synchro_queue,
    ImageObj,
    selected_labels,
    bb_labels,
    config,
    carla_path,
    check_carla_server,
    launch_carla_server,
    close_carla_server,
    convert_coordinates,
    spawn_vehicle,
    spawn_pedestrian,
    change_vehicle_direction,
    pedestrian_jump,
    load_scenario_from_yaml,
    move_spectator,
    run_scenario,
    process_colors,
    is_area_isolated,
    is_big_enough,
    is_object_fragmented,
    check_connection_status,
    ConsoleLogger,
    multiple_bbox_tags,
    CLASS_MAPPING,
    build_projection_matrix,
    get_RGB_DATA,
    update_rgb_camera_view,
    update_seg_camera_view,
    update_bounding_box_view_3D,
    update_bounding_box_view_simple,
    record_tick

)
from corner_case_form import CornerCaseEditor, CornerCaseForm


def create_directory(scenario_name):
    exec_path = PYTHON_DIR / 'execution'
    dir_path = os.path.join(exec_path, scenario_name)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.abspath(dir_path)


def write_data_to_disk():
    # Iterate over the data list and write each item to disk
    print('writing output to disk')

class MainWindow(QMainWindow):
    # Public compatibility signals used by integrations built against the earlier
    # GUI API. Internal workers use their own queued signals.
    simulation_finished = pyqtSignal(dict)
    simulation_failed = pyqtSignal(str)
    postprocessing_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.running = None
        self.inst_seg_camera = None
        self.seg_camera = None
        self.rgb_camera_ref = None
        self.rgb_camera = None
        self.inst_seg_camera_transform = None
        self.inst_seg_camera_bp = None
        self.seg_camera_transform = None
        self.seg_camera_bp = None
        self.rgb_ref_transform = None
        self.rgb_ref_bp = None
        self.rgb_camera_transform = None
        self.rgb_camera_bp = None
        self.modify_button = None
        self.info_label = None
        self.output_label = None
        self.textbox = None
        self.progress_label = None
        self.default_text = None
        self.progress_bar = None
        self.stop_btn = None
        self.pause_btn = None
        self.play_btn = None
        self.recording_label = None
        self.rec_btn = None
        self.corner_case_action = None
        self.record_action = None
        self.pause_action = None
        self.stop_action = None
        self.start_action = None
        self.corner_case_radio = None
        self.normal_radio = None
        self.radio3D = None
        self.radio2D = None
        self.label_bounding = None
        self.label_seg = None
        self.label_rgb = None
        self.folder_button = None
        self.vehicle = None
        self.scenario_length = 100
        self.x = 0
        self.scenario_tick = 0
        self.scenario_name = ''
        self.scenario_folder = ''
        self.global_imabe_error = 0
        self.camera_tick = 0
        self.ThreeD = True
        self.ImageCounter = 0
        self.synchro_list = []
        self.selected_labels = selected_labels
        self.bb_labels = bb_labels
        self.is_running = False
        self.is_recording = True
        self.is_step_by_step = False
        self.desired_width = 1280
        self.desired_height = 1024
        self.sensor_width = 1280
        self.sensor_height = 1024
        self.selected_category = "Scenario Level"
        self.selected_subcategory = "Anomalous Scenario"
        self.selected_example = "Example"
        self.corner_case_form = None
        self.CornerCaseEditor = CornerCaseEditor(str(PYTHON_DIR / 'CC_terminology.json'), self)
        self.is_corner_case = False
        self.spawned_actors = []
        self.multiple_bbox_tags_colors = []
        self.client = None
        self.world = None
        self.simulation_thread = None
        self.simulation_worker = None
        self.postprocessing_thread = None
        self.postprocessing_worker = None
        self.simulation_outcome = None
        self.postprocessing_outcome = None
        self.pending_result = None
        self.pending_error = None
        self.close_pending = False
        self.stop_requested = False
        self.init_ui()
        self.timer_id = None
        self.K = build_projection_matrix(self.sensor_width, self.sensor_height, 90)
        self.simulation_finished.connect(self._capture_succeeded)
        self.simulation_failed.connect(self._capture_failed)
        self.postprocessing_failed.connect(self.on_postprocessing_failed)

        # Initialize the CARLA client and world
        # self.init_carla_client()

        # Adjust the graphics and world settings
        # self.adjust_carla_settings()

        # Initiate bounding_box_labels
        # self.init_bounding_box_labels()

        # Set the weather parameters
        # self.set_weather_parameters()
        # self.K = self.build_projection_matrix(self.sensor_width, self.sesor_height, 90)
        # self.init_carla_scenario()

    def init_ui(self):

        self.folder_button = QPushButton('Change output directory')
        self.generate_Scenario_name()

        # Set up the main window
        self.setGeometry(100, 100, 800, 600)

        group_box = QGroupBox('City Object Labels')
        # Create the checkboxes and add them to the group box
        checkbox_layout = QGridLayout()
        for i, (label, value) in enumerate(bb_labels.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(label in selected_labels)
            # self.selected_labels.append(label)
            checkbox.stateChanged.connect(lambda state, label=label: self.checkbox_state_changed(state, label))
            row = i // 4  # display 4 checkboxes per row
            col = i % 4
            checkbox_layout.addWidget(checkbox, row, col)
        group_box.setLayout(checkbox_layout)

        # Set up the palette of editing a scenario
        output_layout, progress_layout, recording_widget, palette_widget, info_layout = self.setup_palette()

        # Set up the menu bar
        menu_bar = self.setup_menu_bar()

        # Set up the radio buttons for bounding box mode
        main_bb_vbox = self.setup_bb_radio_buttons()
        main_cc_vbox = self.setup_scenario_radio_buttons()

        # Set up the layout for the main window
        self.setup_layout(output_layout, main_bb_vbox, main_cc_vbox, progress_layout, recording_widget, palette_widget,
                          group_box, info_layout)

    def setup_layout(self, output_layout, main_bb_vbox, main_cc_vbox, progress_layout, recording_widget, palette_widget,
                     group_box, info_layout):
        central_widget = QWidget()
        layout = QGridLayout()
        self.label_rgb = QLabel(self)
        self.label_seg = QLabel(self)
        self.label_bounding = QLabel(self)
        layout.addWidget(self.label_rgb, 0, 0)
        layout.addWidget(self.label_seg, 0, 1)
        layout.addWidget(self.label_bounding, 0, 2)
        layout.addLayout(output_layout, 1, 0)
        layout.addLayout(main_bb_vbox, 1, 2)
        layout.addLayout(main_cc_vbox, 1, 3)
        layout.addLayout(progress_layout, 1, 1)
        layout.addWidget(recording_widget, 2, 1, 1, 1)
        layout.addWidget(palette_widget, 3, 0, 1, 3)
        layout.addLayout(info_layout, 3, 3)
        layout.addWidget(group_box)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_carla_client(self):
        i = 0
        self.client = check_carla_server()
        if self.client is None:
            if not carla_path:
                print("CARLA is not running and CARLA_ROOT/config.ini path is not configured")
                return
            print("Starting Carla Server")
            t = threading.Thread(target=launch_carla_server)
            t.start()

        while self.client is None and i < 50:
            i += 1
            self.client = check_carla_server()
            time.sleep(1)
        if self.client is None:
            print("Impossible to connect: Please check that Carla is installed correctly and try again later")
        else:
            self.world = self.client.get_world()
            print("connected")

    def adjust_carla_settings(self):
        pass

    def init_carla_scenario(self):
        pass

    def setup_bb_radio_buttons(self):
        self.radio2D = QRadioButton('2D')
        self.radio3D = QRadioButton('3D')
        # self.radio3 = QRadioButton('Radio 3')

        # Set radio1 as the default selected button
        self.radio3D.setChecked(True)

        # Create a group box and add the radio buttons to it
        group_Radio_box = QGroupBox("Bounding boxes mode")
        vbox = QVBoxLayout()
        vbox.addWidget(self.radio2D)
        vbox.addWidget(self.radio3D)
        self.radio2D.toggled.connect(self.on_radio2D_toggled)
        self.radio3D.toggled.connect(self.on_radio3D_toggled)

        group_Radio_box.setLayout(vbox)
        # Create a layout and add the group box to it
        main_bb_vbox = QVBoxLayout()
        main_bb_vbox.addWidget(group_Radio_box)

        return main_bb_vbox

    def toggle_scenario_selection(self, checked):
        if checked:
            self.info_label.setText("Normal Driving random scenarios")
            self.modify_button.setEnabled(False)
            self.is_corner_case = False
            self.corner_case_form.close()
            self.scenario_length = 200
        else:
            # self.info_label.setText("Normal scenarios")
            self.is_corner_case = True
            self.update_info_label(self.selected_category, self.selected_subcategory, self.selected_example)
            self.modify_button.setEnabled(True)
            self.scenario_length = 200

    def update_info_label(self, category, subcategory, example):
        self.selected_category = category
        self.selected_subcategory = subcategory
        selected_example = example
        self.selected_example = selected_example if selected_example else ""
        self.info_label.setText(f"Category: {category}\nSub-Category: {subcategory}\nExample: {example}")

    def setup_scenario_radio_buttons(self):
        self.normal_radio = QRadioButton("Normal Driving random scenarios")
        self.normal_radio.setChecked(True)
        self.corner_case_radio = QRadioButton("Corner Case Scenarios")
        self.normal_radio.toggled.connect(self.toggle_scenario_selection)
        # self.radio3 = QRadioButton('Radio 3')

        # Create a group box and add the radio buttons to it
        group_Radio_box = QGroupBox("Select Scenario Type")
        vbox = QVBoxLayout()
        vbox.addWidget(self.normal_radio)
        vbox.addWidget(self.corner_case_radio)

        group_Radio_box.setLayout(vbox)
        # Create a layout and add the group box to it
        main_scenario_vbox = QVBoxLayout()
        main_scenario_vbox.addWidget(group_Radio_box)

        return main_scenario_vbox

    def on_radio2D_toggled(self, checked):
        if checked:
            self.ThreeD = False
            # Perform actions specific to the 2D mode

    def on_radio3D_toggled(self, checked):
        if checked:
            self.ThreeD = True
            # Perform actions specific to the 3D mode

    def setup_menu_bar(self):
        menu_bar = self.menuBar()
        scenario_menu = menu_bar.addMenu("Scenario")
        self.start_action = scenario_menu.addAction("Start")
        self.stop_action = scenario_menu.addAction("Stop")
        self.pause_action = scenario_menu.addAction("Pause")
        self.record_action = scenario_menu.addAction("Stop recording")
        self.start_action.setEnabled(True)
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(True)
        # Connect the menu bar actions to functions
        self.start_action.triggered.connect(self.start_scenario)
        self.stop_action.triggered.connect(self.stop_scenario)
        self.pause_action.triggered.connect(self.pause_scenario)
        self.record_action.triggered.connect(self.record_scenario)
        self.corner_case_action = QAction("Corner Cases", self)
        self.corner_case_action.triggered.connect(self.show_cornercase_form)
        # corner_menu = menu_bar.addMenu("Tools")
        scenario_menu.addAction(self.corner_case_action)
        return menu_bar

    def setup_palette(self):
        recoring_layout = QHBoxLayout()
        recording_widget = QWidget()
        progress_layout = QVBoxLayout()  # Create a vertical layout
        # create a horizontal layout for the textbox and button
        h_box = QHBoxLayout()
        self.rec_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/ON.png")), "")
        self.rec_btn.setFixedSize(225, 100)  # set button size to 50x50 pixels
        self.rec_btn.setIconSize(self.rec_btn.size())
        self.rec_btn.setStyleSheet("QPushButton { border: none; }")
        self.recording_label = QLabel('Scenario recording is : ')

        recoring_layout.addStretch(1)
        recoring_layout.addWidget(self.recording_label)
        recoring_layout.addWidget(self.rec_btn)

        self.rec_btn.clicked.connect(self.record_scenario)
        self.play_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/play.png")), "")
        self.play_btn.setFixedSize(100, 100)  # set button size to 50x50 pixels
        self.play_btn.setIconSize(self.play_btn.size())
        self.play_btn.setStyleSheet("QPushButton { border: none; }")
        self.play_btn.clicked.connect(self.start_scenario)
        self.pause_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/pause.png")), "")
        self.pause_btn.setFixedSize(100, 100)  # set button size to 50x50 pixels
        self.pause_btn.setIconSize(self.pause_btn.size())
        self.pause_btn.setStyleSheet("QPushButton { border: none; }")
        self.pause_btn.clicked.connect(self.pause_scenario)
        self.stop_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/stop.png")), "")
        self.stop_btn.setFixedSize(100, 100)  # set button size to 50x50 pixels
        self.stop_btn.setIconSize(self.stop_btn.size())
        self.stop_btn.setStyleSheet("QPushButton { border: none; }")
        self.stop_btn.clicked.connect(self.stop_scenario)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)

        # Create a label widget
        self.default_text = "Select a folder to store the output data then click the play button to run scenario"
        self.progress_label = QLabel(self.default_text)
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)
        self.textbox = QLineEdit(self.scenario_folder)
        self.textbox.setReadOnly(True)
        # create a QPushButton for the folder selection button
        output_layout = QVBoxLayout()
        self.output_label = QLabel('output path:')

        self.folder_button.clicked.connect(self.select_folder)
        self.info_label = QLabel("Normal Driving random scenarios")
        self.info_label.setFixedHeight(100)  # Increased height
        self.modify_button = QPushButton("Modify")
        self.modify_button.setEnabled(False)
        self.modify_button.clicked.connect(self.show_cornercase_form)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.info_label)
        info_layout.addWidget(self.modify_button)

        h_box.addWidget(self.textbox)
        h_box.addWidget(self.folder_button)
        output_layout.addWidget(self.output_label)
        output_layout.addLayout(h_box)
        timeline_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/timeline.png")), "")
        new_event_btn = QPushButton(QIcon(str(REPO_ROOT / "icons/new_event.png")), "")
        palette_layout = QHBoxLayout()
        recording_widget.setLayout(recoring_layout)
        # addWidget(self.rec_btn)
        palette_layout.addWidget(self.play_btn)
        palette_layout.addWidget(self.pause_btn)
        palette_layout.addWidget(self.stop_btn)
        # palette_layout.addWidget(timeline_btn)
        # palette_layout.addWidget(new_event_btn)
        palette_widget = QWidget()
        palette_widget.setLayout(palette_layout)
        return output_layout, progress_layout, recording_widget, palette_widget, info_layout

    def select_folder(self):
        # open the file dialog and get the selected folder path
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.expanduser('~'))

        # update the value of the textbox with the selected folder path
        if folder_path:
            self.scenario_folder = folder_path
            self.textbox.setText(self.scenario_folder)

    def generate_Scenario_name(self):
        now = datetime.datetime.now()
        self.scenario_name = "Scenario_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        self.scenario_folder = create_directory(self.scenario_name)
        self.setWindowTitle("MultiTrans Virtualization Framework | " + self.scenario_folder)
        self.folder_button.setEnabled(True)

        print("new scenario folder is generated")
        print(self.scenario_folder)

    def start_scenario(self):
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            if self.simulation_worker.pause_event.is_set():
                self.simulation_worker.resume()
                self.is_running = True
                self.start_action.setEnabled(False)
                self.pause_action.setEnabled(True)
                self.start_action.setText('Start')
                return
            QMessageBox.warning(self, "Simulation already running", "Stop the current simulation before starting another.")
            return
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.init_carla_client()
        if self.world is None:
            self.on_simulation_failed("Could not connect to the CARLA server.")
            return
        self.is_running = True
        self.timer_id = self.startTimer(100)
        scenario_file = None
        if self.is_corner_case:
            scenario_file = str(PYTHON_DIR / f"{self.selected_example}.yaml")
            if not os.path.isfile(scenario_file):
                self.on_simulation_failed(f"Scenario file does not exist: {scenario_file}")
                return
        self.pending_result = None
        self.pending_error = None
        self.stop_requested = False
        self.simulation_thread = QThread(self)
        self.simulation_outcome = WorkerOutcome()
        self.simulation_worker = SimulationWorker(
            length=self.scenario_length, output_dir=self.scenario_folder, seed=0,
            scenario_path=scenario_file, outcome=self.simulation_outcome)
        self.simulation_worker.moveToThread(self.simulation_thread)
        self.simulation_thread.started.connect(self.simulation_worker.run)
        self.simulation_worker.preview_ready.connect(self.on_preview_ready, Qt.QueuedConnection)
        self.simulation_worker.progress_changed.connect(self.progress_bar.setValue, Qt.QueuedConnection)
        self.simulation_worker.finished.connect(self.simulation_worker.deleteLater, Qt.DirectConnection)
        self.simulation_worker.destroyed.connect(self.simulation_thread.quit, Qt.DirectConnection)
        self.simulation_thread.finished.connect(self.simulation_thread.deleteLater)
        self.simulation_thread.destroyed.connect(self._simulation_thread_destroyed)
        self.simulation_thread.start()

    @pyqtSlot(object)
    def on_preview_ready(self, frame):
        """Construct Qt painting objects only on the GUI thread."""
        if not isinstance(frame, PreviewFrame):
            return
        image = QImage(frame.raw_data, frame.width, frame.height,
                       frame.width * 4, QImage.Format_ARGB32).copy()
        pixmap = QPixmap.fromImage(image.scaled(600, 300, Qt.KeepAspectRatio))
        labels = {"rgb": self.label_rgb, "semantic_segmentation": self.label_seg,
                  "instance_segmentation": self.label_bounding}
        labels[frame.sensor_name].setPixmap(pixmap)

    def stop_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(False)
        self.progress_label.setText("Stopping simulation and restoring CARLA settings…")
        self.stop_requested = True
        if self.simulation_worker is not None:
            self.simulation_worker.stop()

    @pyqtSlot(dict)
    def _capture_succeeded(self, result):
        self.pending_result = result

    @pyqtSlot(str)
    def _capture_failed(self, message):
        self.pending_error = message

    @pyqtSlot()
    def _simulation_thread_destroyed(self):
        result = self.simulation_outcome.result if self.simulation_outcome else self.pending_result
        error = self.simulation_outcome.error if self.simulation_outcome else self.pending_error
        if error:
            self.simulation_failed.emit(error)
        elif result is not None:
            self.simulation_finished.emit(result)
        self.is_running = False
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.pause_action.setEnabled(False)
        self.folder_button.setEnabled(True)
        self.start_action.setText('Start')
        if self.timer_id is not None:
            self.killTimer(self.timer_id)
            self.timer_id = None
        self.client = None
        self.world = None
        self.simulation_worker = None
        self.simulation_thread = None
        self.simulation_outcome = None
        if error:
            self.progress_label.setText("Simulation failed")
            QMessageBox.critical(self, "CARLA simulation failed", error)
        elif result and result.get("stopped"):
            self.progress_label.setText(
                f"Cancelled after {len(result['captured_frames'])} synchronized frames; post-processing skipped")
        elif result:
            self.progress_label.setText(f"Captured {len(result['captured_frames'])} synchronized frames")
            self._start_postprocessing(result['output_dir'])
        self.scenario_tick = 0
        self.spawned_actors.clear()
        self.generate_Scenario_name()
        if self.close_pending:
            QTimer.singleShot(0, self.close)

    def _start_postprocessing(self, output_dir):
        self.postprocessing_thread = QThread(self)
        self.postprocessing_outcome = WorkerOutcome()
        self.postprocessing_worker = PostProcessingWorker(
            output_dir, str(PYTHON_DIR / 'environment_object.json'),
            outcome=self.postprocessing_outcome)
        self.postprocessing_worker.moveToThread(self.postprocessing_thread)
        self.postprocessing_thread.started.connect(self.postprocessing_worker.run)
        self.postprocessing_worker.finished.connect(self.postprocessing_worker.deleteLater, Qt.DirectConnection)
        self.postprocessing_worker.destroyed.connect(self.postprocessing_thread.quit, Qt.DirectConnection)
        self.postprocessing_thread.finished.connect(self.postprocessing_thread.deleteLater)
        self.postprocessing_thread.destroyed.connect(self._postprocessing_thread_destroyed)
        self.postprocessing_thread.start()

    def _postprocessing_thread_destroyed(self):
        error = self.postprocessing_outcome.error if self.postprocessing_outcome else None
        self.postprocessing_worker = None
        self.postprocessing_thread = None
        self.postprocessing_outcome = None
        if error:
            self.postprocessing_failed.emit(error)
        if self.close_pending:
            QTimer.singleShot(0, self.close)

    def on_simulation_failed(self, message):
        self.is_running = False
        self.start_action.setEnabled(True)
        self.stop_action.setEnabled(False)
        self.pause_action.setEnabled(False)
        self.folder_button.setEnabled(True)
        QMessageBox.critical(self, "CARLA simulation failed", message)

    def on_postprocessing_failed(self, message):
        QMessageBox.critical(self, "Dataset post-processing failed", message)

    def closeEvent(self, event):
        if self.simulation_thread is not None and self.simulation_thread.isRunning():
            self.close_pending = True
            self.stop_scenario()
            event.ignore()
            return
        if self.postprocessing_thread is not None and self.postprocessing_thread.isRunning():
            self.close_pending = True
            event.ignore()
            return
        event.accept()

    def pause_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(True)
        self.start_action.setText('Resume')
        self.is_running = False
        if self.simulation_worker is not None:
            self.simulation_worker.pause()

    def record_scenario(self):
        if self.record_action.text() == 'Record':
            self.record_action.setText('Stop recording')
            self.rec_btn.setIcon(QIcon(str(REPO_ROOT / "icons/ON.png")))
        else:
            self.record_action.setText('Record')
            self.rec_btn.setIcon(QIcon(str(REPO_ROOT / "icons/OFF.png")))
        self.is_recording = not self.is_recording

    def show_cornercase_form(self):
        self.CornerCaseEditor.show_cornercase_form()

    def init_arrays(self, instance_image, segmentation_image, rgb_image):
        instance_array = np.array(instance_image.raw_data)
        instance_array = instance_array.reshape((instance_image.height, instance_image.width, 4))
        instance_array = instance_array[:, :, :3]  # Remove alpha channel 

        sem_array = np.array(segmentation_image.raw_data)
        sem_array = sem_array.reshape((segmentation_image.height, segmentation_image.width, 4))
        sem_array = sem_array[:, :, :3]  # Remove alpha channel
        rgb_image_draw = np.array(rgb_image.raw_data)
        rgb_image_draw = rgb_image_draw.reshape((rgb_image.height, rgb_image.width, 4))
        rgb_image_draw = cv2.cvtColor(rgb_image_draw, cv2.COLOR_BGRA2BGR)
        alpha = 0.5
        blended_image = cv2.addWeighted(instance_array, alpha, sem_array, 1 - alpha, 0)
        return instance_array, sem_array, rgb_image_draw, alpha, blended_image
    
    def timerEvent(self, event):

        pass

    def spawn_vehicle_and_cameras(self):
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        # carla.Transform(carla.Location(x=35,y=35,z=0))

        points = self.world.get_map().get_spawn_points()

        vehicle_transform = random.choice(
            self.world.get_map().get_spawn_points())  # carla.Transform(carla.Location(x=-64.644844, y=24.471010, z=0.600000))

        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)
        self.spawned_actors.append(('ego', self.vehicle))
        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', '%d' % self.sensor_width)
        self.rgb_camera_bp.set_attribute('image_size_y', '%d' % self.sensor_height)
        self.rgb_camera_bp.set_attribute('fov', '90')
        self.rgb_camera_bp.set_attribute('sensor_tick', '0.0')
        self.rgb_camera_transform = carla.Transform(carla.Location(x=2, z=1.5, y=0))

        self.rgb_ref_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_ref_bp.set_attribute('image_size_x', '%d' % self.sensor_width)
        self.rgb_ref_bp.set_attribute('image_size_y', '%d' % self.sensor_height)
        self.rgb_ref_bp.set_attribute('fov', '90')
        self.rgb_ref_bp.set_attribute('sensor_tick', '0.0')
        self.rgb_ref_transform = self.rgb_camera_transform

        self.seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_camera_bp.set_attribute('image_size_x', '%d' % self.sensor_width)
        self.seg_camera_bp.set_attribute('image_size_y', '%d' % self.sensor_height)
        self.seg_camera_bp.set_attribute('fov', '90')
        self.seg_camera_bp.set_attribute('sensor_tick', '0.0')
        self.seg_camera_transform = self.rgb_camera_transform

        self.inst_seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.instance_segmentation')
        self.inst_seg_camera_bp.set_attribute('image_size_x', '%d' % self.sensor_width)
        self.inst_seg_camera_bp.set_attribute('image_size_y', '%d' % self.sensor_height)
        self.inst_seg_camera_bp.set_attribute('fov', '90')
        self.inst_seg_camera_bp.set_attribute('sensor_tick', '0.0')
        self.inst_seg_camera_transform = self.rgb_camera_transform

        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp,
            self.rgb_camera_transform,
            attach_to=self.vehicle
        )
        self.spawned_actors.append(('rgb', self.rgb_camera))

        self.rgb_camera_ref = self.world.spawn_actor(
            self.rgb_ref_bp,
            self.rgb_ref_transform,
            attach_to=self.vehicle
        )
        self.spawned_actors.append(('ref', self.rgb_camera_ref))
        self.seg_camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.seg_camera_transform,
            attach_to=self.vehicle
        )
        self.spawned_actors.append(('seg', self.seg_camera))
        self.inst_seg_camera = self.world.spawn_actor(
            self.inst_seg_camera_bp,
            self.inst_seg_camera_transform,
            attach_to=self.vehicle
        )
        self.spawned_actors.append(('inst', self.inst_seg_camera))
        # self.vehicle.set_autopilot(False)

        self.synchro_list.append(self.rgb_camera)
        self.synchro_list.append(self.rgb_camera_ref)
        self.synchro_list.append(self.seg_camera)
        self.synchro_list.append("bounding_boxes")
        self.synchro_list.append("camera_transform")
        self.synchro_list.append(self.inst_seg_camera)
        # self.synchro_list.append("blended_image")

        # Start the timer for updating the real-time views
        self.timer = QTimer()
        # self.timer.timeout.connect(self.update_views)
        self.timer.start(5000)
        self.rgb_camera.listen(
            lambda data: sensor_callback(self.world, self.rgb_camera, data, synchro_queue, "rgb_camera", self.K))
        self.rgb_camera_ref.listen(
            lambda data: sensor_callback(self.world, self.rgb_camera_ref, data, synchro_queue, "rgb_camera_ref",
                                         self.K))
        # self.rgb_camera.listen(lambda data: self.update_bounding_box_view(data))
        self.seg_camera.listen(
            lambda data: sensor_callback(self.world, self.seg_camera, data, synchro_queue, "semantic_segmentation",
                                         self.K))
        self.inst_seg_camera.listen(
            lambda data: sensor_callback(self.world, self.inst_seg_camera, data, synchro_queue, "instance_segmentation",
                                         self.K))
        self.timer = self.startTimer(100)
        self.running = True

        # new_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        # new_vehicle_location = self.vehicle.get_transform().location+ carla.Location(3,0,0)#carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#
        # new_vehicle_transform =random.choice(self.world.get_map().get_spawn_points()) # carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#carla.Transform(new_vehicle_location, vehicle_transform.rotation)
        # self.new_vehicle = self.world.spawn_actor(new_vehicle_bp, new_vehicle_transform)
        self.world.tick()
        # self.new_vehicle.set_transform(carla.Transform(new_vehicle_location))

        self.data = []


def main():
    main_window = None
    try:
        app = QApplication(sys.argv)
        main_window = MainWindow()
        main_window.show()
        return app.exec()
    except Exception as e:
        if main_window is not None and main_window.client is not None:
            check_connection_status(main_window.client)
        print("CornerSim startup failed:", e, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
