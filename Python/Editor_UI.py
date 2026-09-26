import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout, QComboBox, QGridLayout, QCheckBox, QGroupBox, QLineEdit, QProgressBar,QRadioButton,QMessageBox
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
#from PyQt5.QtCore import AspectRatioMode
from PyQt5 import QtGui
import time
from carla import ColorConverter
import carla
import random
import numpy as np
import cv2
import keyboard
from matplotlib import pyplot as plt
import os
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
import psutil
import copy
from cornersim.geometry import build_projection_matrix as maintained_projection_matrix, project_point
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
    close_carla_server
)






# Main Window

class MainWindow(QMainWindow):

    def __init__(self):
        
        super().__init__()

        self.scenario_length=20000
        self.scenario_tick = 0
        self.progress_updated = pyqtSignal(int)
        self.x = 0
        self.scenario_name = ''
        self.scenario_folder = ''
        self.bb_drawn = True
        self.global_imabe_error = 0
        self.camera_tick = 0
        self.ThreeD = True
        self.ImageCounter=0
        self.synchro_list = []
        self.selected_labels=selected_labels
        self.bb_labels= bb_labels
        self.is_running= False
        self.is_recording = True
        self.is_step_by_step=False
        self.desired_width = 1280
        self.desired_height = 1024
        self.sensor_width=1280
        self.sesor_height=1024
        self.init_ui()
        # Initialize the CARLA client and world
        self.init_carla_client()
        
        # Adjust the graphics and world settings
        self.adjust_carla_settings()
        
        # Initiate bounding_box_labels
        #self.init_bounding_box_labels()

        # Set the weather parameters
        self.set_weather_parameters()
        self.K = self.build_projection_matrix(self.sensor_width, self.sesor_height, 90)
        self.init_carla_scenario()
        #self.carla_setup()
        # Call other functions to initialize other variables if needed
 
        

    def spawn_vehicle_and_cameras(self):
        vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        #carla.Transform(carla.Location(x=35,y=35,z=0))

        points=self.world.get_map().get_spawn_points()
        
        
        vehicle_transform =random.choice(self.world.get_map().get_spawn_points()) #carla.Transform(carla.Location(x=-64.644844, y=24.471010, z=0.600000))
        
        self.vehicle = self.world.spawn_actor(vehicle_bp, vehicle_transform)

        self.rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', '%d' % self.sensor_width)
        self.rgb_camera_bp.set_attribute('image_size_y',  '%d' %self.sesor_height )
        self.rgb_camera_bp.set_attribute('fov', '90')
        self.rgb_camera_bp.set_attribute('sensor_tick', '0.0')
        self.rgb_camera_transform = carla.Transform(carla.Location(x=2, z=1.5, y=0))

        self.rgb_ref_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.rgb_ref_bp.set_attribute('image_size_x', '%d' %self.sensor_width)
        self.rgb_ref_bp.set_attribute('image_size_y','%d' % self.sesor_height)
        self.rgb_ref_bp.set_attribute('fov', '90')
        self.rgb_ref_bp.set_attribute('sensor_tick', '0.0')
        self.rgb_ref_transform=  self.rgb_camera_transform

        self.seg_camera_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        self.seg_camera_bp.set_attribute('image_size_x', '%d' %self.sensor_width)
        self.seg_camera_bp.set_attribute('image_size_y', '%d' %self.sesor_height)
        self.seg_camera_bp.set_attribute('fov', '90')
        self.seg_camera_bp.set_attribute('sensor_tick', '0.0')
        self.seg_camera_transform = self.rgb_camera_transform


        self.rgb_camera = self.world.spawn_actor(
            self.rgb_camera_bp,
            self.rgb_camera_transform,
            attach_to=self.vehicle
        )

        self.rgb_camera_ref = self.world.spawn_actor(
            self.rgb_ref_bp,
            self.rgb_ref_transform,
            attach_to=self.vehicle
        )

        self.seg_camera = self.world.spawn_actor(
            self.seg_camera_bp,
            self.seg_camera_transform,
            attach_to=self.vehicle
        )
        self.vehicle.set_autopilot(False)

        self.synchro_list.append(self.rgb_camera)
        self.synchro_list.append(self.rgb_camera_ref)
        self.synchro_list.append(self.seg_camera)
        self.synchro_list.append("bounding_boxes")
        self.synchro_list.append("camera_transform")

        # Start the timer for updating the real-time views
        self.timer = QTimer()
        #self.timer.timeout.connect(self.update_views)
        self.timer.start(5000)
        self.rgb_camera.listen(lambda data: sensor_callback(self.world,self.rgb_camera, data,synchro_queue,"rgb_camera" , self.K))
        self.rgb_camera_ref.listen(lambda data: sensor_callback(self.world,self.rgb_camera_ref, data,synchro_queue,"rgb_camera_ref",self.K))
        #self.rgb_camera.listen(lambda data: self.update_bounding_box_view(data))
        self.seg_camera.listen(lambda data: sensor_callback(self.world,self.seg_camera, data,synchro_queue,"semantic_segmentation",self.K))
        self.timer = self.startTimer(100)
        self.running = True

        new_vehicle_bp = random.choice(self.world.get_blueprint_library().filter('vehicle*'))
        new_vehicle_location = self.vehicle.get_transform().location+ carla.Location(3,0,0)#carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#
        new_vehicle_transform =random.choice(self.world.get_map().get_spawn_points()) # carla.Transform(carla.Location(x=-67.254570, y=27.963758, z=0.600000))#carla.Transform(new_vehicle_location, vehicle_transform.rotation)
        self.new_vehicle = self.world.spawn_actor(new_vehicle_bp, new_vehicle_transform)
        self.world.tick()
        self.new_vehicle.set_transform(carla.Transform(new_vehicle_location))

        self.data = []

    def init_ui(self):
        
        
        self.folder_button = QPushButton('Change output directory')
        self.generate_Scenario_name()
        
        # Set up the main window
        self.setGeometry(100, 100, 800, 600)
        


        group_box = QGroupBox('City Object Labels')
        # Create the checkboxes and add them to the group box
        checkbox_layout =  QGridLayout()
        for i, (label, value) in enumerate(bb_labels.items()):
            checkbox = QCheckBox(label)
            checkbox.setChecked(label in selected_labels)
            #self.selected_labels.append(label)
            checkbox.stateChanged.connect(lambda state, label=label: self.checkbox_state_changed(state, label))
            row = i // 4  # display 4 checkboxes per row
            col = i % 4
            checkbox_layout.addWidget(checkbox, row, col)
        group_box.setLayout(checkbox_layout)

        

       


        # Set up the palette of editing a scenario
        output_layout, progress_layout, recording_widget, palette_widget = self.setup_palette()

        # Set up the menu bar
        menu_bar = self.setup_menu_bar()

        # Set up the radio buttons for bounding box mode
        main_bb_vbox = self.setup_radio_buttons()

        # Set up the layout for the main window
        self.setup_layout(output_layout, main_bb_vbox, progress_layout, recording_widget, palette_widget,group_box)

    def setup_layout(self,output_layout, main_bb_vbox, progress_layout, recording_widget, palette_widget,group_box):
        central_widget = QWidget()
        layout = QGridLayout()
        self.label_rgb = QLabel(self)
        self.label_seg = QLabel(self)
        self.label_bounding = QLabel(self)
        layout.addWidget(self.label_rgb, 0, 0)
        layout.addWidget(self.label_seg, 0, 1)
        layout.addWidget(self.label_bounding, 0, 2)
        layout.addLayout(output_layout, 1,0)
        layout.addLayout(main_bb_vbox,1,2)
        layout.addLayout(progress_layout, 1, 1)
        layout.addWidget(recording_widget,2,1,1,1)
        layout.addWidget(palette_widget, 3, 0, 1, 3)
        layout.addWidget(group_box)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def setup_radio_buttons(self):
        self.radio2D = QRadioButton('2D')
        self.radio3D = QRadioButton('3D')
        #self.radio3 = QRadioButton('Radio 3')

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
    def on_radio2D_toggled(self, checked):
        if checked:
            self.ThreeD=False
            # Perform actions specific to the 2D mode

    def on_radio3D_toggled(self, checked):
        if checked:
            self.ThreeD=True
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
        return menu_bar

    def setup_palette(self):
        recoring_layout = QHBoxLayout()
        recording_widget = QWidget()
        progress_layout = QVBoxLayout()  # Create a vertical layout
        # create a horizontal layout for the textbox and button
        hbox = QHBoxLayout()
        self.rec_btn = QPushButton(QIcon("icons/ON.png"), "")
        self.rec_btn.setFixedSize(225, 100)  # set button size to 50x50 pixels
        self.rec_btn.setIconSize(self.rec_btn.size())
        self.rec_btn.setStyleSheet("QPushButton { border: none; }")
        self.recording_label = QLabel('Scenario recording is : ')
        
        recoring_layout.addStretch(1)
        recoring_layout.addWidget(self.recording_label)
        recoring_layout.addWidget(self.rec_btn)
        
        self.rec_btn.clicked.connect(self.record_scenario)
        self.play_btn = QPushButton(QIcon("icons/play.png"), "")
        self.play_btn.setFixedSize(100, 100)  # set button size to 50x50 pixels
        self.play_btn.setIconSize(self.play_btn.size())
        self.play_btn.setStyleSheet("QPushButton { border: none; }")
        self.play_btn.clicked.connect(self.start_scenario)
        self.pause_btn = QPushButton(QIcon("icons/pause.png"), "")
        self.pause_btn.setFixedSize(100, 100)  # set button size to 50x50 pixels
        self.pause_btn.setIconSize(self.pause_btn.size())
        self.pause_btn.setStyleSheet("QPushButton { border: none; }")
        self.pause_btn.clicked.connect(self.pause_scenario)
        self.stop_btn = QPushButton(QIcon("icons/stop.png"), "")
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
        output_layout= QVBoxLayout()
        self.output_label= QLabel('output path:')
        
        self.folder_button.clicked.connect(self.select_folder)
        
        
        hbox.addWidget(self.textbox)
        hbox.addWidget(self.folder_button)
        output_layout.addWidget(self.output_label)
        output_layout.addLayout(hbox)
        timeline_btn = QPushButton(QIcon("icons/timeline.png"), "")
        new_event_btn = QPushButton(QIcon("icons/new_event.png"), "")
        palette_layout = QHBoxLayout()
        recording_widget.setLayout(recoring_layout)
        #addWidget(self.rec_btn)
        palette_layout.addWidget(self.play_btn)
        palette_layout.addWidget(self.pause_btn)
        palette_layout.addWidget(self.stop_btn)
        #palette_layout.addWidget(timeline_btn)
        #palette_layout.addWidget(new_event_btn)
        palette_widget = QWidget()
        palette_widget.setLayout(palette_layout)
        return output_layout, progress_layout, recording_widget, palette_widget

    def init_carla_client(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(30.0)
        #self.client.load_world('Town05')
        self.world = self.client.get_world()
        
    def adjust_carla_settings(self):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        settings.quality_level = 'Ultra'
        settings.resolution = (1920, 1080)
        settings.anti_aliasing = '16x'
        settings.shadow_quality = 'Epic'
        settings.particles_quality_level = 'High'
        self.world.apply_settings(settings)

    def set_weather_parameters(self):
        weather = carla.WeatherParameters(
            cloudiness=random.uniform(0.0, 50.0),
            precipitation=random.uniform(-50, 0),
            precipitation_deposits=random.uniform(0.0, 50.0),
            wind_intensity=random.uniform(0.0, 50.0),
            sun_azimuth_angle=random.uniform(45.0, 135.0),
            sun_altitude_angle=random.uniform(45.0, 145.0),
            fog_density=random.uniform(0.0, 25.0),
            fog_distance=random.uniform(0.0, 200.0),
            fog_falloff=random.uniform(0.0, 200.0),
            wetness=random.uniform(0.0, 50.0),
            #puddles=random.uniform(0.0, 50.0),
            scattering_intensity=random.uniform(0.0, 50.0), 
            mie_scattering_scale=random.uniform(0.0, 50.0), 
            rayleigh_scattering_scale=0.03310000151395798, 
            dust_storm=random.uniform(0.0, 25.0),
            #snow_depth=random.uniform(0.0, 50.0),
            #ice_adherence=random.uniform(0.0, 50.0),
            #precipitation_type=carla.WeatherParameters.PrecipitationType.Snow
            #is_wet=True

        )
    def init_carla_scenario(self):
        # Initialize CARLA scenario-related variables
        self.is_running = False
        self.is_recording = True
        self.is_step_by_step = False

        # Spawn the vehicle and cameras
        self.spawn_vehicle_and_cameras()

        # Start the timer for updating the real-time views
        #self.start_timer()

        # Initialize other variables related to the CARLA scenario
        #self.new_vehicle = None
        #self.data = []

    def run_script(stop_event):
        try:
            subprocess.run(['python', 'generate_traffic.py'])
        except KeyboardInterrupt:
            stop_event.set()

    # Create a stop event
    stop_event = threading.Event()

    # Create a new thread
    thread = threading.Thread(target=run_script, args=(stop_event,))

    # Start the thread
    thread.start()

    def record_tick(self, json_array, rgb, semantic_image, img_bb):
        if self.is_running and self.is_recording:
            # Create a dictionary with the data to be stored
            #rgb_data=self.get_RGB_DATA(rgb)
            data_dict = {
                "json_array": json_array,
                "rgb": rgb,
                "semantic_image": semantic_image,
                "img_bb": img_bb,
            }
            # Append the data dictionary to the list
            self.data.append(data_dict)

    def write_data_to_disk(self):
        # Iterate over the data list and write each item to disk
        print('writing output to disk')
        self.progress_label.setText("initiating output structure")
        total = len(self.data)
        sem_seg_images= os.path.join(self.scenario_folder, "semantic_segmentation")
        bounding_box_images = os.path.join(self.scenario_folder,"bounding_box_images")
        json_files = os.path.join(self.scenario_folder,"bounding_box_json")
        original_images =  os.path.join(self.scenario_folder,"RGB_images")
        os.makedirs(sem_seg_images, exist_ok=True)
        os.makedirs(bounding_box_images, exist_ok=True)
        os.makedirs(json_files, exist_ok=True)
        os.makedirs(original_images, exist_ok=True)
        self.progress_label.setText("saving output to disk, please wait")
        for i, data_dict in enumerate(self.data):
            pro= 100*(i+1)/total
            json_path = os.path.join(json_files , 'image_{:06d}.json'.format(i))
            image_path = os.path.join(original_images, 'image_{:06d}.png'.format(i))
            image_seg_path = os.path.join(sem_seg_images, 'image_{:06d}.png'.format(i))
            image_bb_path = os.path.join(bounding_box_images, 'image_{:06d}.png'.format(i))

            if None not in data_dict["rgb"] and None not in data_dict["img_bb"] and data_dict["semantic_image"]!=None :
                try:
                    with open(json_path, "w") as f:
                        json.dump(data_dict["json_array"], f)

                    cv2.imwrite(image_path, data_dict["rgb"])
                    data_dict["semantic_image"].save_to_disk(image_seg_path)
                    cv2.imwrite(image_bb_path, data_dict["img_bb"])

                    self.progress_bar.setValue(pro)

                except Exception as e:
                    # Handle the exception here (e.g., log the error, display an error message)
                    print("An error occurred:", str(e))
                    # Optionally, undo any changes made before the exception occurred
            else:
                print("One or more objects are NoneType. The code was not executed.")
            #with open(json_path, "w") as f:
            ##    json.dump(data_dict["json_array"], f)
            ##cv2.imwrite(image_path, data_dict["rgb"])
           ## data_dict["semantic_image"].save_to_disk(image_seg_path)
           # #cv2.imwrite(image_bb_path, data_dict["img_bb"])
            #self.progress_bar.setValue(pro)
        # Clear the data list
        self.data = []
        print("scenario data is written to :")
        print(self.scenario_folder)
        self.progress_label.setText(self.default_text)
        self.show_alert()
        
    #    self.record_thread = None
#
    #def record_tick(self, json_array, rgb, semantic_image, img_bb):
    #    if self.is_running and self.is_recording:
    #        self.record_thread = RecordThread(json_array, rgb, semantic_image, img_bb, self.scenario_folder, self.ImageCounter)
    #        self.record_thread.finished.connect(self.record_finished)
    #        self.record_thread.error.connect(self.record_error)
    #        self.record_thread.start()

    def record_finished(self):
        self.record_thread = None

    def record_error(self, error_message):
        print(f"Recording error: {error_message}")
    def closeEvent(self, event):
        # This function will be called when the window is closed
        # Put your code here
        print("Window is closing")
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        self.destroy_actors()
        close_carla_server()

        event.accept()  # Accept the event to actually close the window

    def destroy_actors(self):
        
        self.seg_camera.destroy()
        self.rgb_camera.destroy()
        self.rgb_camera_ref.destroy()
        self.vehicle.destroy()

    def select_folder(self):
        # open the file dialog and get the selected folder path
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder', os.path.expanduser('~'))
        
        # update the value of the textbox with the selected folder path
        if folder_path:
            self.scenario_folder= folder_path
            self.textbox.setText(self.scenario_folder)
    def generate_Scenario_name(self):
        now = datetime.datetime.now()
        self.scenario_name = "Scenario_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        self.scenario_folder= self.create_directory(self.scenario_name)
        self.setWindowTitle("MultiTrans Virtualization Framework | " +  self.scenario_folder)
        self.folder_button.setEnabled(True)

        print("new scenario folder is generated")
        print(self.scenario_folder)
    def start_scenario(self):
        
        self.pause_action.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(False)
        self.folder_button.setEnabled(False)
        self.is_running=True
        #self.progress_updated.connect(self.update_progress)

    def stop_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(True)
        self.is_running=False
        self.write_data_to_disk()
        self.generate_Scenario_name()
        self.start_action.setText('Start')
        self.scenario_tick=0

    def pause_scenario(self):
        self.pause_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(True)
        self.start_action.setText('Resume')
        self.is_running=False
    def record_scenario(self):
        if self.record_action.text()=='Record':
            self.record_action.setText('Stop recording')
            self.rec_btn.setIcon(QIcon("icons/ON.png"))
        else:
            self.record_action.setText('Record')
            self.rec_btn.setIcon(QIcon("icons/OFF.png"))
        self.is_recording= not self.is_recording
        
    def synchroTick(self):
        if(self.camera_tick == 3):
            self.world.tick()
            self.camera_tick= 0;
            self.vehicle.set_autopilot(False)

    def checkbox_state_changed(self, state, label):
        # Update the value of the corresponding label in the dictionary
        if state == 2:
            self.selected_labels.append(label)
        elif state == 0:
            self.selected_labels.remove(label)
        #print(self.selected_labels)

    def timerEvent(self, event):
        #print(self.is_running)
        #keyboard.wait('1')
        #☺self.world.tick()
        if self.is_running and self.scenario_tick>= self.scenario_length:
            self.stop_scenario()
        progress=self.scenario_tick *100 /self.scenario_length
        self.progress_bar.setValue(progress)
        if self.is_running:
            self.progress_label.setText("running ..")
            self.world.tick()
            w_frame = self.world.get_snapshot().frame
            rgb_image=None
            segmentation_image=None
            boxes=None
            transform =None
            rgb_ref=None
            self.x+=1
            frame=0
            new_vehicle_location = self.vehicle.get_transform().location+ carla.Location(100-self.x,+4,0)
            #self.new_vehicle.set_transform(carla.Transform(new_vehicle_location, self.vehicle.get_transform().rotation))
            try:
                
                
                #print("Try")
                #print("sensors")
                #print(self.synchro_list)
                for _ in range(len(self.synchro_list)):
                    s_frame = synchro_queue.get(True, 10.0)
                    #print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))
                    frame=s_frame[0]

                    if s_frame[1] == "rgb_camera":
                        rgb_image=self.get_RGB_DATA(s_frame[2])
                    if s_frame[1] == "rgb_camera_ref":
                        rgb_ref=self.update_rgb_camera_view(s_frame[2])
                    if s_frame[1] == "semantic_segmentation":
                        segmentation_image=self.update_seg_camera_view(s_frame[2])
                    if s_frame[1] == "bounding_boxes":
                        boxes= s_frame[2]
                    if s_frame[1] == "camera_transform":
                        transform= s_frame[2]
                    #print(s_frame[1] )
                    #print(s_frame[2] )
                    print(frame)
                if rgb_image!=None and boxes!=None and transform!=None :
                    #print(rgb_image.width)
                    if frame%20==0:
                        if self.ThreeD:
                            self.update_bounding_box_view_3D(self.rgb_camera,rgb_image,segmentation_image,boxes,transform,rgb_ref)
                        else:
                            self.update_bounding_box_view_smart(self.rgb_camera,rgb_image,segmentation_image,boxes,transform,rgb_ref)
                else:
                    print("at least one of these data is missing:")
                    if rgb_image==None:
                        print('rgb')
                    if segmentation_image==None:
                        print('segmentation_image')
                    if boxes==None:
                        print('boxes')
                    if transform==None:
                        print('transform')

                
                self.scenario_tick+=1

            except Exception as e:
                    print(e)
            self.vehicle.set_autopilot(True)

    def start_stop_camera(self):
        # Start or stop the camera movement
        if self.running:
            self.camera.stop()
            self.folder_button.setText('Start')
        else:
            self.camera.listen(lambda data: self.update_camera_view(data))
            self.folder_button.setText('Stop')
        self.running = not self.running

    def get_RGB_DATA(self,image):
        new_raw_data = copyImageData(image)
        #copy.deepcopy(image.raw_data)#bytearray(image.raw_data)
        new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)
        return new_image

    def update_rgb_camera_view(self, image):
        # Convert the Image object to a QImage object
        #new_raw_data = image.raw_data#bytearray(image.raw_data)
        #new_image = ImageObj(raw_data=new_raw_data, width=image.width, height=image.height,fov=image.fov)

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        qimage = QtGui.QImage(array.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled_qimage = qimage.scaled(self.desired_width, self.desired_height, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
        self.label_rgb.setPixmap(pixmap)
        actors = self.world.get_actors()
        objects = self.selected_labels
        return self.get_RGB_DATA(image)

    def build_projection_matrix(self, w, h, fov):
        return maintained_projection_matrix(int(w), int(h), float(fov))

    

    def get_image_point(self, loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        projected = project_point((loc.x, loc.y, loc.z), K, w2c)
        if projected is None:
            raise ValueError("point is behind the camera or intersects the near plane")
        return np.asarray(projected[:2])

    def no_tighten_bb(self,sem_seg_image, x_min, y_min, x_max, y_max, object_color):
        return x_min, y_min, x_max, y_max

    def tighten_bb(self,sem_seg_image, x_min, y_min, x_max, y_max, object_color):
        #if x_min<=0:
        #    print(f"x_min: {x_min} -> image coordinates: {np.max(x_min-1,0)}")
        #    x_min=0
        #if y_min<=0:
        #    print(f"y_min: {x_min} -> image coordinates: {np.max(x_min-1,0)}")
        #    y_min=0

        
        if sem_seg_image==None:
            #print("sem_seg_image is not valid")
            return -1,-1,-1,-1
        np_img = np.frombuffer(sem_seg_image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((sem_seg_image.height, sem_seg_image.width, 4))
        np_img = np_img[..., :3]
        object_mask = cv2.inRange(np_img, object_color, object_color)
        # Crop the object mask to the bounding box
        object_mask_bb = object_mask[y_min:y_max, x_min:x_max]
        # Find the indices of all pixels that belong to the object
        object_pixels_indices = np.argwhere(object_mask_bb != 0)

        if len(object_pixels_indices) == 0:
            self.global_imabe_error+=1
            return -1, -1, -1, -1

        x_min_new = x_min + np.min(object_pixels_indices[:, 1])
        y_min_new = y_min + np.min(object_pixels_indices[:, 0])
        x_max_new = x_min_new + np.max(object_pixels_indices[:, 1])
        y_max_new = y_min_new + np.max(object_pixels_indices[:, 0])

        if x_max_new<0 or y_max_new<0 or x_min_new>sem_seg_image.width or y_min_new> sem_seg_image.height:
            return -1, -1, -1, -1

        #if x_min_new<0:
        #    x_min_new=0
        #if y_min_new<0:
        #    y_min_new=0
        #if x_max_new>sem_seg_image.width:
        #    x_max_new=sem_seg_image.width
        #if y_max_new>sem_seg_image.height:
        #    y_max_new=sem_seg_image.height

        return x_min_new, y_min_new, x_max_new, y_max_new

    def compute_bb_distance(self,camera_location,corners):
        #corners = bb.get_world_vertices(carla.Transform())

        # Compute the center of the bounding box in world coordinates
        bounding_box_center = carla.Location()
        for corner in corners:
            bounding_box_center.x += corner.x
            bounding_box_center.y += corner.y
            bounding_box_center.z += corner.z
        bounding_box_center.x /= len(corners)
        bounding_box_center.y /= len(corners)
        bounding_box_center.z /= len(corners)
        distance = camera_location.distance(bounding_box_center)
        return distance

    def update_bounding_box_view_smart(self, camera, image, semantic_image, bounding_box_set, transform,image_ref):
        json_array = []
        world_2_camera = np.array(transform.get_inverse_matrix())
        #print("world_2_camera")
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        rgb = np.frombuffer(image_ref.raw_data, dtype=np.dtype("uint8"))
        #print("frombuffer")
        img = np.reshape(img, (image.height, image.width, 4))
        rgb = np.reshape(rgb, (image.height, image.width, 4))
        #rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        #print("reshape")
        #ego_bb = self.vehicle.bounding_box
        bb_id=0
        processed_boxes = {label: [] for label in bounding_box_set.keys()}
        for label, bb_set in bounding_box_set.items():
            for bb in bb_set:
                corners = bb.get_world_vertices(carla.Transform())
                distance=self.compute_bb_distance(camera.get_transform().location,corners)
                corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]


                corners = np.array(corners, dtype=int)
                x_min, y_min = np.min(corners, axis=0)
                x_max, y_max = np.max(corners, axis=0)
                if x_max - x_min >= image.width or y_max - y_min >= image.height:
                    continue
                if x_min<0 and x_max>0:
                    x_min=0
                if y_min <0 and y_max>0:
                    y_min=0
               

                if label in ['Bicycle','Motorcycle','Rider','Pedestrians']:
                    #w_max = (f * (x_max - x_min)) / z
                    w,h=self.bb_max_dim[label]
                    yaw=bb.rotation.yaw
                    pitch=bb.rotation.pitch
                    w_proj = w * np.abs(np.cos(yaw))
                    h_proj = h * np.abs(np.sin(pitch))
                    z= bb.location.distance(transform.location)
                    bb_width= int(max_projected_length(w_proj,z,self.K)/2)
                    bb_h = int(max_projected_length(h_proj,z,self.K)/2)
                    x_min=x_min - bb_width
                    x_max=x_max + bb_width

                    y_min=y_min- bb_h
                    y_max=y_max+ bb_h

                # Check if the bounding box is completely included in any previously processed box
                included = False
                for processed_bb in processed_boxes[label]:
                    if x_min >= processed_bb[0] and x_max <= processed_bb[2] and y_min >= processed_bb[1] and y_max <= processed_bb[3] and processed_bb[4]<distance:
                        included = True
                        break

                if not included:
                    # Process the bounding box if it's not included in any previously processed box
                    processed_boxes[label].append((x_min, y_min, x_max, y_max,distance))
                    # Check if bounding box is inside image dimensions
                    if x_min >= img.shape[1] or x_max < 0. or y_min >= img.shape[0] or y_max < 0:
                        continue
                    object_color=self.CLASS_MAPPING[label]
                    object_color=(object_color[2], object_color[1], object_color[0])
                    x_min_new , y_min_new, x_max_new, y_max_new = self.tighten_bb(semantic_image, x_min, y_min, x_max, y_max, object_color )
                    if not(x_min_new==-1 and y_min_new ==-1 and x_max_new==-1 and y_max_new== -1):

                        # Draw bounding box using OpenCV's rectangle function
                        cv2.rectangle(img, (x_min_new, y_min_new), (x_max_new, y_max_new), object_color, 2)

                        #label = 'vehicle'  # replace with the appropriate label for each object type
                        cv2.putText(img, label, (x_min_new, y_min_new-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
                        json_entry = {"id": bb_id, "label": label, "x_min": int(x_min_new), "y_min": int(y_min_new), "x_max": int(x_min_new), "y_max":int(y_max_new)}
                        json_array.append(json_entry)
                        bb_id += 1

        self.record_tick(json_array,rgb,semantic_image,img)
        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        scaled_qimage = qimage.scaled(self.desired_width, self.desired_height, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
        #pixmap = QtGui.QPixmap.fromImage(qimage)
        
        self.label_bounding.setPixmap(pixmap)
        self.ImageCounter += 1

    def update_bounding_box_view_3D(self, camera, image, semantic_image, bounding_box_set, transform,image_ref):
        # Convert the Image object to a QImage object
         # Define percentage to reduce bounding box size by
        reduction_percentage = 0.1
        json_array = []
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        img = np.frombuffer(copyImageData(image), dtype=np.dtype("uint8"))#np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
        rgb = np.frombuffer(copyImageData(image_ref), dtype=np.dtype("uint8"))#np.frombuffer(copy.deepcopy(image.raw_data), dtype=np.dtype("uint8"))
        rgb = np.reshape(rgb, (image.height, image.width, 4))
        #rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        img = np.reshape(img, (image.height, image.width, 4))
        #carla.CityObjectLabel.Car
        bb_id=0
        processed_boxes = {label: [] for label in bounding_box_set.keys()}
        for label, bb_set in bounding_box_set.items():
            for bb in bb_set:
                try :
                    corners = bb.get_world_vertices(carla.Transform())
                    distance=self.compute_bb_distance(camera.get_transform().location,corners)
                    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in corners]
                    object_color=self.CLASS_MAPPING[label]
                    object_color=(object_color[2], object_color[1], object_color[0])
                    # Check if the bounding box is visible to the camera
                    #line_of_sight = self.world.get_line_of_sight(camera.get_location(), bb.location)
                    #forward_vec = self.vehicle.get_transform().get_forward_vector()
                    #bb_direction = bb.location - camera.get_transform().location
                    #dot_product = forward_vec.x * bb_direction.x + forward_vec.y * bb_direction.y + forward_vec.z * bb_direction.z
                    #if dot_product > 0:
                    #if np.dot(forward_vec, bb_direction) > 0:
                    #relative_location = bb.location - camera.get_location()
                    #ang=camera.get_forward_vector().get_angle(relative_location)
                    #if ang < 80 and ang>=0:
                    # Define percentage to reduce bounding box size by
                    verts = [v for v in bb.get_world_vertices(carla.Transform())]
                    #    center_point = carla.Location()
                    #    for v in verts:
                    #        center_point += v
                    #    center_point /= len(verts)

                        # Calculate new vertices by reducing distance from center point by reduction percentage
                    #    new_verts = []
                    #    for v in verts:
                    #        direction = v - center_point
                    #        new_direction = direction - direction * reduction_percentage
                    #        new_v = center_point + new_direction
                    #        new_verts.append(new_v)

                        # Get image coordinates of new vertices
                    #    corners = [self.get_image_point(corner, self.K, world_2_camera) for corner in new_verts]

                        # Use NumPy to calculate min/max corners
                    corners = np.array(corners, dtype=int)
                    x_min, y_min = np.min(corners, axis=0).astype(int)
                    x_max, y_max = np.max(corners, axis=0).astype(int)
                    included=False
                    # Extract the region of interest from the semantic image using the bounding box coordinates
                    # Assume that 'semantic_image' is a CARLA Image object
                    semantic_data = np.frombuffer(semantic_image.raw_data, dtype=np.dtype("uint8"))
                    semantic_data = np.reshape(semantic_data, (semantic_image.height, semantic_image.width, 4))
                    semantic_data = semantic_data[:, :, :3]
                    roi = semantic_data[y_min:y_max, x_min:x_max]

                    # Count the number of pixels within the bounding box coordinates that have the correct semantic color
                    count = np.sum((roi == object_color).all(axis=2))

                    # Compute the total number of pixels within the bounding box coordinates
                    total = roi.shape[0] * roi.shape[1]

                    # If the ratio of the number of pixels with the correct semantic color to the total number of pixels is greater than or equal to 0.5, process the bounding box
                    
                    if count*2 >= total :
                        
                        for processed_bb in processed_boxes[label]:
                            if x_min >= processed_bb[0] and x_max <= processed_bb[2] and y_min >= processed_bb[1] and y_max <= processed_bb[3] and processed_bb[4]<distance:
                                included = True
                                break
                        if not included:
                            # Process the bounding box if it's not included in any previously processed box
                            processed_boxes[label].append((x_min, y_min, x_max, y_max,distance))
                            edge_array=[]
                            # Draw edges of the bounding box into the camera output
                            for edge in edges:
                                try: 
                                    p1 = self.get_image_point(verts[edge[0]], self.K, world_2_camera)
                                    p2 = self.get_image_point(verts[edge[1]], self.K, world_2_camera)
                                    # Draw the edges into the camera output
                                    cv2.line(img, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), object_color, 1)
                                    edge_array.append((p1.tolist(), p2.tolist()))
                                except Exception as e:
                                    print(e)
                                    continue
                                #label = 'vehicle'  # replace with the appropriate label for each object type
                            cv2.putText(img, label, (x_min, y_min -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, object_color, 1)
                            json_entry = {"id": bb_id, "label": label, "edges": edge_array}
                            json_array.append(json_entry)
                            bb_id += 1
                except Exception as e:
                    continue

        self.record_tick(json_array,rgb,semantic_image,img)
        # Convert the image back to a QImage object and display it
        qimage = QtGui.QImage(img.data, image.width, image.height, QtGui.QImage.Format_RGB32)
        #pixmap = QtGui.QPixmap.fromImage(qimage)

        # Convert the QImage object to a QPixmap object and display it
        #pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled_qimage = qimage.scaled(self.desired_width, self.desired_height, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
        self.label_bounding.setPixmap(pixmap)
        self.ImageCounter += 1


    def update_seg_camera_view(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        np_img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        np_img = np_img.reshape((image.height, image.width, 4))
        np_img = np_img[..., :3]  # Remove the alpha channel
        #qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888)
        #qimage = qimage.rgbSwapped()
        #self.semantic_image= image
    # Use the palette to create a QImage
        qimage = QtGui.QImage(bytes(np_img.data), image.width, image.height, QtGui.QImage.Format_RGB888).rgbSwapped()
        #qimage = qimage.rgbSwapped()
        #self.semsegConversion(qimage)
        #qimage.setColorTable(palette_list)
        scaled_qimage = qimage.scaled(self.desired_width, self.desired_height, Qt.AspectRatioMode.KeepAspectRatio)
        pixmap = QtGui.QPixmap.fromImage(scaled_qimage)
        #pixmap = QtGui.QPixmap.fromImage(qimage)
        self.label_seg.setPixmap(pixmap)
        return image
        #self.camera_tick +=1
        #self.synchroTick()

    CLASS_MAPPING = {
        'unlabeled': (0, 0, 0),
        'Buildings': (70, 70, 70),
        'Fences': (100, 40, 40),
        'Other': (55, 90, 80),
        'Pedestrians': (220, 20, 60),
        'Rider': (255, 0, 0),
        'pole': (153, 153, 153),
        'RoadLines': (157, 234, 50),
        'Roads': (128, 64, 128),
        'Sidewalks': (244, 35, 232),
        'Vegetation': (107, 142, 35),
        'Bicycle' : (119,  11,  32),
        'Bus': (0, 60, 100),
        'Car': (0, 0, 142),
        'Truck': (0, 0, 70),
        'Motorcycle': (0, 0, 230),
        'vehicle': (0, 0, 142),
        'Walls': (102, 102, 156),
        'traffic_sign': (220, 220, 0),
        'Sky': (70, 130, 180),
        'Ground': (81, 0, 81),
        'Bridge': (150, 100, 100),
        'RailTrack': (230, 150, 140),
        'GuardRail': (180, 165, 180),
        'traffic_light': (250, 170, 30),
        'Static': (110, 190, 160),
        'Dynamic': (170, 120, 50),
        'Water': (45, 60, 150),
        'Terrain': (145, 170, 100)
    }
    bb_max_dim = {
        'Pedestrians' : (1,2.5),
        'Bicycle' : (3,2),
        'Motorcycle' : (3,2),
        'Rider' : (2,2.5)

    }
    def create_directory(self, scenario_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        exec_path= os.path.join(script_dir,'execution')
        dir_path = os.path.join(exec_path, scenario_name)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.abspath(dir_path)

    def semsegConversion(self,qimage):
        # Define a dictionary that maps label colors to class names
        label_color_map = {
            (0, 0, 0): 'Unlabeled',
            (70, 70, 70): 'Building',
            (190, 153, 153): 'Fence',
            (250, 170, 160): 'Other',
            (220, 220, 0): 'Pedestrian',
            (107, 142, 35): 'Vegetation',
            (152, 251, 152): 'Terrain',
            (70, 130, 180): 'Sky',
            (220, 20, 60): 'Car',
            (255, 0, 0): 'Traffic Sign',
            (0, 0, 142): 'Traffic Light'
        }

        # Create a list of color tuples from the dictionary
        palette_list = [(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)) for color in label_color_map.keys()]
        palette_list = [color[0] << 16 | color[1] << 8 | color[2] for color in palette_list]


        #palette_list = [color[0] << 16 | color[1] << 8 | color[2] for color in label_color_map.keys()]

        # Create a color table from the list of color tuples
        qimage.setColorTable(palette_list)

    def show_alert(self):
        # Create a QMessageBox
        message_box = QMessageBox()

        # Load the icon from a PNG file
        icon = QIcon('icons/save_success.png')

        # Set the size of the icon
        pixmap = icon.pixmap(160, 160)
        message_box.setIconPixmap(pixmap)

        # Set the message box title
        message_box.setWindowTitle('Data Saved')

        # Set the message text
        message_box.setText(f'Scenario data saved successfully to:\n\n{self.scenario_folder}.')

        # Create a button to open the folder
        open_button = message_box.addButton('Open Folder', QMessageBox.ActionRole)
        open_button.clicked.connect(self.open_folder)

        # Add the button to the QMessageBox
        message_box.addButton(open_button, QMessageBox.ActionRole)

        # Adjust the layout of the QMessageBox
        layout = message_box.layout()
        layout.setSizeConstraint(QVBoxLayout.SetNoConstraint)
        layout.setContentsMargins(20, 20, 20, 20)

        # Set the message box properties
        message_box.exec_()

    def open_folder(self):
        # Open the folder using QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.scenario_folder))





if __name__ == '__main__':
    if not check_carla_server():
        print('Starting Carla server...')
        t = threading.Thread(target=launch_carla_server)
        t.start()

        while not check_carla_server():
            time.sleep(1)

    app = QApplication(sys.argv)
    main_window = MainWindow()
    k=main_window.K
    #print(k)
    camera_location= main_window.rgb_camera.get_transform().location
    w2c=np.array(main_window.rgb_camera.get_transform().get_inverse_matrix())
    p1 = camera_location+ carla.Location(5, 0, 0)
    p2 = camera_location+ carla.Location(0, 5, 0)
    p3 = camera_location+ carla.Location(5, 0, 5)
    p4 = camera_location+ carla.Location(5, 5, 0)
    p5 = camera_location+ carla.Location(5, 0, 5)
    p6 = camera_location+ carla.Location(5, 5, 5)
    p7 = camera_location+ carla.Location(-5, -5, 0)
    p8 = camera_location+ carla.Location(-5, -5, -5)
    # Test the function with the in-view point
    point_img = main_window.get_image_point(p1, k, w2c)
    #print(f"point: {p1} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p2, k, w2c)
    #print(f"point: {p2} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p3, k, w2c)
    #print(f"point: {p3} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p4, k, w2c)
    #print(f"point: {p4} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p5, k, w2c)
    #print(f"point: {p5} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p6, k, w2c)
    #print(f"point: {p6} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p7, k, w2c)
    #print(f"point: {p7} -> image coordinates: {point_img}")
    point_img = main_window.get_image_point(p8, k, w2c)
    #print(f"point: {p8} -> image coordinates: {point_img}")

    

    # Define a test 3D point out of the camera's field of view
    point_out_of_view = carla.Location(-20, 10, 50)

    # Test the function with the out-of-view point
    point_img = main_window.get_image_point(point_out_of_view, k, w2c)
    #print(f"Out-of-view point: {point_out_of_view} -> image coordinates: {point_img}")
    main_window.show()
    sys.exit(app.exec_())







