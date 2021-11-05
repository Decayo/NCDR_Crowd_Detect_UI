#region import 依賴

import numpy
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.insert(0, './yolov5')
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadScreen_Capture
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, \
    check_imshow
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os,io
import shutil
import urllib
import time
from pathlib import Path
import cv2

from mss import mss
import os
import threading
from tkinter_function import *
#from ui_main import *
from win32 import win32api, win32gui
from win32.lib import win32con
import win32gui, win32ui, win32api, win32con
from win32.win32api import GetSystemMetrics
from datetime import datetime
import os,shutil,sys
from datetime import timezone
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
# selenium 爬蟲插件 用於抓取cctv串流圖片
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
from types import SimpleNamespace as Namespace
#from playsound import playsound
from pygame import mixer
import webbrowser
import atexit
import pandas as pd
import json
#endregion
class Main_Tracking():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'J:\Anaconda\envs\Pytorch_3,6\Lib\site-packages\PyQt5\Qt5\plugins'
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    class savable_var(object):
        def __init__(self):
            #print log and output time ,default 0 is changed then print
            self._log_time = 1
            #show streaming opencv float windows
            self._show_video_streaming = False
            # 候選框id過幾秒後會被重新識別成獨立之ID (ex: 5號id在180內被視為同一人 180秒之後會被重新計算一次並持續loop該邏輯)
            self.unique_refresh_time = 180
            self.g_isheadless = False
            #output format, user can define log out output format
            self.output_format_string = "{dtstring} - 目前偵測到 {people_count} (FPS:{fps:.2f})\\n人流 ：{len(g_unique_id_list)} / {_log_time}秒 "
            self.img_size_zoom = 1
            self.detect_interval_time = 0
            self.write_interval_time = 10
            self.write_csv_format='Time|Title|Count|AvgCount|TotalCount'
            self.sound_effect = 'alarm.wav'
            self.use_sound_effect = False
            self.sound_threshold = 10
            self.now_detect_count = 0
            self.avg_detect_count = 0.0
            self.total_detect_count = 0
            self.date_string = ''
            self.exe_title = '即時人流偵測軟體'
            self.model_cfg_json_path = 'yolov5_model_default.json'
            self.csv_Writing = False
            
            self.csv_w_path = None
            self.input_url = ''

        def to_json(self):
            '''
        convert the instance of this class to json
        '''
            return json.dumps(self, indent = 4, default=lambda o: o.__dict__)
    def __init__(self,run_id,saved_file_init_load = None):
        #ui Mainwindow class ref
        self.ui = None
        #args model config ref
        self.args = None
        
        #_time_counter use to calculate fps
        self._time_counter = 0
        
        #the unique id list, len this is the count
        self.g_unique_id_list = []
        #dictionary unique id list (id,time)
        self.total_unique_id_dict = {}
        
        #now log out put , has a thread detect and print when it was updated
        self.now_log_out = ''
        #src_state,button state , 0 is non-detecting , 1 is detecting
        self.src_state = 0
        #global webdriver , use to close when python crash or get exit
        self.g_webdriver = None
        #selenium webdriver, default is browser work on background , False will show the browser
        
        self.palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        
        
        self.csv_wr_thread_instance = None
        #初始化可保存參數

        self.sv = self.savable_var()
        self.sv_tojson =  self.sv.to_json()
        #print("model path 1 : "+ self.sv.model_cfg_json_path)
    
       
            #self.label_2.setText(os.path.basename(saved_file_init_load))
        # print(self.sv_tojson)
        # self.sv.sound_effect = 'haha bitch'
        # print(self.sv.sound_effect)
        # #self.sv = json.loads(self.sv_tojson, object_hook=lambda d: Namespace(**d))
        # print(self.sv.sound_effect)
        

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--weights', type=str, default='yolov5/weights/crowdhuman_yolov5m.pt', help='model.pt path')
        #parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5m.pt', help='model.pt path')
        # file/folder, 0 for webcam
        #model parameter
        self.parser.add_argument('--source', type=str, default='0', help='source')
        self.parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        self.parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        self.parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        self.parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        self.parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        self.parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        self.parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
        self.parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
        self.parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
        # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
        self.parser.add_argument('--classes', nargs='+', default=[0], type=int, help='filter by class')
        self.parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        self.parser.add_argument('--augment', action='store_true', help='augmented inference')
        self.parser.add_argument("--config-deepsort", type=str, default="deep_sort_pytorch/configs/deep_sort.yaml")
        self.parser.add_argument("--monitor-num", type=int, default=1)
        self.args = self.parser.parse_args()
        # with open(self.sv.model_cfg_json_path, 'w') as f:
        #     json.dump(self.args.__dict__, f, indent=2)
        
        self.app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = self.Ui_MainWindow(self)
        self.ui.setupUi(MainWindow)
        
        #設置按鈕邏輯
        MainWindow.show()
        self.ui.get_savable_ui_var()
        #載入初始設定
        if(saved_file_init_load is not None):
            with open(saved_file_init_load,encoding = 'utf-8') as json_data:
                data_dict = json.load(json_data)
            #print(data_dict)
            tmp_dump_dict = json.dumps(data_dict, indent=4)
            print(tmp_dump_dict)
            #self.outer.sv = self.outer.savable_var(**data_dict)
            self.sv = json.loads(tmp_dump_dict,object_hook=lambda d: Namespace(**d))
            self.ui.setting_savable_ui_var()
            self.now_log_out = '\n 讀取設定檔完成 ：\n' + saved_file_init_load +"\n"
        else:
            self.ui.setting_savable_ui_var()
        try:
            with open(self.sv.model_cfg_json_path, 'r') as f:
                tmp = "讀取模型檔案成功，參數如下：\n"
                tmp += self.sv.model_cfg_json_path +'\n'
                #print("WHY? " + tmp)
                self.args = self.parser.parse_args()
                try :
                    self.args.__dict__ = json.load(f)
                    tmpstr = str(self.args.__dict__)
                    tmpstr = tmpstr.replace(',',',\n')
                    tmp += tmpstr
                    self.now_log_out = tmp
                except:
                    tmp += "讀取模型檔案失敗，格式不符合：\n"
                    tmp += '將採用初始設置... \n'
                    #print("wrong state?")
                    tmpstr = str(self.args.__dict__)
                    tmpstr = tmpstr.replace(',',',\n')
                    tmp += tmpstr
                    self.now_log_out = tmp                 
        except EnvironmentError:
            print("error model...")
            time.sleep(1)
            self.args = self.parser.parse_args()
            tmp = "讀取模型檔案失敗，採用預設設置\n"
            tmpstr = str(self.args.__dict__)
            tmpstr = tmpstr.replace(',',',\n')
            tmp += tmpstr
            self.now_log_out = tmp
        #---------check pytorch edition
        
        self.now_log_out =self.now_log_out + "\n\nPytorch 版本：" + str(torch.__version__) + "\n cuda是否啟用(GPU加速:)" + str( torch.cuda.is_available()) +"\n\n 若GPU加速未開啟，請安裝與pytorch版本相應cuda版本"

        #self.args = parser.parse_args()
        self.args.img_size = check_img_size(self.args.img_size)
        self.root_dir = os.getcwd()
        sys.exit(self.app.exec())
    #region 背景計算thread
    class TotalCount_UpdateThread(QtCore.QThread):
        received = QtCore.pyqtSignal([str])
        def __init__(self,outer):
            super().__init__()
            self.outer = outer
        def run(self):
            self.last_ttc = self.outer.sv.total_detect_count
            #self.sig_msg.emit(string)
            """
            Pretend this worker method does work that takes a long time. During this time, the thread's
            event loop is blocked, except if the application's processEvents() is called: this gives every
            thread (incl. main) a chance to process events, which in this sample means processing signals
            received from GUI (such as abort).
            """
            while True:
                time.sleep(0.01)
                if(self.outer.sv._log_time <= 0.1):
                    time.sleep(0.1)
                else:
                    time.sleep(self.outer.sv._log_time)
                if(self.outer.now_log_out == self.last_ttc):
                    #print('now waiting output')
                    continue
                else:
                    self.received.emit(str(self.outer.sv.total_detect_count))
                    self.last_ttc = self.outer.sv.total_detect_count
                    # self.textBrowser.append(now_log_out)
                    # verScrollBar = self.textBrowser.verticalScrollBar()
                    # verScrollBar.setValue(verScrollBar.maximum())
                    # print("emit result:",now_log_out)
    class AvgCount_UpdateThread(QtCore.QThread):
        received = QtCore.pyqtSignal([str])
        def __init__(self,outer):
            super().__init__()
            self.outer = outer
        def run(self):
            self.last_adc = self.outer.sv.avg_detect_count
            #self.sig_msg.emit(string)
            """
            Pretend this worker method does work that takes a long time. During this time, the thread's
            event loop is blocked, except if the application's processEvents() is called: this gives every
            thread (incl. main) a chance to process events, which in this sample means processing signals
            received from GUI (such as abort).
            """
            while True:
                time.sleep(0.01)
                if(self.outer.sv._log_time <= 0.1):
                    time.sleep(0.1)
                else:
                    time.sleep(self.outer.sv._log_time)
                if(self.outer.sv.avg_detect_count >= self.outer.sv.sound_threshold) and self.outer.sv.use_sound_effect:
                    try:
                        mixer.music.load(self.outer.sv.sound_effect)
                        mixer.music.play()
                    except:
                        time.sleep(0.3)
                        self.outer.now_log_out = "音效檔案遺失"
                    #playsound(self.outer.sv.sound_effect)
                if(self.outer.sv.avg_detect_count == self.last_adc):
                    #print('now waiting output')
                    continue
                else:
                    self.received.emit(str(self.outer.sv.avg_detect_count))
                    self.last_adc = self.outer.sv.avg_detect_count
    class Logout_UpdateThread(QtCore.QThread):
        received = QtCore.pyqtSignal([str])
        def __init__(self,outer):
            super().__init__()
            self.outer = outer
        def run(self):
            self.last_log_out = self.outer.now_log_out
            #self.sig_msg.emit(string)
            """
            Pretend this worker method does work that takes a long time. During this time, the thread's
            event loop is blocked, except if the application's processEvents() is called: this gives every
            thread (incl. main) a chance to process events, which in this sample means processing signals
            received from GUI (such as abort).
            """
            while True:
                time.sleep(0.01)
                if(self.outer.sv._log_time <= 0.1):
                    time.sleep(0.1)
                else:
                    time.sleep(self.outer.sv._log_time)
                if(self.outer.now_log_out == self.last_log_out):
                    #print('now waiting output')
                    continue
                else:
                    self.received.emit(self.outer.now_log_out)
                    self.last_log_out = self.outer.now_log_out
                    # self.textBrowser.append(now_log_out)
                    # verScrollBar = self.textBrowser.verticalScrollBar()
                    # verScrollBar.setValue(verScrollBar.maximum())
                    # print("emit result:",now_log_out)
    #endregion

    #ui 布局
    class Ui_MainWindow(object):
        sig_start = pyqtSignal()
        #----------不可替代-------
        def __init__(self, outer):
            self.outer = outer
            self.m_MainWindow = None
        def setupUi(self, MainWindow):
            MainWindow.setObjectName("MainWindow")
            MainWindow.resize(656, 683)
            self.centralwidget = QtWidgets.QWidget(MainWindow)
            self.centralwidget.setObjectName("centralwidget")
            self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
            self.gridLayout.setObjectName("gridLayout")
            self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
            self.checkBox_3.setObjectName("checkBox_3")
            self.gridLayout.addWidget(self.checkBox_3, 27, 4, 1, 1)
            self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.lineEdit_4.sizePolicy().hasHeightForWidth())
            self.lineEdit_4.setSizePolicy(sizePolicy)
            self.lineEdit_4.setObjectName("lineEdit_4")
            self.gridLayout.addWidget(self.lineEdit_4, 0, 1, 1, 1)
            spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
            self.gridLayout.addItem(spacerItem, 14, 3, 1, 1)
            self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.lineEdit_3.sizePolicy().hasHeightForWidth())
            self.lineEdit_3.setSizePolicy(sizePolicy)
            self.lineEdit_3.setObjectName("lineEdit_3")
            self.gridLayout.addWidget(self.lineEdit_3, 1, 1, 1, 2)
            self.line_2 = QtWidgets.QFrame(self.centralwidget)
            self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
            self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line_2.setObjectName("line_2")
            self.gridLayout.addWidget(self.line_2, 5, 4, 1, 1)
            self.label_8 = QtWidgets.QLabel(self.centralwidget)
            self.label_8.setObjectName("label_8")
            self.gridLayout.addWidget(self.label_8, 21, 4, 1, 1)
            self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
            self.checkBox_4.setChecked(False)
            self.checkBox_4.setObjectName("checkBox_4")
            self.gridLayout.addWidget(self.checkBox_4, 2, 5, 1, 1)
            self.label_2 = QtWidgets.QLabel(self.centralwidget)
            self.label_2.setObjectName("label_2")
            self.gridLayout.addWidget(self.label_2, 31, 6, 1, 1)
            self.label_21 = QtWidgets.QLabel(self.centralwidget)
            self.label_21.setObjectName("label_21")
            self.gridLayout.addWidget(self.label_21, 26, 4, 1, 1)
            self.label_20 = QtWidgets.QLabel(self.centralwidget)
            self.label_20.setObjectName("label_20")
            self.gridLayout.addWidget(self.label_20, 11, 6, 1, 1)
            self.label_19 = QtWidgets.QLabel(self.centralwidget)
            self.label_19.setObjectName("label_19")
            self.gridLayout.addWidget(self.label_19, 31, 4, 1, 1)
            self.label_5 = QtWidgets.QLabel(self.centralwidget)
            font = QtGui.QFont()
            font.setFamily("Agency FB")
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.label_5.setFont(font)
            self.label_5.setMouseTracking(False)
            self.label_5.setObjectName("label_5")
            self.gridLayout.addWidget(self.label_5, 13, 4, 1, 1)
            self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.textBrowser.sizePolicy().hasHeightForWidth())
            self.textBrowser.setSizePolicy(sizePolicy)
            self.textBrowser.setObjectName("textBrowser")
            self.gridLayout.addWidget(self.textBrowser, 2, 0, 32, 3)
            self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
            self.checkBox_2.setChecked(False)
            self.checkBox_2.setObjectName("checkBox_2")
            self.gridLayout.addWidget(self.checkBox_2, 10, 4, 1, 1)
            self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.lineEdit_5.sizePolicy().hasHeightForWidth())
            self.lineEdit_5.setSizePolicy(sizePolicy)
            self.lineEdit_5.setMaximumSize(QtCore.QSize(20, 16777215))
            self.lineEdit_5.setObjectName("lineEdit_5")
            self.gridLayout.addWidget(self.lineEdit_5, 7, 5, 1, 1)
            self.label_10 = QtWidgets.QLabel(self.centralwidget)
            self.label_10.setObjectName("label_10")
            self.gridLayout.addWidget(self.label_10, 21, 5, 1, 1)
            self.label_15 = QtWidgets.QLabel(self.centralwidget)
            self.label_15.setObjectName("label_15")
            self.gridLayout.addWidget(self.label_15, 3, 4, 1, 5)
            self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.pushButton_5.sizePolicy().hasHeightForWidth())
            self.pushButton_5.setSizePolicy(sizePolicy)
            self.pushButton_5.setObjectName("pushButton_5")
            self.gridLayout.addWidget(self.pushButton_5, 1, 4, 1, 1)
            self.label_6 = QtWidgets.QLabel(self.centralwidget)
            font = QtGui.QFont()
            font.setFamily("Agency FB")
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.label_6.setFont(font)
            self.label_6.setMouseTracking(False)
            self.label_6.setObjectName("label_6")
            self.gridLayout.addWidget(self.label_6, 25, 4, 1, 1)
            self.label_25 = QtWidgets.QLabel(self.centralwidget)
            self.label_25.setObjectName("label_25")
            self.gridLayout.addWidget(self.label_25, 27, 6, 1, 1)
            self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_2.setObjectName("pushButton_2")
            self.gridLayout.addWidget(self.pushButton_2, 10, 5, 1, 1)
            self.label_17 = QtWidgets.QLabel(self.centralwidget)
            self.label_17.setObjectName("label_17")
            self.gridLayout.addWidget(self.label_17, 0, 0, 1, 1)
            self.label = QtWidgets.QLabel(self.centralwidget)
            font = QtGui.QFont()
            font.setFamily("Agency FB")
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.label.setFont(font)
            self.label.setMouseTracking(False)
            self.label.setObjectName("label")
            self.gridLayout.addWidget(self.label, 0, 4, 1, 4)
            self.label_16 = QtWidgets.QLabel(self.centralwidget)
            self.label_16.setObjectName("label_16")
            self.gridLayout.addWidget(self.label_16, 1, 0, 1, 1)
            self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_4.setObjectName("pushButton_4")
            self.gridLayout.addWidget(self.pushButton_4, 27, 5, 1, 1)
            self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_9.setObjectName("pushButton_9")
            self.gridLayout.addWidget(self.pushButton_9, 31, 5, 1, 1)
            self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_8.setObjectName("pushButton_8")
            self.gridLayout.addWidget(self.pushButton_8, 29, 5, 1, 1)
            self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
            self.checkBox.setSizePolicy(sizePolicy)
            self.checkBox.setChecked(True)
            self.checkBox.setObjectName("checkBox")
            self.gridLayout.addWidget(self.checkBox, 9, 4, 1, 1)
            self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_2.setObjectName("lineEdit_2")
            self.gridLayout.addWidget(self.lineEdit_2, 4, 4, 1, 4)
            self.label_7 = QtWidgets.QLabel(self.centralwidget)
            self.label_7.setMinimumSize(QtCore.QSize(80, 0))
            self.label_7.setObjectName("label_7")
            self.gridLayout.addWidget(self.label_7, 10, 6, 1, 2)
            self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_6.setMaximumSize(QtCore.QSize(50, 16777215))
            self.lineEdit_6.setObjectName("lineEdit_6")
            self.gridLayout.addWidget(self.lineEdit_6, 26, 5, 1, 1)
            self.label_13 = QtWidgets.QLabel(self.centralwidget)
            self.label_13.setObjectName("label_13")
            self.gridLayout.addWidget(self.label_13, 18, 4, 1, 1)
            self.pushButton = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton.setObjectName("pushButton")
            self.gridLayout.addWidget(self.pushButton, 11, 5, 1, 1)
            self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.pushButton_6.sizePolicy().hasHeightForWidth())
            self.pushButton_6.setSizePolicy(sizePolicy)
            self.pushButton_6.setObjectName("pushButton_6")
            self.gridLayout.addWidget(self.pushButton_6, 1, 5, 1, 1)
            self.line = QtWidgets.QFrame(self.centralwidget)
            self.line.setFrameShape(QtWidgets.QFrame.VLine)
            self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
            self.line.setObjectName("line")
            self.gridLayout.addWidget(self.line, 12, 4, 1, 1)
            self.label_3 = QtWidgets.QLabel(self.centralwidget)
            font = QtGui.QFont()
            font.setFamily("Agency FB")
            font.setPointSize(12)
            font.setBold(True)
            font.setWeight(75)
            self.label_3.setFont(font)
            self.label_3.setMouseTracking(False)
            self.label_3.setObjectName("label_3")
            self.gridLayout.addWidget(self.label_3, 6, 4, 1, 7)
            self.label_14 = QtWidgets.QLabel(self.centralwidget)
            self.label_14.setObjectName("label_14")
            self.gridLayout.addWidget(self.label_14, 18, 5, 1, 1)
            self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_3.setObjectName("pushButton_3")
            self.gridLayout.addWidget(self.pushButton_3, 33, 5, 1, 1)
            self.label_9 = QtWidgets.QLabel(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
            self.label_9.setSizePolicy(sizePolicy)
            self.label_9.setObjectName("label_9")
            self.gridLayout.addWidget(self.label_9, 7, 4, 1, 1)
            self.label_18 = QtWidgets.QLabel(self.centralwidget)
            self.label_18.setObjectName("label_18")
            self.gridLayout.addWidget(self.label_18, 29, 4, 1, 1)
            self.lineEdit_7 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_7.setMaximumSize(QtCore.QSize(20, 16777215))
            self.lineEdit_7.setObjectName("lineEdit_7")
            self.gridLayout.addWidget(self.lineEdit_7, 8, 5, 1, 1)
            self.label_11 = QtWidgets.QLabel(self.centralwidget)
            self.label_11.setObjectName("label_11")
            self.gridLayout.addWidget(self.label_11, 11, 4, 1, 1)
            self.label_22 = QtWidgets.QLabel(self.centralwidget)
            self.label_22.setObjectName("label_22")
            self.gridLayout.addWidget(self.label_22, 8, 4, 1, 1)
            self.label_26 = QtWidgets.QLabel(self.centralwidget)
            self.label_26.setObjectName("label_26")
            self.gridLayout.addWidget(self.label_26, 17, 4, 1, 1)
            self.label_12 = QtWidgets.QLabel(self.centralwidget)
            self.label_12.setObjectName("label_12")
            self.gridLayout.addWidget(self.label_12, 14, 4, 1, 1)
            self.label_23 = QtWidgets.QLabel(self.centralwidget)
            self.label_23.setObjectName("label_23")
            self.gridLayout.addWidget(self.label_23, 16, 4, 1, 1)
            self.lineEdit_9 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_9.setMaximumSize(QtCore.QSize(30, 16777215))
            self.lineEdit_9.setObjectName("lineEdit_9")
            self.gridLayout.addWidget(self.lineEdit_9, 16, 5, 1, 1)
            self.lineEdit_10 = QtWidgets.QLineEdit(self.centralwidget)
            self.lineEdit_10.setMaximumSize(QtCore.QSize(30, 16777215))
            self.lineEdit_10.setObjectName("lineEdit_10")
            self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
            self.pushButton_7.setObjectName("pushButton_7")
            self.gridLayout.addWidget(self.pushButton_7, 25, 4, 1, 1)
            self.gridLayout.addWidget(self.lineEdit_10, 17, 5, 1, 1)
            self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
            self.lineEdit.setSizePolicy(sizePolicy)
            self.lineEdit.setMaximumSize(QtCore.QSize(30, 16777215))
            self.lineEdit.setObjectName("lineEdit")
            self.gridLayout.addWidget(self.lineEdit, 14, 5, 1, 1)
            self.commandLinkButton = QtWidgets.QCommandLinkButton(self.centralwidget)
            sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            sizePolicy.setHorizontalStretch(0)
            sizePolicy.setVerticalStretch(0)
            sizePolicy.setHeightForWidth(self.commandLinkButton.sizePolicy().hasHeightForWidth())
            self.commandLinkButton.setSizePolicy(sizePolicy)
            self.commandLinkButton.setMinimumSize(QtCore.QSize(100, 30))
            self.commandLinkButton.setMaximumSize(QtCore.QSize(120, 30))
            self.commandLinkButton.setSizeIncrement(QtCore.QSize(20, 0))
            self.commandLinkButton.setLayoutDirection(QtCore.Qt.LeftToRight)
            self.commandLinkButton.setAutoFillBackground(False)
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap("1200px-Torchlight_help_icon.svg.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            self.commandLinkButton.setIcon(icon)
            self.commandLinkButton.setObjectName("commandLinkButton")
            self.gridLayout.addWidget(self.commandLinkButton, 1, 6, 1, 1)
            MainWindow.setCentralWidget(self.centralwidget)
            self.statusbar = QtWidgets.QStatusBar(MainWindow)
            self.statusbar.setObjectName("statusbar")
            MainWindow.setStatusBar(self.statusbar)
            self.menubar = QtWidgets.QMenuBar(MainWindow)
            self.menubar.setGeometry(QtCore.QRect(0, 0, 656, 21))
            self.menubar.setObjectName("menubar")
            MainWindow.setMenuBar(self.menubar)

            self.retranslateUi(MainWindow)
            QtCore.QMetaObject.connectSlotsByName(MainWindow)

            self.setup_ui_extension(MainWindow)

        def retranslateUi(self, MainWindow):
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "即時串流影片群眾監測軟體"))
            self.checkBox_3.setText(_translate("MainWindow", "輸出結果(.csv)"))
            self.lineEdit_4.setText(_translate("MainWindow", "即時串流影片群眾監測軟體"))
            self.lineEdit_3.setText(_translate("MainWindow", "{dtstring}目前偵測到 {people_count} (FPS:{fps:.2f})\\\\n人流 ：{unique_id_list_count} / {l_log_time}秒"))
            self.label_8.setText(_translate("MainWindow", "偵測行人總數量："))
            self.checkBox_4.setText(_translate("MainWindow", "瀏覽器背景執行"))
            self.label_2.setText(_translate("MainWindow", "現在檔案名稱"))
            self.label_21.setText(_translate("MainWindow", "寫入間隔(秒)："))
            self.label_20.setText(_translate("MainWindow", "現在檔案名稱"))
            self.label_19.setText(_translate("MainWindow", "讀取設定檔(.json)"))
            self.label_5.setText(_translate("MainWindow", "串流狀態"))
            self.checkBox_2.setText(_translate("MainWindow", "音效提示"))
            self.lineEdit_5.setText(_translate("MainWindow", "1"))
            self.label_10.setText(_translate("MainWindow", "0"))
            self.label_15.setText(_translate("MainWindow", "串流網址："))
            self.pushButton_5.setText(_translate("MainWindow", "螢幕擷取"))
            self.label_6.setText(_translate("MainWindow", "保存監測數據"))
            self.label_25.setText(_translate("MainWindow", "現在檔案名稱"))
            self.pushButton_2.setText(_translate("MainWindow", "..."))
            self.label_17.setText(_translate("MainWindow", "視窗標題："))
            self.label.setText(_translate("MainWindow", "系統功能"))
            self.label_16.setText(_translate("MainWindow", "輸出格式："))
            self.pushButton_4.setText(_translate("MainWindow", "..."))
            self.pushButton_9.setText(_translate("MainWindow", "..."))
            self.pushButton_8.setText(_translate("MainWindow", "..."))
            self.checkBox.setText(_translate("MainWindow", "顯示即時結果"))
            self.label_7.setText(_translate("MainWindow", "現在檔案名稱"))
            self.lineEdit_6.setText(_translate("MainWindow", "10"))
            self.label_13.setText(_translate("MainWindow", "平均統計人數："))
            self.pushButton.setText(_translate("MainWindow", "..."))
            self.pushButton_6.setText(_translate("MainWindow", "串流網址"))
            self.label_3.setText(_translate("MainWindow", "參數選項"))
            self.label_14.setText(_translate("MainWindow", "0"))
            self.pushButton_3.setText(_translate("MainWindow", "退出"))
            self.label_9.setText(_translate("MainWindow", "偵測圖像放大放大倍率"))
            self.label_18.setText(_translate("MainWindow", "保存設定檔(.json)"))
            self.lineEdit_7.setText(_translate("MainWindow", "0"))
            self.label_11.setText(_translate("MainWindow", "模型參數檔案"))
            self.label_22.setText(_translate("MainWindow", "偵測間隔(秒)："))
            self.label_26.setText(_translate("MainWindow", "音效提示閾值(每秒/人)："))
            self.label_12.setText(_translate("MainWindow", "統計間隔(秒)："))
            self.label_23.setText(_translate("MainWindow", "獨立ID刷新間隔(秒)："))
            self.lineEdit_9.setText(_translate("MainWindow", "180"))
            self.lineEdit_10.setText(_translate("MainWindow", "10"))
            self.lineEdit.setText(_translate("MainWindow", "1"))
            self.commandLinkButton.setText(_translate("MainWindow", "使用手冊"))
            self.pushButton_7.setText(_translate("MainWindow", "統計人數歸零"))
    
        def setup_ui_extension(self,MainWindow):
            self.m_MainWindow = MainWindow 
            self.pushButton_5.clicked.connect(lambda: self.outer.screen_shot(self))
            self.pushButton_3.clicked.connect(lambda: self.outer.exit_btn())
            self.pushButton_6.clicked.connect(lambda: self.outer.url_detecting_thread(self))
            self.pushButton_2.clicked.connect(self.sound_effect_path)
            self.pushButton.clicked.connect(self.model_parameter_setting)
            self.pushButton_4.clicked.connect(self.csv_path_setting)
            self.pushButton_8.clicked.connect(self.save_setting)
            self.pushButton_9.clicked.connect(self.load_setting)
            self.pushButton.clicked.connect(self.load_model_config)
            self.pushButton_7.clicked.connect(self.reset_people_count)
            self.commandLinkButton.clicked.connect(lambda: webbrowser.open('https://hackmd.io/@ksv1v_gOQaSTb706S_k4BQ/H14737lHt'))

            self.lineEdit.textChanged.connect(self.outer.get_log_time_chaged)
            self.lineEdit_4.textChanged.connect(self.change_title)
            self.lineEdit_5.textChanged.connect(self.change_detect_imgsize)
            self.lineEdit_7.textChanged.connect(self.change_detect_interval)
            self.lineEdit_6.textChanged.connect(self.change_write_interval)
            #self.lineEdit_8.textChanged.connect(self.set_output_format)
            self.lineEdit_9.textChanged.connect(self.change_detect_id_refresh_time)
            self.checkBox.setChecked(False)
            self.checkBox.clicked.connect(lambda : self.outer.show_vid_checkbox())
            self.checkBox_4.clicked.connect(lambda : self.outer.headless_checkbox())
            self.checkBox_2.clicked.connect(self.sound_effect_checkbox)
            self.checkBox_3.clicked.connect(lambda: self.outer.csv_checkbox())
            self.t = self.outer.Logout_UpdateThread(self.outer)
            self.t.received.connect(self.textBrowser.append)
            self.t.start()

            self.t2 = self.outer.TotalCount_UpdateThread(self.outer)
            self.t2.received.connect(self.label_10.setText)
            self.t2.start()

            self.t3 = self.outer.AvgCount_UpdateThread(self.outer)
            self.t3.received.connect(self.label_14.setText)
            self.t3.start()
        def reset_people_count(self):
            self.outer.sv.now_detect_count = 0
            self.outer.sv.avg_detect_count = 0.0
            self.outer.sv.total_detect_count = 0
        def change_title(self):
            _translate = QtCore.QCoreApplication.translate
            self.outer.sv.exe_title = self.lineEdit_4.text()
            self.m_MainWindow.setWindowTitle(_translate("MainWindow",self.outer.sv.exe_title ))
        def change_detect_imgsize(self,text):
            try:
                f = float(text)

                if(f == 0):
                    self.outer.now_log_out = "放大倍率不能為0"
                    return
                elif(f < 0):
                    self.outer.now_log_out = "輸入不能為負數"
                    return
                self.outer.sv.img_size_zoom = f
                self.outer.now_log_out = "圖像倍率更改為 " +text+" 倍"
            except ValueError:
                self.outer.now_log_out = '輸入僅能為整數或小數點'
        def change_detect_interval(self,text):
            try:
                f = float(text)

                if(f < 0):
                    self.outer.now_log_out = "輸入不能為負數"
                    return
                self.outer.sv.detect_interval_time = f
                self.outer.now_log_out = "更改偵測間隔為 " +text+" 秒"
            except ValueError:
                self.outer.now_log_out = '輸入僅能為整數或小數點'
        def change_write_interval(self,text):
            try:
                f = float(text)
                if(f < 3):
                    self.outer.now_log_out = "寫入間隔不能低於3秒"
                    return
                if(f < 0):
                    self.outer.now_log_out = "輸入不能為負數"
                    return
                self.outer.sv.write_interval_time = f
                self.outer.now_log_out = "更改輸出表格間隔為 " +text+" 秒"
            except ValueError:
                self.outer.now_log_out = '輸入僅能為整數或小數點'


        def change_detect_id_refresh_time(self,text):
            try:
                f = int(text)

                if(f == 1):
                    self.outer.now_log_out = "刷新秒數僅能為大於1之整數"
                    return
                elif(f < 1):
                    self.outer.now_log_out = "輸入不能為負數"
                    return
                self.outer.unique_refresh_time = f
                self.outer.now_log_out = "獨立ID重新判別刷新秒數更改為 " +text+" 秒"
            except ValueError:
                self.outer.now_log_out = '輸入僅能為整數'
        
        def load_model_config(self):
            #讀取模型參數之JSON檔
            print("load model config")
            self.outer.sv.model_cfg_json_path = QFileDialog.getOpenFileName(None,'Open a file', '',
                                         "JSON (*.json)")
            self.outer.sv.model_cfg_json_path = self.outer.sv.model_cfg_json_path[0]
            self.label_20.setText(os.path.basename(self.outer.sv.model_cfg_json_path))
            try:
                with open(self.outer.sv.model_cfg_json_path, 'r') as f:
                    tmp = "讀取模型檔案成功，參數如下：\n"
                    tmp += self.outer.sv.model_cfg_json_path +'\n'
                    self.outer.args = self.outer.parser.parse_args()
                    try :
                        self.outer.args.__dict__ = json.load(f)
                        tmpstr = str(self.outer.args.__dict__)
                        tmpstr = tmpstr.replace(',',',\n')
                        tmp += tmpstr
                        self.outer.now_log_out = tmp
                    except:
                        tmp += "讀取模型檔案失敗，格式不符合：\n"
                        tmp += '將採用初始設置... \n'
                        tmpstr = str(self.outer.args.__dict__)
                        tmpstr = tmpstr.replace(',',',\n')
                        tmp += tmpstr
                        self.outer.now_log_out = tmp
            except EnvironmentError:
                print("error model...")
                time.sleep(1)
                self.outer.args = self.outer.parser.parse_args()
                tmp = "讀取模型檔案失敗，採用預設設置\n"
                tmpstr = str(self.outer.args.__dict__)
                tmpstr = tmpstr.replace(',',',\n')
                tmp += tmpstr
                self.outer.now_log_out = tmp

        def set_output_format(self):
            #傳字串 交由外部THREAD讀取 開始偵測時DISABLE
            u = self
        def sound_effect_path(self):
            filename,_ = QtWidgets.QFileDialog.getOpenFileName(None, 
            "Open a media file", "", "wav (*.wav)")
            self.outer.sv.sound_effect = filename
            self.label_7.setText(os.path.basename(filename))
            
        def sound_effect_checkbox(self):
            self.outer.sv.use_sound_effect = not self.outer.sv.use_sound_effect
            if(self.outer.sv.sound_effect is None):
                self.outer.now_log_out = "請先指定音效檔案路徑！"
                self.outer.sv.use_sound_effect = False
                self.checkBox_2.setChecked(False)
            else:
                if(self.outer.sv.use_sound_effect):
                    #print(self.outer.sv.sound_effect)
                    mixer.init()
                    try:
                        mixer.music.load(self.outer.sv.sound_effect)
                        mixer.music.play()
                    except:
                        self.outer.sv.use_sound_effect = not self.outer.sv.use_sound_effect
                        self.checkBox_2.setChecked(False)
                        time.sleep(0.3)
                        self.outer.now_log_out = "\n音效檔案遺失，請重新指定"
                
        def csv_path_setting(self):
            filename,_ = QtWidgets.QFileDialog.getSaveFileName(None, 
            "Save a csv File", "", "Data File (*.xlsx *.csv *.dat);; Excel File (*.xlsx *.xls)")
            self.outer.sv.csv_w_path = filename
            self.label_25.setText(os.path.basename(filename))
            
            

        def save_setting(self):
            filename,_ = QtWidgets.QFileDialog.getSaveFileName(None, 
            "Save File", "", "JSON (*.json)")
            str_ = ''
            self.get_savable_ui_var()
            print("Wth")
            print(self.outer.sv.exe_title)
            with io.open(filename, 'w', encoding='utf8') as outfile:
                str_ =json.dumps(self.outer.sv, indent = 4, default=lambda o: o.__dict__)
                #str_ = self.outer.sv_tojson
                outfile.write(str_)
            print(str_)
        
        def load_setting(self):
            tmp_setting_path,_ = QFileDialog.getOpenFileName(None,'Open a file', '',
                                         "JSON (*.json)")
            with open(tmp_setting_path) as json_data:
                data_dict = json.load(json_data)
            #print(data_dict)
            tmp_dump_dict = json.dumps(data_dict, indent=4)
            #print(tmp_dump_dict)
            #self.outer.sv = self.outer.savable_var(**data_dict)
            self.outer.sv = json.loads(tmp_dump_dict,object_hook=lambda d: Namespace(**d))
            self.setting_savable_ui_var()
            self.outer.now_log_out = '\n 讀取設定檔完成 ：\n' + tmp_setting_path +"\n"
            self.label_2.setText(os.path.basename(tmp_setting_path))
           
            #print(self.outer.sv_tojson)    
            #s = json.dumps(self.outer.__dict__)
            #print(s) 
        def model_parameter_setting(self):
            u=self
        
        def setting_savable_ui_var(self):
            self.lineEdit.setText(str(self.outer.sv._log_time))
            #show streaming opencv float windows
            
            # 候選框id過幾秒後會被重新識別成獨立之ID (ex: 5號id在180內被視為同一人 180秒之後會被重新計算一次並持續loop該邏輯)
            #self.outer.sv.unique_refresh_time = 180
            self.checkBox.setChecked(self.outer.sv._show_video_streaming)
            self.checkBox_4.setChecked(self.outer.sv.g_isheadless)
            self.checkBox_2.setChecked(self.outer.sv.use_sound_effect)
            self.checkBox_3.setChecked(self.outer.sv.csv_Writing)
            if(self.outer.sv.csv_Writing == True):
                # init_thread so we need to reclick two time
                self.outer.csv_checkbox()  #false
                self.outer.csv_checkbox()  #true and start thread
            #output format, user can define log out output format
            self.lineEdit_3.setText(str(self.outer.sv.output_format_string)) 
            self.lineEdit_5.setText(str(self.outer.sv.img_size_zoom))
            self.lineEdit_7.setText(str(self.outer.sv.detect_interval_time))
            self.lineEdit_6.setText(str(self.outer.sv.write_interval_time)) 
            #self.lineEdit_8.setText(str(self.outer.sv.write_csv_format))
            self.lineEdit_9.setText(str(self.outer.sv.unique_refresh_time))
            self.lineEdit_10.setText(str(self.outer.sv.sound_threshold))
            self.label_7.setText(os.path.basename(self.outer.sv.sound_effect))
            if(self.outer.sv.csv_w_path is not None):
                self.label_25.setText(os.path.basename(self.outer.sv.csv_w_path))
            self.label_20.setText(os.path.basename(self.outer.sv.model_cfg_json_path))
            # self.outer.sv.sound_effect = 'path/'
            # self.outer.sv.now_detect_count = 0
            # self.outer.sv.avg_detect_count = 0.0
            # self.outer.sv.total_detect_count = 0
            #self.outer.sv.date_string = ''
            self.lineEdit_4.setText(str(self.outer.sv.exe_title))
            #self.outer.svlf.model_cfg_json_path = 'yolov5_model_args.txt'
             
            self.lineEdit_2.setText(str(self.outer.sv.input_url))

        def get_savable_ui_var(self):
            self.outer.sv._log_time = int(self.lineEdit.text())
            #show streaming opencv float windows
            self.outer.sv._show_video_streaming = self.checkBox.isChecked()
            # 候選框id過幾秒後會被重新識別成獨立之ID (ex: 5號id在180內被視為同一人 180秒之後會被重新計算一次並持續loop該邏輯)
            self.outer.sv.use_sound_effect = self.checkBox_2.isChecked()
            self.outer.sv.g_isheadless = self.checkBox_4.isChecked()
            #output format, user can define log out output format
            self.outer.sv.output_format_string = self.lineEdit_3.text() 
            self.outer.sv.img_size_zoom = float(self.lineEdit_5.text())
            self.outer.sv.detect_interval_time = int(self.lineEdit_7.text())
            self.outer.sv.write_interval_time = int(self.lineEdit_6.text())
            #self.outer.sv.write_csv_format = self.lineEdit_8.text()
            self.outer.sv.sound_threshold = int(self.lineEdit_10.text())
            # self.outer.sv.sound_effect = 'path/'
            self.outer.sv.unique_refresh_time = int(self.lineEdit_9.text())
            # self.outer.sv.now_detect_count = 0
            # self.outer.sv.avg_detect_count = 0.0
            # self.outer.sv.total_detect_count = 0
            #self.outer.sv.date_string = ''
            self.outer.sv.exe_title = self.lineEdit_4.text()
            #print("???")
            #print(self.outer.sv.exe_title)
            #self.outer.svlf.model_cfg_json_path = 'yolov5_model_args.txt'
            
            self.outer.sv.csv_Writing = self.checkBox_3.isChecked()
            self.outer.sv.input_url = self.lineEdit_2.text()
            
    #--------不可替代之邏輯FUNCTION-----------
    #region ui button function
    def show_vid_checkbox(self):
        self.sv._show_video_streaming = not self.sv._show_video_streaming
    
    def headless_checkbox(self):
        self.sv.g_isheadless = not self.sv.g_isheadless     
    #endregion
    
    #region YOLOV5 原生FUNCTION 用於判斷 BBOX的位置
    def xyxy_to_xywh(self,*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h
    def xyxy_to_tlwh(self,bbox_xyxy):
        tlwh_bboxs = []
        for i, box in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in box]
            top = x1
            left = y1
            w = int(x2 - x1)
            h = int(y2 - y1)
            tlwh_obj = [top, left, w, h]
            tlwh_bboxs.append(tlwh_obj)
        return tlwh_bboxs
    def compute_color_for_labels(self,label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)
    #endregion
    def total_unique_list_update(self,listOfElems,nowtime):
        ''' Check if given list contains any duplicates '''  
        '''一個global DICTIONARY 一段時間後重新記錄unique id'''  
        for id in listOfElems:
            if id in self.total_unique_id_dict:
                #print("#total unique 1")
                for _key in self.total_unique_id_dict.keys():
                    #print("#total unique 2")
                    if _key == id:
                        #print("#total unique 3")
                        id_record_time_s = self.total_unique_id_dict[id]
                        id_record_time_s = id_record_time_s.replace("_", "")
                        id_record_time = float(id_record_time_s)
                        if(abs(id_record_time-nowtime) >= self.sv.unique_refresh_time):
                            self.total_unique_id_dict.pop(_key,None)
                            self.sv.total_detect_count += 1
                            d = {_key:"_"+str(nowtime)}
                            self.total_unique_id_dict.update(d)
                        continue    
                continue
            else:
                self.sv.total_detect_count += 1
                d = {id:"_"+str(nowtime)}
                self.total_unique_id_dict.update(d)
    
    def checkIfDuplicates(self,listOfElems):
        ''' Check if given list contains any duplicates ''' 
        #print("checkIfDuplicates")
        #print(listOfElems)   
        for elem in listOfElems:
            if elem in self.g_unique_id_list:
                continue
            else:
                self.g_unique_id_list.append(elem)  
        #print(self.g_unique_id_list)
    #url detect function
    def detect_function(self,opt,mode,url):
        out, source, weights, show_vid, save_vid, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.show_vid, opt.save_vid, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith(
            'rtsp') or source.startswith('http') or source.endswith('.txt')
        self.sv.total_detect_count = 0
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
                
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA
        #imgsz *= self.img_size_zoom
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16

        if show_vid:
            show_vid = check_imshow()
        names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        
        #region 爬蟲用變數
        location = None
        size = None
        src = None
        #endregion
        
        #start_time = time_synchronized()
        #region 區域截屏用變數
        sx=None
        sy=None
        ex=None
        ey=None
        monitor_number = opt.monitor_num
        sct = None
        monitor_info = None

        
        
        #endregion
        if  ((mode == "yt") or (mode == "cctv")):
            #region 初始化 selenium webdriver and url 判別 (水利署與公路總局)
            options = Options()
            if(self.sv.g_isheadless is True):
                options.add_argument('-headless')
            #採用無頭模式的FIREFOX 不採用chrome 因為無頭模式下有問題
            
            self.g_webdriver = webdriver.Firefox(executable_path=GeckoDriverManager().install(),options=options)
            #now_log_out = logging.getLogger("WDM")
            self.g_webdriver.get(url)
            
            if(mode == "yt"):
                try:
                    WebDriverWait(self.g_webdriver, 15).until(EC.element_to_be_clickable(
                    (By.XPATH, "//button[@class='ytp-large-play-button ytp-button']"))).click()
                    # WebDriverWait(driver,5).until(EC.element_to_be_clickable((By.CLASS_NAME,'ytp-iv-video-content')))
                except TimeoutException as ex:
                    print('out of time')
                    sys.exit(1)
                # time.sleep(10)

                youtube_panel_xy = None
                retrycount = 0
                retrymax_count = 10
                while(youtube_panel_xy is None):
                    self.now_log_out = "搜尋YOUTUBE 視窗 (Retry time : " + str(retrycount) + " / " + str(retrymax_count) + "\n"
                    retrycount+=1
                    time.sleep(0.5)
                    if(retrycount == retrymax_count):
                        self.now_log_out = "超過重新尋找次數，請重新開啟程序..."
                        break
                    try:
                        WebDriverWait(self.g_webdriver, 15).until(EC.element_to_be_clickable(
                        (By.XPATH, "//div[@class='ytp-iv-video-content']")))
                        youtube_panel_xy = self.g_webdriver.find_element_by_xpath("//div[@class='ytp-iv-video-content']")
                    except:
                        try:
                            WebDriverWait(self.g_webdriver, 15).until(EC.element_to_be_clickable(
                            (By.XPATH, "//div[@id='player-container-inner']")))
                            print("'ytp-iv-video-content' not found try find player-api")
                            youtube_panel_xy = self.g_webdriver.find_element_by_xpath("//div[@id='player-container-inner']")
                        except: 
                            self.now_log_out = " 找不到YOUTUBE 撥放器HTML，請嘗試重新讀取"
                    location = youtube_panel_xy.location
                    size  = youtube_panel_xy.size
            elif(mode == 'cctv'):
                if 'cctv.aspx' in url :        
                    self.g_webdriver.get(url)
                    #time.sleep(10)

                    try:
                        WebDriverWait(self.g_webdriver,10).until(EC.element_to_be_clickable((By.ID,'frmMain')))
                    except TimeoutException as ex:
                        print(ex.message)
                    iframe_target = self.g_webdriver.find_element_by_xpath('//iframe[@id="frmMain"]')
                    src = iframe_target.get_attribute('src')
                    print(src)
                    url = src
                    self.g_webdriver.get(src)
                
                try:
                    WebDriverWait(self.g_webdriver,10).until(EC.element_to_be_clickable((By.ID,'cctv_image')))
                except TimeoutException as ex:
                    print(ex.message)

                img = self.g_webdriver.find_element_by_xpath('//div[@id="cctv_image"]/img')
                location = img.location
                size  = img.size
                # print('l:',location)
                # print('s:',size)
                #png = driver.get_screenshot_as_png()
                src = img.get_attribute('src')
            
            #size  = youtube_panel_xy.size
            print('l:',location)
            print('s:',size)
                #png = driver.get_screenshot_as_png()
            #src = img.get_attribute('src')
        frame_idx = 0
        start_time = time_synchronized()
        #endregion
        #for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if (mode == "area"):
            sx=int(self.srs.box_area[0])
            sy=int(self.srs.box_area[1])
            ex=int(self.srs.box_area[2])
            ey=int(self.srs.box_area[3])
            sct = mss()
            monitor_info = {'top' : sy, 'left' : sx, 'width' : ex-sx, 'height' : ey-sy,
            "mon": monitor_number
            }
            mon = sct.monitors[monitor_number]

        self.sv.total_detect_count = 0
        self.total_unique_id_dict.clear()

        last_img_size = self.sv.img_size_zoom
        sct_img = None
        src_img_h ,src_img_w = (0,0) 
        tt1=0
        tt2=0
        self._time_counter = 0
        while (self.src_state == 1):  
            dt = datetime.now()
            tt1 = time.time()
            time.sleep(self.sv.detect_interval_time)
            # if(detecting==True):
            #     break
            frame_idx = frame_idx + 1

            
            if (mode == "yt"):
                try:
                    png = self.g_webdriver.get_screenshot_as_png()
                except WebDriverException:
                    self.now_log_out = "使用者關閉瀏覽器..."
                    break
                nparr = numpy.frombuffer(png, numpy.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                left = location['x']
                top = location['y']
                right = location['x'] + size['width']
                bottom = location['y'] + size['height']
                sct_img = img[top:int(bottom), left:int(right)]
            elif (mode=='cctv'):
                if 'jpg'in src or 'jpeg'in src or 'png'in src or 'JPEG'in src or 'JPG' in src:
                    # 水利署影像為圖片檔，直接進行讀取
                    try:
                        img = self.g_webdriver.find_element_by_xpath('//div[@id="cctv_image"]/img')
                    except WebDriverException:
                        self.now_log_out = "使用者關閉瀏覽器..."
                        break
                    src = img.get_attribute('src')
                    try:
                        req = urllib.request.urlopen(src)
                    except :
                        self.now_log_out = '請求網頁連結失敗，重新讀取'
                        continue
                    arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
                    sct_img = cv2.imdecode(arr, -1)
                else:
                    # 公路總局影像非圖片檔，直接使用擷取圖片
                    png = self.g_webdriver.get_screenshot_as_png()
                    nparr = numpy.frombuffer(png, numpy.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    left = location['x']
                    top = location['y']
                    right = location['x'] + size['width']
                    bottom = location['y'] + size['height']
                    sct_img = img[top:int(bottom), left:int(right)]
            elif (mode == 'area'):
                sct_img = sct.grab(monitor_info)
                sct_img = numpy.array(sct_img)
                sct_img = cv2.cvtColor(sct_img, cv2.COLOR_RGBA2RGB)
            
            if(src_img_h  == 0):
                #print(img.shape)
                src_img_h,src_img_w,_ = sct_img.shape
            #放大倍率功能 同時放大模型所需的內建imgsz 
            if(self.sv.img_size_zoom >= 1):
                #print("放大倍率?",imgsz )
                imgsz = opt.img_size
                imgsz = int(imgsz *self.sv.img_size_zoom)
                tmph = int(src_img_h * self.sv.img_size_zoom)
                tmpw = int(src_img_w * self.sv.img_size_zoom)
                #print(img.shape)
                try:
                    sct_img = cv2.resize(sct_img,(tmpw,tmph),interpolation=cv2.INTER_CUBIC)
                except:
                    continue

            dataset = LoadScreen_Capture(sct_img,img_size=imgsz)
            time.sleep(0.01)
            img, im0 ,_ = dataset.__next__()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            #print(img.shape)
            #print(img.mode)
            # Inference
            t1 = time_synchronized()
            
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()
            self._time_counter += abs(tt2-tt1)
            #print(self._time_counter)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                #p, s, im0 = path, '', im0s
                s = ''
                #im0= im0s
                #s += '%gx%g ' % img.shape[2:]  # print string
                #save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    #print("unique : det[:, -1].unique()")
                    # Print results
                    for c in det[:, -1].unique():
                        
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                        #print("in ? ",s)

                    xywh_bboxs = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # to deep sort format
                        x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(*xyxy)
                        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
                        xywh_bboxs.append(xywh_obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(xywh_bboxs)
                    confss = torch.Tensor(confs)
                    
                    


                    for redetect in range(3):
                    # pass detections to deepsort
                        outputs = deepsort.update(xywhs, confss, im0)
                    
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        #print("output?")
                        #print(outputs)
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.draw_boxes(im0, bbox_xyxy, identities)
                        
                        # to MOT format
                        tlwh_bboxs = self.xyxy_to_tlwh(bbox_xyxy)

                        # Write MOT compliant results to file
                        for j, (tlwh_bbox, output) in enumerate(zip(tlwh_bboxs, outputs)):
                            # bbox_top = tlwh_bbox[0]
                            # bbox_left = tlwh_bbox[1]
                            # bbox_w = tlwh_bbox[2]
                            # bbox_h = tlwh_bbox[3]
                            # identity = output[-1]
                            self.checkIfDuplicates(identities)
                else:
                    s += '%g %s, ' % (0, 'person')  # add to string
                    deepsort.increment_ages()
                people_count = s    
                fps = 1/(t2 - t1)
                unique_id_list_count = len(self.g_unique_id_list)
                self.sv.now_detect_count = unique_id_list_count
                l_log_time = self.sv._log_time
                dtstring = dt.strftime( '%Y-%m-%d %H:%M:%S' )
                self.sv.date_string = dtstring
                #self.sv.output_format_string = "{dtstring}目前偵測到 {people_count} (FPS:{fps:.2f})\\n人流 ：{unique_id_list_count} / {l_log_time}秒 "
                self.sv.output_format_string = self.ui.lineEdit_3.text() 
                try:
                    logging_out = self.sv.output_format_string.format(**locals())
                except Exception as e :
                    print(e)
                    logging_out = "輸出格式錯誤，請重新確認"
                if(self._time_counter >= self.sv._log_time):
                    self._time_counter = 0
                    self.total_unique_list_update(self.g_unique_id_list,start_time)
                    
                    self.sv.avg_detect_count = unique_id_list_count
                    
                    self.now_log_out = logging_out
                    self.g_unique_id_list.clear()
                # Stream results
                if self.sv._show_video_streaming:
                    cv2.waitKey(1)
                    tmphei =int(im0.shape[0] * (1/self.sv.img_size_zoom))
                    tmpwid =int(im0.shape[1] * (1/self.sv.img_size_zoom))
                    dim = (tmpwid, tmphei)
                    im = cv2.resize(im0,dim)
                    cv2.imshow("show", im)                 
                else:
                    cv2.destroyAllWindows()
        tt2 = time.time()       
        cv2.destroyAllWindows()
        if(mode == 'yt' or mode == 'cctv'):
            self.g_webdriver.quit()

    
    def draw_boxes(self,img, bbox, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img
    
    #開始框選之FUNCTION
    def start_drawing(self):
        self.srs = ScreenShot()
        try:
            sx=int(self.srs.box_area[0])
            sy=int(self.srs.box_area[1])
            ex=int(self.srs.box_area[2])
            ey=int(self.srs.box_area[3])
        except:
            print("out of range")
            sys.exit()
        
        # draw_area_thread = threading.Thread(target = self.draw_area_box,args=(sx,sy,ex,ey))
        # draw_area_thread.setDaemon(True)
        # draw_area_thread.start()
    #endregion
    #region UI 功能
    #UI BUTTON FUNCTION 執行區域截圖
    def screen_shot(self,_ui):
        

        if self.src_state == 0:
            # pill2kill.set(False)
            mode = 'screenshot'
            print('state 0 : start drawing')
            self.now_log_out = "開始框選區域，框選完畢請按ENTER，重新選請按ESC"
            self.src_state =1
            print('state 0 ->',self.src_state)
            self.start_drawing()
            _translate = QtCore.QCoreApplication.translate
            _ui.pushButton_5.setText(_translate("MainWindow", "停止偵測"))
            _ui.pushButton_6.setEnabled(False)
            self.ui_locker(_ui,False)
            with torch.no_grad():
                drawing_thread = threading.Thread(target =  self.detect_function,args=(self.args,"area",None,))
                drawing_thread.setDaemon(True)
                drawing_thread.start()
        elif (self.src_state == 1):
            mode ='default'
            print('state 1 start detection ?')
            _translate = QtCore.QCoreApplication.translate
            self.now_log_out = "結束框選"
            verScrollBar = _ui.textBrowser.verticalScrollBar()
            verScrollBar.setValue(verScrollBar.maximum()) # Scrolls to the bottom
            _ui.pushButton_5.setText(_translate("MainWindow", "螢幕擷取"))
            self.ui_locker(_ui,True)
            _ui.pushButton_6.setEnabled(True)
            self.sv.img_size_zoom = 1
            _ui.lineEdit_5.setText(str(self.sv.img_size_zoom))
            self.src_state = 0
    #退出BUTTON    
    def exit_btn(self):
        if(self.g_webdriver is not None):
            print("im close")
            self.g_webdriver.quit()
            self.g_webdriver = None
        sys.exit()
    #使用者輸入 打印時間之FUNCTION
    def get_log_time_chaged(self,text):
        print("get changed text : ", text)
        try:
            float(text)
            self.sv._log_time = float(text)
        except ValueError:
            print ("Not a float")
    #是否顯示 VIDEO CHECKBOX之FUNCTION

    #URL BUTTON FUNCTION
    def url_detecting_thread(self,_ui):
        print("check in")
        if self.src_state == 0:

            #print('state 0 : start drawing')
            self.src_state =1
            #print('state 0 ->',self.src_state)
            _translate = QtCore.QCoreApplication.translate

            url= _ui.lineEdit_2.text()
            url = url.strip('\n')
            url = url.strip('\t')
            url = url.strip('\r')
            if(url == url.startswith('rtsp') or url.startswith('http') or url.endswith('.txt')):
                _ui.pushButton_6.setText(_translate("MainWindow", "停止偵測"))
                self.ui_locker(_ui,False)
                _ui.pushButton_5.setEnabled(False)
                #_ui.textBrowser.append("偵測目標連結："+url)
                self.now_log_out = "偵測目標連結："+url
                if 'youtube' in url or'youtu.be'in url :
                    now_log_out = '偵測到youtube網址'
                # url_detect(args,url)
                    with torch.no_grad():
                        url_thread = threading.Thread(target =  self.detect_function,args=(self.args,"yt",url,))
                        url_thread.setDaemon(True)
                        url_thread.start()
                elif 'cctv'in url or 'CCTV' in url:
                    self.now_log_out = '偵測到cctv串流網址'
                    with torch.no_grad():
                        url_thread = threading.Thread(target =  self.detect_function,args=(self.args,"cctv",url,))
                        url_thread.setDaemon(True)
                        url_thread.start()
                    
            else:
                #_ui.textBrowser.append("連結不符合格式(http or rtsp)",url)
                self.now_log_out = "偵測目標連結："+url
                self.src_state = 0
                return
        elif (self.src_state == 1):
            print('state 1 start detection ?')
            # pill2kill.set()
            mode = 'default'
            _translate = QtCore.QCoreApplication.translate
            verScrollBar = _ui.textBrowser.verticalScrollBar()
            verScrollBar.setValue(verScrollBar.maximum())
            _ui.pushButton_6.setText(_translate("MainWindow", "串流網址"))
            self.ui_locker(_ui,True)
            _ui.pushButton_5.setEnabled(True)
            self.g_webdriver.quit()
            self.sv.img_size_zoom = 1
            _ui.lineEdit_5.setText(str(self.sv.img_size_zoom))
            self.src_state = 0
    #endregion
    def ui_locker(self,_ui,state):
        
        #_ui.pushButton_6.setEnabled(state)
        _ui.pushButton.setEnabled(state)
        _ui.lineEdit_6.setEnabled(state)
        #_ui.lineEdit_8.setEnabled(state)
        _ui.pushButton_4.setEnabled(state)
        _ui.pushButton.setEnabled(state)
        _ui.lineEdit_4.setEnabled(state)
        _ui.pushButton_8.setEnabled(state)
        _ui.pushButton_9.setEnabled(state)

    def csv_checkbox(self):
        #print("csv check?")
        self.sv.csv_Writing = not self.sv.csv_Writing
        if (self.sv.csv_w_path and not self.sv.csv_w_path.isspace()):
            if self.sv.csv_Writing:
                #print("check in2:")
                if not os.path.exists(self.sv.csv_w_path):
                    print("create new csv... : " + self.sv.csv_w_path)
                    open(self.sv.csv_w_path, 'w',encoding='utf-8').close()
                self.csv_wr_thread_instance = threading.Thread(target =  self.csv_writing_thread)
                self.csv_wr_thread_instance.setDaemon(True)
                self.csv_wr_thread_instance.start()
        else :
            print("csv存檔尚未指定路徑")
            self.now_log_out = "csv存檔尚未指定路徑！"
            self.ui.checkBox_3.setChecked(False)
            self.sv.csv_Writing = False

           

    def csv_writing_thread(self):
        
        split_format = []
        split_format = self.sv.write_csv_format.split("|")
        result_col = []
        result_row = []
        
        tmp_time_counter = 0
        while(self.sv.csv_Writing):
            t1 = time.time()
            time.sleep(0.01)
            #print("thread 1...")
            if(self.src_state == 1):
                #print("thread 2...")
                if(tmp_time_counter >= self.sv.write_interval_time):
                    tmp_time_counter = 0
                    for ele in split_format:
                        if ele == 'Time':
                            result_col.append('Time')
                            result_row.append(self.sv.date_string)
                        elif ele == 'Title':
                            result_col.append('Title')
                            result_row.append(self.sv.exe_title)
                        elif ele == 'Count':
                            result_col.append('Count')
                            result_row.append(self.sv.now_detect_count)
                        elif ele == 'AvgCount':
                            result_col.append('AvgCount')
                            result_row.append(self.sv.avg_detect_count)
                        elif ele == 'TotalCount':
                            result_col.append('TotalCount')
                            result_row.append(self.sv.total_detect_count)
                    df = pd.DataFrame(columns =result_col , data = [result_row] )
                    try:
                        with open(self.sv.csv_w_path, mode = 'a+',encoding='utf-8') as f:
                            df.to_csv(f, header=f.tell()==0,index = False)
                    except:
                        result_col.clear()
                        result_row.clear()
                        #self.now_log_out = "請關閉CSV檔案，關閉後將繼續寫入..."
                        continue
                    #print(result_row)
                    
                    result_col.clear()
                    result_row.clear()
            t2 =  time.time()
            tmp_time_counter += (t2-t1)
        self.csv_wr_thread_instance = None

    
@atexit.register
def delete_webdriver():
    #print(Main_Tracking.sv.exe_title+" close")
    if(hasattr(Main_Tracking,'g_webdriver')):
        if(Main_Tracking.g_webdriver is not None):
            #print(Main_Tracking.sv.exe_title+" explorer close")
            Main_Tracking.g_webdriver.quit()
        

    # main function

from multiprocessing import Process
import multiprocessing
def create_process(id,path):
    Main_Tracking(id,saved_file_init_load=path)
if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--json-list-path', type=str, default=None)
    # args = parser.parse_args()
    json_list_path='saved_setting/saved_group.json'
    multiprocessing.freeze_support()
    # ex: saved_group.json
    #{
    # 0:"saved_setting\八德路3段74號.json"
    # 1:"J:\master_1_down\yolo\crowd_detect-20210701T114137Z-001\crowd_detect\saved_setting\default.json"
    # ...
    # }
    #可以使用相對路徑或絕對路徑
    
    data_dict = None
    with open(json_list_path,encoding="utf-8") as json_data:
        data_dict = json.load(json_data)
        print(data_dict)
    procs = []
    for key, value in data_dict.items():
        proc = Process(target=create_process , args = (key,value,))
        print("in :"+value)
        procs.append(proc)
        proc.start()
    for p in procs:
        p.join()

