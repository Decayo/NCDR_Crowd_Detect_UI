import sys,os
from PyQt5 import QtCore, QtWidgets
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'J:\Anaconda\envs\Pytorch_3,6\Lib\site-packages\PyQt5\Qt5\plugins'

import sys
from random import randint
from multiprocessing import Process

from main_tracking import *
def create_process(): 
    Main_Tracking("main")
 
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
procs = [] 
if __name__=='__main__':
    class AnotherWindow(QWidget):
        """
        This "window" is a QWidget. If it has no parent,
        it will appear as a free-floating window.
        """
        def __init__(self):
            super().__init__()
            layout = QVBoxLayout()
            self.label = QLabel("Another Window % d" % randint(0, 100))
            layout.addWidget(self.label)
            self.setLayout(layout)


    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.window1 = AnotherWindow()
            self.window2 = AnotherWindow()

            l = QVBoxLayout()
            button1 = QPushButton("Push for Window 1")
            button1.clicked.connect(
                lambda checked: self.toggle_window(self.window1)
            )
            l.addWidget(button1)

            button2 = QPushButton("Push for Window 2")
            button2.clicked.connect(
                lambda checked: self.toggle_window(self.window2)
            )
            l.addWidget(button2)

            w = QWidget()
            w.setLayout(l)
            self.setCentralWidget(w)

        def toggle_window(self, window):
            proc = Process(target=create_process)
            proc.start()
            procs.append(proc)
            if window.isVisible():
                window.hide()

            else:
                window.show()


    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()