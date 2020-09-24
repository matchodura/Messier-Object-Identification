import io
import random
import sys
import os
cwd = os.getcwd()
sys.path.insert(0,cwd + '\lib')
import PIL
import time
import cv2
import logging
# import PIL
import keras
import os
import glob
import gc
# import tensorflow.python.keras.engine.base_layer_v1
# import tensorflow.python.ops.numpy_ops
import pandas as pd
import numpy as np
import threading
import matplotlib
matplotlib.use('QT5Agg')
from multiprocessing import Queue
from PIL import Image, ImageOps, ImageQt
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtGui import QPainter, QIntValidator, QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QAbstractButton, QLineEdit, QApplication, QWidget
from PyQt5.QtCore import *
from tensorflow.keras.preprocessing import image
from keras import backend as K
from keras import activations, initializers, regularizers, constraints, metrics
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, Layer,
                          BatchNormalization, LocallyConnected2D,
                          ZeroPadding2D, Conv2D, MaxPooling2D, Conv2DTranspose,
                          GaussianNoise, UpSampling2D, Input, InputSpec, GlobalAveragePooling2D)
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras.models import load_model
from io import StringIO
from contextlib import redirect_stdout
from mainwindow import Ui_MainWindow
from popup_resize import Ui_Resize
from popup_rotation import Ui_Rotation
from popup_noise import Ui_Noise
from popup_model import Ui_Model
from model_creation import model_setup
from pandas import ExcelFile, ExcelWriter, isnull
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.metrics import classification, classification_report, confusion_matrix
matplotlib.use('QT5Agg')
os.getcwd()
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class PicButton(QAbstractButton):
    def __init__(self, pixmap, pixmap_hover, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap
        self.pixmap_hover = pixmap_hover
        # self.pixmap_pressed = pixmap_pressed

        self.pressed.connect(self.update)
        # self.released.connect(self.update)

    def paintEvent(self, event):
        pix = self.pixmap_hover if self.underMouse() else self.pixmap
        # if self.isDown():
        #     pix = self.pixmap_pressed

        painter = QPainter(self)
        painter.drawPixmap(event.rect(), pix)

    def enterEvent(self, event):
        self.update()

    def leaveEvent(self, event):
        self.update()

    def sizeHint(self):
        return QSize(32, 32)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        # print(self.fn)
        self.fn(*self.args, **self.kwargs)


class PopUpResize(QtWidgets.QDialog, Ui_Resize):
    def __init__(self, imageDims, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setFixedSize(340, 280)
        self.imageDims = imageDims
        self.label_ImageSize.setText(str(self.imageDims))
        self.onlyInt = QIntValidator()
        self.lineEditHeight.setValidator(self.onlyInt)
        self.lineEditWidth.setValidator(self.onlyInt)

    def values(self):
        self.width = self.lineEditWidth.text()
        self.height = self.lineEditHeight.text()
        self.Dimensions = (self.width, self.height)
        return self.Dimensions


class PopUpRotation(QtWidgets.QDialog, Ui_Rotation):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setFixedSize(340, 280)
        self.radioButton_90.toggled.connect(self.onClicked)
        self.radioButton_180.toggled.connect(self.onClicked)
        self.radioButton_270.toggled.connect(self.onClicked)

    def onClicked(self):
        radioBtn = self.sender()
        if radioBtn.isChecked():
            self.rotationValue = radioBtn.text()

    def values(self):
        return self.rotationValue


class PopUpNoise(QtWidgets.QDialog, Ui_Noise):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setupUi(self)
        self.setFixedSize(340, 280)
        self.noiseType = 0
        self.mean = 0
        self.var = 0
        self.radioButton_Gauss.toggled.connect(self.onClicked)
        self.radioButton_Poisson.toggled.connect(self.onClicked)
        self.radioButton_Speckle.toggled.connect(self.onClicked)
        self.radioButton_SaltPepper.toggled.connect(self.onClicked)

        self.label_2.setVisible(False)
        self.label_3.setVisible(False)
        self.label_Mean.setVisible(False)
        self.label_Var.setVisible(False)
        self.horizontalSlider.setVisible(False)
        self.horizontalSlider_2.setVisible(False)

    def onClicked(self):
        radioBtn = self.sender()
        self.horizontalSlider.setSliderPosition(1)
        self.horizontalSlider_2.setSliderPosition(1)

        if radioBtn.isChecked():
            self.noiseType = radioBtn.text()
        if self.noiseType == 'Gauss':
            self.label_2.setVisible(True)
            self.label_3.setVisible(True)
            self.label_Mean.setVisible(True)
            self.label_Var.setVisible(True)
            self.horizontalSlider.setVisible(True)
            self.horizontalSlider_2.setVisible(True)
            self.horizontalSlider.setMinimum(1)
            self.horizontalSlider.setMaximum(250)
            self.horizontalSlider_2.setMinimum(1)
            self.horizontalSlider_2.setMaximum(250)
            self.label_Mean.setText(str(1))
            self.label_Var.setText(str(1))
        elif self.noiseType == 'Salt&Pepper':
            self.label_2.setVisible(True)
            self.label_2.setText('sVSp')
            self.label_3.setVisible(True)
            self.label_3.setText('amount')
            self.label_Mean.setVisible(True)
            self.label_Var.setVisible(True)
            self.horizontalSlider.setVisible(True)
            self.horizontalSlider_2.setVisible(True)
            self.horizontalSlider.setMinimum(1)
            self.horizontalSlider.setMaximum(99)
            self.horizontalSlider_2.setMinimum(1)
            self.horizontalSlider_2.setMaximum(99)
            self.label_Mean.setText(str(1 / 100))
            self.label_Var.setText(str(1 / 100))
        else:
            self.label_2.setVisible(False)
            self.label_3.setVisible(False)
            self.label_Mean.setVisible(False)
            self.label_Var.setVisible(False)
            self.horizontalSlider.setVisible(False)
            self.horizontalSlider_2.setVisible(False)

    def slider_change(self, value):

        if self.noiseType == 'Salt&Pepper':
            self.mean = value / 100
            self.label_Mean.setText(str(self.mean))
        else:
            self.mean = value
            self.label_Mean.setText(str(self.mean))

    def slider_change_2(self, value):
        self.var = 1
        if self.noiseType == 'Salt&Pepper':
            self.var = value / 100
            self.label_Var.setText(str(self.var))
        else:
            self.var = value
            self.label_Var.setText(str(self.var))

    def values(self):
        self.vals = [self.noiseType, self.mean, self.var]
        return self.vals


class PopUpModel(QtWidgets.QDialog, Ui_Model):
    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setFixedSize(580, 525)
        self.setWindowTitle("Budowa sieci ")
        text = open(path).read()
        self.textEdit.setPlainText(text)
        #
        # pixmap = QtGui.QPixmap(filepath)
        #
        # self.labelModel.setPixmap(pixmap)


class MyStream(QtCore.QObject):
    message = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(MyStream, self).__init__(parent)

    def write(self, message):
        self.message.emit(str(message))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        self.setFixedSize(QSize(960, 520))
        self.sciezka = os.path.dirname(os.path.realpath(__file__))
        self.setupUi(self)
        self.currentObject = 0
        self.epochsNumber = 10
        self.BS = 20
        self.read_excel()
        self.setWindowTitle("Identyfikacja obiektów Messiera ")
        self.setWindowIcon(QtGui.QIcon(self.sciezka + r'\images\icons\M1.ico'))
        # worker_redirect = Worker(self.console_redirecting)
        # self.threadpool.start(worker_redirect)

        self.katalog_obiektow()

        self.imageOpened = False
        self.modelOpened = False
        self.ImageOn = False
        self.ImageClassifierPath = ''
        self.ImageProcessingPath = ''
        self.activeProcessing = False
        self.activeSaving = False
        self.imageType = ''
        self.dataOK = False
        self.trainPath = ''
        self.log_path = ''
        # przyciski wyboru okna
        self.pushButton_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.pushButton_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.pushButton_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.pushButton_4.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.pushButton_5.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(4))
        self.pushButton_6.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(5))

        self.pushButton_ChooseFolder.clicked.connect(lambda: self.open_folder(1))
        self.pushButton_ChoosePath.clicked.connect(lambda: self.open_folder(2))

        self.pushButton_GenerateFolders.clicked.connect(lambda: self.generate_folders())

        self.pushButton_Resizing.clicked.connect(lambda: self.open_popup(self.activeProcessing, 0))
        self.pushButton_Noise.clicked.connect(lambda: self.open_popup(self.activeProcessing, 1))
        self.pushButton_Rotation.clicked.connect(lambda: self.open_popup(self.activeProcessing, 2))
        self.pushButton_BW.clicked.connect(lambda: self.open_popup(self.activeProcessing, 3))
        self.pushButton_Mirror.clicked.connect(lambda: self.open_popup(self.activeProcessing, 4))
        self.pushButton_SaveImage.clicked.connect(lambda: self.open_popup(self.activeProcessing, 5))

        pixmap_1 = QtGui.QPixmap(self.sciezka + r'\images\help\struktura_1.png')
        self.label_FolderStructure_1.setPixmap(pixmap_1)
        # print(self.sciezka)
        pixmap_2 = QtGui.QPixmap(self.sciezka + r'\images\help\struktura_2.png')
        self.label_FolderStructure_2.setPixmap(pixmap_2)

        pixmap_helpClassifier = QtGui.QPixmap(self.sciezka + r'\images\help\pomoc_klasyfikator.png')
        self.label_helpClassifier.setPixmap(pixmap_helpClassifier)

        pixmap_helpProcessing = QtGui.QPixmap(self.sciezka + r'\images\help\pomoc_processing.png')
        self.label_helpProcessing.setPixmap(pixmap_helpProcessing)

        pixmap_helpTeaching = QtGui.QPixmap(self.sciezka + r'\images\help\pomoc_uczenie.png')
        self.label_helpTeaching.setPixmap(pixmap_helpTeaching)

        pixmap_helpClassifierCreate = QtGui.QPixmap(self.sciezka + r'\images\help\pomoc_tworzenie_klasyfikatora.png')
        self.label_helpClassifierCreate.setPixmap(pixmap_helpClassifierCreate)


        # wybór okna katalogu messiera - zawsze to samo okno
        self.pushM_1.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_4.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_5.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_6.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_7.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_8.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_9.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_10.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_11.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_12.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_13.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_14.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_15.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_16.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_17.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_18.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_19.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_20.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_21.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_22.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_23.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_24.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_25.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_26.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_27.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_28.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_29.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_30.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_31.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_32.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_33.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_34.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_35.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_36.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_37.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_38.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_39.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_40.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_41.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_42.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_43.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_44.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_45.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_46.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_47.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_48.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_49.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_50.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_51.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_52.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_53.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_54.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_55.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_56.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_57.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_58.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_59.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_60.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_61.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_62.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_63.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_64.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_65.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_66.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_67.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_68.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_69.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_70.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_71.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_72.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_73.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_74.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_75.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_76.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_77.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_78.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_79.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_80.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_81.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_82.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_83.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_84.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_85.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_86.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_87.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_88.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_89.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_90.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_91.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_92.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_93.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_94.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_95.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_96.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_97.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_98.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_99.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_100.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_101.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_102.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_103.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_104.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_105.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_106.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_107.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_108.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_109.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))
        self.pushM_110.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(6))

        # wybór obiektu Messiera - cyfra = obiekt
        self.pushM_1.clicked.connect(lambda: self.setValue(1))
        self.pushM_2.clicked.connect(lambda: self.setValue(2))
        self.pushM_3.clicked.connect(lambda: self.setValue(3))
        self.pushM_4.clicked.connect(lambda: self.setValue(4))
        self.pushM_5.clicked.connect(lambda: self.setValue(5))
        self.pushM_6.clicked.connect(lambda: self.setValue(6))
        self.pushM_7.clicked.connect(lambda: self.setValue(7))
        self.pushM_8.clicked.connect(lambda: self.setValue(8))
        self.pushM_9.clicked.connect(lambda: self.setValue(9))
        self.pushM_10.clicked.connect(lambda: self.setValue(10))
        self.pushM_11.clicked.connect(lambda: self.setValue(11))
        self.pushM_12.clicked.connect(lambda: self.setValue(12))
        self.pushM_13.clicked.connect(lambda: self.setValue(13))
        self.pushM_14.clicked.connect(lambda: self.setValue(14))
        self.pushM_15.clicked.connect(lambda: self.setValue(15))
        self.pushM_16.clicked.connect(lambda: self.setValue(16))
        self.pushM_17.clicked.connect(lambda: self.setValue(17))
        self.pushM_18.clicked.connect(lambda: self.setValue(18))
        self.pushM_19.clicked.connect(lambda: self.setValue(19))
        self.pushM_20.clicked.connect(lambda: self.setValue(20))
        self.pushM_21.clicked.connect(lambda: self.setValue(21))
        self.pushM_22.clicked.connect(lambda: self.setValue(22))
        self.pushM_23.clicked.connect(lambda: self.setValue(23))
        self.pushM_24.clicked.connect(lambda: self.setValue(24))
        self.pushM_25.clicked.connect(lambda: self.setValue(25))
        self.pushM_26.clicked.connect(lambda: self.setValue(26))
        self.pushM_27.clicked.connect(lambda: self.setValue(27))
        self.pushM_28.clicked.connect(lambda: self.setValue(28))
        self.pushM_29.clicked.connect(lambda: self.setValue(29))
        self.pushM_30.clicked.connect(lambda: self.setValue(30))
        self.pushM_31.clicked.connect(lambda: self.setValue(31))
        self.pushM_32.clicked.connect(lambda: self.setValue(32))
        self.pushM_33.clicked.connect(lambda: self.setValue(33))
        self.pushM_34.clicked.connect(lambda: self.setValue(34))
        self.pushM_35.clicked.connect(lambda: self.setValue(35))
        self.pushM_36.clicked.connect(lambda: self.setValue(36))
        self.pushM_37.clicked.connect(lambda: self.setValue(37))
        self.pushM_38.clicked.connect(lambda: self.setValue(38))
        self.pushM_39.clicked.connect(lambda: self.setValue(39))
        self.pushM_40.clicked.connect(lambda: self.setValue(40))
        self.pushM_41.clicked.connect(lambda: self.setValue(41))
        self.pushM_42.clicked.connect(lambda: self.setValue(42))
        self.pushM_43.clicked.connect(lambda: self.setValue(43))
        self.pushM_44.clicked.connect(lambda: self.setValue(44))
        self.pushM_45.clicked.connect(lambda: self.setValue(45))
        self.pushM_46.clicked.connect(lambda: self.setValue(46))
        self.pushM_47.clicked.connect(lambda: self.setValue(47))
        self.pushM_48.clicked.connect(lambda: self.setValue(48))
        self.pushM_49.clicked.connect(lambda: self.setValue(49))
        self.pushM_50.clicked.connect(lambda: self.setValue(50))
        self.pushM_51.clicked.connect(lambda: self.setValue(51))
        self.pushM_52.clicked.connect(lambda: self.setValue(52))
        self.pushM_53.clicked.connect(lambda: self.setValue(53))
        self.pushM_54.clicked.connect(lambda: self.setValue(54))
        self.pushM_55.clicked.connect(lambda: self.setValue(55))
        self.pushM_56.clicked.connect(lambda: self.setValue(56))
        self.pushM_57.clicked.connect(lambda: self.setValue(57))
        self.pushM_58.clicked.connect(lambda: self.setValue(58))
        self.pushM_59.clicked.connect(lambda: self.setValue(59))
        self.pushM_60.clicked.connect(lambda: self.setValue(60))
        self.pushM_61.clicked.connect(lambda: self.setValue(61))
        self.pushM_62.clicked.connect(lambda: self.setValue(62))
        self.pushM_63.clicked.connect(lambda: self.setValue(63))
        self.pushM_64.clicked.connect(lambda: self.setValue(64))
        self.pushM_65.clicked.connect(lambda: self.setValue(65))
        self.pushM_66.clicked.connect(lambda: self.setValue(66))
        self.pushM_67.clicked.connect(lambda: self.setValue(67))
        self.pushM_68.clicked.connect(lambda: self.setValue(68))
        self.pushM_69.clicked.connect(lambda: self.setValue(69))
        self.pushM_70.clicked.connect(lambda: self.setValue(70))
        self.pushM_71.clicked.connect(lambda: self.setValue(71))
        self.pushM_72.clicked.connect(lambda: self.setValue(72))
        self.pushM_73.clicked.connect(lambda: self.setValue(73))
        self.pushM_74.clicked.connect(lambda: self.setValue(74))
        self.pushM_75.clicked.connect(lambda: self.setValue(75))
        self.pushM_76.clicked.connect(lambda: self.setValue(76))
        self.pushM_77.clicked.connect(lambda: self.setValue(77))
        self.pushM_78.clicked.connect(lambda: self.setValue(78))
        self.pushM_79.clicked.connect(lambda: self.setValue(79))
        self.pushM_80.clicked.connect(lambda: self.setValue(80))
        self.pushM_81.clicked.connect(lambda: self.setValue(81))
        self.pushM_82.clicked.connect(lambda: self.setValue(82))
        self.pushM_83.clicked.connect(lambda: self.setValue(83))
        self.pushM_84.clicked.connect(lambda: self.setValue(84))
        self.pushM_85.clicked.connect(lambda: self.setValue(85))
        self.pushM_86.clicked.connect(lambda: self.setValue(86))
        self.pushM_87.clicked.connect(lambda: self.setValue(87))
        self.pushM_88.clicked.connect(lambda: self.setValue(88))
        self.pushM_89.clicked.connect(lambda: self.setValue(89))
        self.pushM_90.clicked.connect(lambda: self.setValue(90))
        self.pushM_91.clicked.connect(lambda: self.setValue(91))
        self.pushM_92.clicked.connect(lambda: self.setValue(92))
        self.pushM_93.clicked.connect(lambda: self.setValue(93))
        self.pushM_94.clicked.connect(lambda: self.setValue(94))
        self.pushM_95.clicked.connect(lambda: self.setValue(95))
        self.pushM_96.clicked.connect(lambda: self.setValue(96))
        self.pushM_97.clicked.connect(lambda: self.setValue(97))
        self.pushM_98.clicked.connect(lambda: self.setValue(98))
        self.pushM_99.clicked.connect(lambda: self.setValue(99))
        self.pushM_100.clicked.connect(lambda: self.setValue(100))
        self.pushM_101.clicked.connect(lambda: self.setValue(101))
        self.pushM_102.clicked.connect(lambda: self.setValue(102))
        self.pushM_103.clicked.connect(lambda: self.setValue(103))
        self.pushM_104.clicked.connect(lambda: self.setValue(104))
        self.pushM_105.clicked.connect(lambda: self.setValue(105))
        self.pushM_106.clicked.connect(lambda: self.setValue(106))
        self.pushM_107.clicked.connect(lambda: self.setValue(107))
        self.pushM_108.clicked.connect(lambda: self.setValue(108))
        self.pushM_109.clicked.connect(lambda: self.setValue(109))
        self.pushM_110.clicked.connect(lambda: self.setValue(110))

        # nawigacja między obiektami messiera
        self.pushButtonPrevious.clicked.connect(lambda: self.updateObjectDown())
        self.pushButtonNext.clicked.connect(lambda: self.updateObjectUp())
        self.pushButton_ModelGenerate.clicked.connect(lambda: self.generate_model())

        self.pushButton_BSOK.clicked.connect(lambda: self.accept_bs())
        self.pushButton_EpochOK.clicked.connect(lambda: self.accept_epochs())

        self.pushButton_SaveModel.clicked.connect(lambda: self.save_model())
        self.pushButton_PlotClear.setIcon(QtGui.QIcon(self.sciezka + r"\images\icons\arrow-circle-double.png"))
        self.pushButton_PlotClear.clicked.connect(lambda: self.clear_plots())
        self.pushButton_DisplayModel.clicked.connect(lambda: self.display_model())
        self.pushButton_SavePythonFile.clicked.connect(lambda: self.new_file_save())
        self.pushButton_SaveNewModel.clicked.connect(lambda: self.save_new_model())
        self.pushButton_DisplayModel_2.clicked.connect(lambda: self.display_model_2())
        self.model_change()



    def save_new_model(self):
        path = self.save_file("new_model", ".h5")
        self.model_new.save(path)

    def model_change(self):
        self.pathToModel = self.sciezka + r'\model_creation.py'
        file = open(self.pathToModel)
        with file:
            text = file.read()
            self.textEdit_2.setText(text)

    def new_file_save(self):
        if os.path.isfile(self.pathToModel):
            file = open(self.pathToModel, 'w')
            file.write(self.textEdit_2.toPlainText())
            file.close()

    def generate_model(self):
        self.model_new = model_setup()
        print('generating')
        filepath = self.sciezka + r'\tmp\new_model.txt'
        with open(filepath, 'w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model_new.summary(print_fn=lambda x: fh.write(x + '\n'))

            # print(self.model_new.summary())


    def display_model_2(self):
        filepath = self.sciezka + r'\tmp\new_model.txt'
        with open(filepath, 'r') as fh:
            print('test')
            print(filepath)
            self.window_new_model = PopUpModel(filepath)
            self.window_new_model.show()


    def display_model(self):
        if self.modelOpened == True:
            filepath = self.sciezka + r'\tmp\model.txt'
            with open(filepath, 'w') as fh:

                self.model.summary(print_fn=lambda x: fh.write(x + '\n'))

            self.window_model = PopUpModel(filepath)
            self.window_model.show()

    def accept_bs(self):
        self.onlyInt = QIntValidator()
        self.lineEditBS.setValidator(self.onlyInt)
        self.BS = self.lineEditBS.text()
        print('Ustawiony batch size: ' + str(self.BS))

    def accept_epochs(self):
        self.onlyInt = QIntValidator()
        self.lineEditEpoch.setValidator(self.onlyInt)
        self.epochsNumber = self.lineEditEpoch.text()
        print('Ustawiona liczba epok: ' + str(self.epochsNumber))

    def clear_plots(self):
        self.plotWidget.canvas.ax.clear()
        self.plotWidget.canvas.draw()
        self.plotWidget_2.canvas.ax.clear()
        self.plotWidget_2.canvas.draw()


    def plot_data(self):

        while True:
            time.sleep(5)

            if os.path.isfile(self.log_path) and os.path.getsize(self.log_path) > 0:

                df = pd.read_csv(self.log_path, sep=';')

                if df.empty == True:
                    print('Plik CSV jest pusty')
                elif self.modelReadyToSave:
                    break
                else:
                    epoki = df['epoch']
                    acc = df['accuracy']
                    loss = df['loss']
                    val_acc = df['val_accuracy']
                    val_loss = df['val_loss']
                    print(df)
                    self.plotWidget.canvas.ax.set_xlabel('Epoki')
                    self.plotWidget.canvas.ax.set_ylabel('Acc')
                    self.plotWidget.canvas.ax.plot(epoki + 1, acc, 'b')  # , label='train')
                    self.plotWidget.canvas.ax.plot(epoki + 1, val_acc, 'r')  # , label='val')
                    # self.plotWidget.canvas.ax.legend()
                    self.plotWidget.canvas.draw()
                    self.plotWidget_2.canvas.ax.set_xlabel('Epoki')
                    self.plotWidget_2.canvas.ax.set_ylabel('Błąd')
                    self.plotWidget_2.canvas.ax.plot(epoki + 1, loss, 'b')  # , label='train')
                    self.plotWidget_2.canvas.ax.plot(epoki + 1, val_loss, 'r')  # , label='val')
                    # self.plotWidget.canvas.ax.legend()
                    self.plotWidget_2.canvas.draw()

            else:
                print('.............................................')





    @QtCore.pyqtSlot(str)
    def on_myStream_message(self, message):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(message)

    @pyqtSlot(str)
    def append_text(self, text):
        self.textEdit.moveCursor(QtGui.QTextCursor.End)
        self.textEdit.insertPlainText(text)

    def teach_model(self):
        if self.dataOK == True and self.modelOpened == True and self.BS and self.epochsNumber:
            worker = Worker(self.teaching)
            self.threadpool.start(worker)
            worker2 = Worker(self.plot_data)
            self.threadpool.start(worker2)
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Błąd w danych lub brak załadowanego modelu!')
            error_dialog.exec_()

    def save_file(self, filename, extension):
        name = QFileDialog.getSaveFileName(self, "Select File", filename, extension)
        dest = name[0] + name[1]
        f = open(dest, "w")
        return dest

    def teaching(self):
        self.log_path = self.save_file("log", ".csv")

        self.modelReadyToSave = False
        print('Teaching process started')

        BS = int(self.BS)
        epochsNumber = int(self.epochsNumber)

        csv_logger = CSVLogger(self.log_path, append=True, separator=';')
        output_model = self.model.fit(
            self.training_flow,
            steps_per_epoch=self.training_flow.samples // BS,
            validation_data=self.validation_flow,
            validation_steps=self.validation_flow.samples // BS,
            epochs=epochsNumber,
            callbacks=[csv_logger],
            verbose=0
        )

        print("Uczenie zakończone")
        self.modelReadyToSave = True

        prediction_propability = self.model.predict(self.test_flow, steps=(self.training_flow.samples // BS) + 1)

        prediction_classes = np.argmax(prediction_propability, axis=1)

        print(classification_report(self.test_flow.classes,
                                    prediction_classes,
                                    target_names=self.test_flow.class_indices.keys()))

        report_dict = classification_report(self.test_flow.classes,
                                            prediction_classes,
                                            target_names=self.test_flow.class_indices.keys(),
                                            output_dict=True)

        dic1 = report_dict
        df = pd.DataFrame(dic1).transpose()
        df.to_excel(r"C:\Users\Mateusz\PycharmProjects\magisterka\gui\mgr\\" + "pizda_" + str(BS) + r"_classification_report.xlsx")

    def save_model(self):
        if self.modelReadyToSave:
            path = self.save_file("model", ".h5")
            self.model.save(path)
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Nie można zapisać modelu!')
            error_dialog.exec_()

    def updateObjectUp(self):

        self.currentObject = self.currentObject + 1
        newPicture = self.currentObject
        # print(newPicture)
        if newPicture == 111:
            newPicture = 1

        self.setValue(newPicture)
        # print('w gore')

    def updateObjectDown(self):

        self.currentObject = self.currentObject - 1
        newPicture = self.currentObject
        if newPicture == 0:
            newPicture = 110

        self.setValue(newPicture)

    def open_popup(self, active, type):

        number = type
        if active:
            self.imageDims = (self.widthProcessing, self.heightProcessing)
            self.popup_buttons(number)
        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Brak wybranego zdjęcia!')
            error_dialog.exec_()

    def popup_buttons(self, number):

        toolNumber = number

        if toolNumber == 0:
            print('resize')
            self.window_resize = PopUpResize(self.imageDims)
            self.window_resize.show()
            self.window_resize.pushButton_Resize.clicked.connect(self.resize_image)

        elif toolNumber == 1:
            print('szumy')
            self.window_noise = PopUpNoise()
            self.window_noise.show()
            self.window_noise.pushButton_Noise.clicked.connect(self.noise_image)

        elif toolNumber == 2:
            print('rotacje')
            self.window_rotation = PopUpRotation()
            self.window_rotation.show()
            self.window_rotation.pushButton_Rotation.clicked.connect(self.rotation_image)

        elif toolNumber == 3:
            self.grayscale_image()

            print('bw')

        elif toolNumber == 4:
            self.mirror_image()
            print('mirroring')
        elif toolNumber == 5:
            print('zapisywanie')

            self.save_image()

    def set_image_after(self, libType, image):
        self.libType = libType
        if libType == 'PIL':
            print(image)
            qim = ImageQt(image)
            pix = QtGui.QPixmap.fromImage(qim)
            print(pix)
            self.label_ImageAfter.setPixmap(pix.scaled(self.label_ImageAfter.size()))

        elif libType == 'CV2':
            print('que pasa?')

            pix = QtGui.QPixmap(self.dest_opencv)
            print(self.dest_opencv)
            self.label_ImageAfter.setPixmap(pix.scaled(self.label_ImageAfter.size()))

    def noise_image(self):
        vals = self.window_noise.values()
        noise_typ = vals[0]
        image = cv2.imread(self.ImageProcessingPath)
        head_tail = os.path.split(self.ImageProcessingPath)
        filename, file_extension = os.path.splitext(head_tail[1])
        self.dest_opencv = self.sciezka + r'\tmp\\' + filename + '_gauss' + file_extension

        if noise_typ == "Gauss":
            print('jestesmy w gaussie')
            row, col, ch = image.shape
            mean = vals[1]
            var = vals[2]
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss

            self.activeSaving = True
            self.imageType = '_' + str(noise_typ)
            self.imageToSave = noisy
            cv2.imwrite(self.dest_opencv, noisy)
            self.imageToSave = noisy
            self.imageType = '_gauss'
            self.set_image_after('CV2', noisy)

        elif noise_typ == "Poisson":
            p_vals = len(np.unique(image))
            p_vals = 2 ** np.ceil(np.log2(p_vals))
            noisy = np.random.poisson(image * p_vals) / float(p_vals)
            cv2.imwrite(self.dest_opencv, noisy)
            self.imageToSave = noisy
            self.imageType = '_poisson'
            self.set_image_after('CV2', noisy)

        elif noise_typ == "Speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            cv2.imwrite(self.dest_opencv, noisy)
            self.imageToSave = noisy
            self.imageType = '_speckle'
            self.set_image_after('CV2', noisy)

        elif noise_typ == "Salt&Pepper":
            print('szumimyt sp')
            row, col, ch = image.shape
            s_vs_p = vals[1]
            amount = vals[2]
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0

            cv2.imwrite(self.dest_opencv, out)
            self.imageType = '_s&p'
            self.imageToSave = out
            self.set_image_after('CV2', out)

        self.activeSaving = True

    def rotation_image(self):
        angle = self.window_rotation.values()
        im = Image.open(self.ImageProcessingPath)
        im = im.rotate(int(angle))
        self.activeSaving = True
        self.imageType = '_rotation_' + str(angle)
        self.imageToSave = im
        self.set_image_after('PIL', im)

    def mirror_image(self):
        og_image = Image.open(self.ImageProcessingPath)
        im = ImageOps.mirror(og_image)
        self.activeSaving = True
        self.imageType = '_MIRROR'
        self.imageToSave = im
        self.set_image_after('PIL', im)

    def grayscale_image(self):
        og_image = Image.open(self.ImageProcessingPath)
        im = ImageOps.grayscale(og_image)
        self.activeSaving = True
        self.imageType = '_BW'
        self.imageToSave = im
        self.set_image_after('PIL', im)

    def resize_image(self):
        width, height = self.window_resize.values()
        if width and height:
            dims = (width, height)
            im = Image.open(self.ImageProcessingPath)
            im = im.resize((int(width), int(height)))
            self.activeSaving = True
            self.imageType = '_resized_' + str(width) + 'x' + str(height)
            self.imageToSave = im
            self.set_image_after('PIL', im)

        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Nie możesz wpisać zerowych wartości!')
            error_dialog.exec_()

    def save_image(self):

        temp = os.path.splitext(self.ImageProcessingPath)
        # dest = temp[0] + self.imageType + temp[1]

        if self.activeSaving:
            if self.libType == 'PIL':
                destination = self.save_file(temp[0]  + self.imageType, temp[1])
                self.imageToSave.save(destination)
            elif self.libType == 'CV2':
                print(self.imageToSave)
                destination = self.save_file(temp[0]  + self.imageType, temp[1])
                cv2.imwrite(destination, self.imageToSave)
        else:
            print('Brak zdjęcia')

    def setValue(self, n):

        path = self.sciezka + r"\images\objects\\"
        klasy_zdjecia = ['spam']
        klasy = ['spam']
        for filename in os.listdir(path):
            klasy_zdjecia.append(path + filename)

        for i in range(1, 111):
            nazwa = "M" + str(i)
            klasy.append(nazwa)
            self.currentObject = n

        self.label_MessierNumber.setText(self.numer_messiera[self.currentObject - 1])
        self.label_NGC.setText(self.numer_NGC[self.currentObject - 1])
        self.label_NazwaZwyczajowa.setText(self.nazwa_zwyczajowa[self.currentObject - 1])

        self.label_TypObiektu.setText(self.typ_obiektu[self.currentObject - 1])
        self.label_Odleglosc.setText(str(self.odlegosc[self.currentObject - 1]))
        self.label_Gwiazdozbior.setText(self.gwiazdozbior[self.currentObject - 1])
        self.label_Jasnosc.setText(str(self.jasnosc[self.currentObject - 1]))

        pixmap = QtGui.QPixmap(path + "M" + str(self.currentObject) + ".png")
        self.label_21.setPixmap(pixmap.scaled(self.label.size()))

        # self.numer_messiera = df['Numer Messiera']
        # self.numer_NGC = df['Numer NGC']
        # self.nazwa_zwyczajowa = df['Nazwa zwyczajowa']
        # self.typ_obiektu = df['Typ obiektu']
        # self.odlegosc = df['Odległość od Ziemi (w tys. lat świetlnych)']
        # self.gwiazdozbior = df['Gwiazdozbiór']
        # self.jasnosc = df['Jasność widoma (m)']

    def open_folder(self, mode):
        # 1 - uczenie, 2 - generacja folderów 3 - split danych

        operationType = mode
        file = str(QFileDialog.getExistingDirectory(self, "Wybierz ścieżkę"))

        if operationType == 1:
            self.label_pathName.setText(file)
            self.trainPath = file + '/train'
            self.testPath = file + '/test'
            self.valPath = file + '/val'


        elif operationType == 2:
            self.paths = file

    def generate_folders(self):
        mainFolders = ['/train', '/test', '/val']

        for folder in mainFolders:
            dest = self.paths + folder
            print(dest)
            if not os.path.exists(dest):
                os.mkdir(dest)
            else:
                print('Folder istnieje ' + dest)

        for dir in mainFolders:
            for i in range(1, 111):
                dest_classes = self.paths + dir + '/M' + str(i)
                print(dest_classes)
                if not os.path.exists(dest_classes):
                    os.mkdir(dest_classes)
                else:
                    print('Folder istnieje ' + dest_classes)

    def check_data(self):
        print(self.trainPath)
        if self.trainPath:
            BS = int(self.BS)  # batch size
            shape = 294
            rodzaje_kolorow = ['grayscale', 'rgb']
            color = rodzaje_kolorow[1]

            training_generator = ImageDataGenerator(rescale=1 / 255, rotation_range=20, zoom_range=0.05,
                                                    width_shift_range=0.1, height_shift_range=0.1, shear_range=0.05)

            self.training_flow = training_generator.flow_from_directory(
                directory=self.trainPath,
                class_mode="categorical",
                target_size=(shape, shape),
                color_mode=color,
                shuffle=True,
                batch_size=BS)

            validation_generator = ImageDataGenerator(rescale=1 / 255)

            self.validation_flow = validation_generator.flow_from_directory(
                directory=self.valPath,
                class_mode="categorical",
                target_size=(shape, shape),
                color_mode=color,
                shuffle=True,
                batch_size=BS)

            test_generator = ImageDataGenerator(rescale=1 / 255)

            self.test_flow = test_generator.flow_from_directory(
                directory=self.testPath,
                class_mode="categorical",
                target_size=(shape, shape),
                color_mode=color,
                shuffle=False,
                batch_size=BS)

            if self.training_flow.samples != 0:
                self.dataOK = True
            else:
                self.dataOK = False

        else:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Brak wybranego datasetu!')
            error_dialog.exec_()

    def open_image(self):
        filename = QFileDialog.getOpenFileName(self, "Select File", "", "*.png *.jpg")
        path = filename[0]
        head, tail = os.path.split(path)
        self.image = path

        if not filename[0]:
            print('Zły plik')
        else:
            image = PIL.Image.open(path)
            width, height = image.size

            place = 0
            self.ImageClassifierPath = path
            self.updatePicture(path, tail, place, width, height)

    def open_image_processing(self):
        filename = QFileDialog.getOpenFileName(self, "Select File", "", "*.png *.jpg")
        path = filename[0]
        head, tail = os.path.split(path)
        print(filename)
        if not filename[0]:
            self.activeProcessing = False
        else:
            image = Image.open(path)
            self.activeProcessing = True
            self.widthProcessing, self.heightProcessing = image.size
            place = 1
            self.ImageProcessingPath = path
            self.updatePicture(path, tail, place, self.widthProcessing, self.heightProcessing)

    def updatePicture(self, path, tail, place, width, height):
        imageDimensions = (width, height)
        if place == 0:
            pixmap = QtGui.QPixmap(path)
            self.label.setPixmap(pixmap.scaled(self.label.size()))
            self.label_ImageName.setText(tail)
            self.label_ImageDimClassify.setText(str(imageDimensions))
            self.imageOpened = True

        elif place == 1:
            pixmap = QtGui.QPixmap(path)
            self.label_ImageBefore.setPixmap(pixmap.scaled(self.label_ImageBefore.size()))
            self.label_ImageProcessingName.setText(tail)
            self.label_ImageDimProcessing.setText(str(imageDimensions))

    def open_model(self):
        filename = QFileDialog.getOpenFileName(self, "Select File", "", "*.h5")
        self.path = filename[0]
        print(self.path)
        if not filename[0]:
            print('Brak modelu')
        else:
            worker = Worker(self.opening_model)
            self.threadpool.start(worker)

    def opening_model(self):
        print('Otwieranie modelu')
        self.model = load_model(self.path, compile=True)
        head, tail = os.path.split(self.path)
        self.label_ModelName.setText(tail)
        self.label_ModelName_2.setText(tail)
        self.modelOpened = True
        print('Model załadowany')

    def classify(self):
        # print('test')
        if self.imageOpened == True and self.modelOpened == True:
            img = image.load_img(self.image, target_size=(294, 294))
            img_tensor = image.img_to_array(img)  # (height, width, channels)
            img_tensor = np.expand_dims(img_tensor,
                                        axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
            img_tensor /= 255.  # imshow expects values in the range [0, 1]

            self.img_tens = img_tensor

            # print(self.img_tens)

            klasy = ['M1', 'M10', 'M100', 'M101', 'M102', 'M103', 'M104', 'M105', 'M106', 'M107', 'M108', 'M109', 'M11',
                     'M110',
                     'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M2', 'M20', 'M21', 'M22', 'M23', 'M24',
                     'M25', 'M26', 'M27', 'M28', 'M29', 'M3', 'M30', 'M31', 'M32', 'M33', 'M34', 'M35', 'M36', 'M37',
                     'M38',
                     'M39', 'M4', 'M40', 'M41', 'M42', 'M43', 'M44', 'M45', 'M46', 'M47', 'M48', 'M49', 'M5', 'M50',
                     'M51', 'M52', 'M53', 'M54', 'M55', 'M56', 'M57', 'M58', 'M59', 'M6', 'M60', 'M61', 'M62', 'M63',
                     'M64', 'M65', 'M66', 'M67', 'M68', 'M69', 'M7', 'M70', 'M71', 'M72', 'M73', 'M74', 'M75', 'M76',
                     'M77', 'M78', 'M79', 'M8', 'M80', 'M81', 'M82', 'M83', 'M84', 'M85', 'M86', 'M87', 'M88', 'M89',
                     'M9', 'M91', 'M92', 'M93', 'M94', 'M95', 'M96', 'M97', 'M98', 'M99']

            y_prob = self.model.predict(self.img_tens)

            lol = y_prob[0]

            dictionary = dict(zip(klasy, lol))

            a = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)

            self.classes = []
            self.values = []

            for i in a[:5]:
                self.classes.append(i[0])
                self.values.append(i[1])

            self.label_5.setText(self.classes[0])
            self.label_6.setText(self.classes[1])
            self.label_7.setText(self.classes[2])
            self.label_8.setText(self.classes[3])
            self.label_9.setText(self.classes[4])

            self.label_10.setText(str(round(100 * self.values[0], 2)))
            self.label_11.setText(str(round(100 * self.values[1], 2)))
            self.label_12.setText(str(round(100 * self.values[2], 2)))
            self.label_13.setText(str(round(100 * self.values[3], 2)))
            self.label_14.setText(str(round(100 * self.values[4], 2)))

        elif not self.modelOpened:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Brak modelu!')
            error_dialog.exec_()
        elif not self.imageOpened:
            error_dialog = QtWidgets.QErrorMessage()
            error_dialog.showMessage('Brak zdjęcia!')
            error_dialog.exec_()

    def read_excel(self):
        df = pd.read_excel(self.sciezka + "\wikipedia\wikipedia_messier.xlsx")
        self.numer_messiera = df['Numer Messiera']
        self.numer_NGC = df['Numer NGC']
        self.nazwa_zwyczajowa = df['Nazwa zwyczajowa']
        self.typ_obiektu = df['Typ obiektu']
        self.odlegosc = df['Odległość od Ziemi (w tys. lat świetlnych)']
        self.gwiazdozbior = df['Gwiazdozbiór']
        self.jasnosc = df['Jasność widoma (m)']

    def katalog_obiektow(self):

        self.pushM_1 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M1.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M1_on.png"))
        self.pushM_2 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M2.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M2_on.png"))
        self.pushM_3 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M3.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M3_on.png"))
        self.pushM_4 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M4.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M4_on.png"))
        self.pushM_5 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M5.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M5_on.png"))
        self.pushM_6 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M6.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M6_on.png"))
        self.pushM_7 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M7.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M7_on.png"))
        self.pushM_8 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M8.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M8_on.png"))
        self.pushM_9 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M9.png"),
                                 QtGui.QPixmap(self.sciezka + r"\images\icons\M9_on.png"))
        self.pushM_10 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M10.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M10_on.png"))
        self.pushM_11 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M11.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M11_on.png"))
        self.pushM_12 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M12.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M12_on.png"))
        self.pushM_13 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M13.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M13_on.png"))
        self.pushM_14 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M14.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M14_on.png"))
        self.pushM_15 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M15.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M15_on.png"))
        self.pushM_16 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M16.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M16_on.png"))
        self.pushM_17 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M17.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M17_on.png"))
        self.pushM_18 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M18.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M18_on.png"))
        self.pushM_19 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M19.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M19_on.png"))
        self.pushM_20 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M20.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M20_on.png"))
        self.pushM_21 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M21.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M21_on.png"))
        self.pushM_22 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M22.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M22_on.png"))
        self.pushM_23 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M23.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M23_on.png"))
        self.pushM_24 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M24.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M24_on.png"))
        self.pushM_25 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M25.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M25_on.png"))
        self.pushM_26 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M26.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M26_on.png"))
        self.pushM_27 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M27.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M27_on.png"))
        self.pushM_28 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M28.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M28_on.png"))
        self.pushM_29 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M29.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M29_on.png"))
        self.pushM_30 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M30.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M30_on.png"))
        self.pushM_31 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M31.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M31_on.png"))
        self.pushM_32 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M32.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M32_on.png"))
        self.pushM_33 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M33.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M33_on.png"))
        self.pushM_34 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M34.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M34_on.png"))
        self.pushM_35 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M35.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M35_on.png"))
        self.pushM_36 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M36.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M36_on.png"))
        self.pushM_37 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M37.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M37_on.png"))
        self.pushM_38 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M38.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M38_on.png"))
        self.pushM_39 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M39.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M39_on.png"))
        self.pushM_40 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M40.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M40_on.png"))
        self.pushM_41 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M41.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M41_on.png"))
        self.pushM_42 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M42.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M42_on.png"))
        self.pushM_43 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M43.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M43_on.png"))
        self.pushM_44 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M44.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M44_on.png"))
        self.pushM_45 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M45.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M45_on.png"))
        self.pushM_46 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M46.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M46_on.png"))
        self.pushM_47 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M47.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M47_on.png"))
        self.pushM_48 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M48.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M48_on.png"))
        self.pushM_49 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M49.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M49_on.png"))
        self.pushM_50 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M50.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M50_on.png"))
        self.pushM_51 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M51.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M51_on.png"))
        self.pushM_52 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M52.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M52_on.png"))
        self.pushM_53 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M53.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M53_on.png"))
        self.pushM_54 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M54.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M54_on.png"))
        self.pushM_55 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M55.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M55_on.png"))
        self.pushM_56 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M56.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M56_on.png"))
        self.pushM_57 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M57.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M57_on.png"))
        self.pushM_58 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M58.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M58_on.png"))
        self.pushM_59 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M59.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M59_on.png"))
        self.pushM_60 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M60.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M60_on.png"))
        self.pushM_61 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M61.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M61_on.png"))
        self.pushM_62 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M62.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M62_on.png"))
        self.pushM_63 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M63.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M63_on.png"))
        self.pushM_64 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M64.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M64_on.png"))
        self.pushM_65 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M65.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M65_on.png"))
        self.pushM_66 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M66.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M66_on.png"))
        self.pushM_67 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M67.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M67_on.png"))
        self.pushM_68 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M68.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M68_on.png"))
        self.pushM_69 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M69.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M69_on.png"))
        self.pushM_70 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M70.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M70_on.png"))
        self.pushM_71 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M71.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M71_on.png"))
        self.pushM_72 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M72.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M72_on.png"))
        self.pushM_73 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M73.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M73_on.png"))
        self.pushM_74 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M74.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M74_on.png"))
        self.pushM_75 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M75.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M75_on.png"))
        self.pushM_76 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M76.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M76_on.png"))
        self.pushM_77 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M77.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M77_on.png"))
        self.pushM_78 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M78.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M78_on.png"))
        self.pushM_79 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M79.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M79_on.png"))
        self.pushM_80 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M80.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M80_on.png"))
        self.pushM_81 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M81.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M81_on.png"))
        self.pushM_82 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M82.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M82_on.png"))
        self.pushM_83 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M83.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M83_on.png"))
        self.pushM_84 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M84.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M84_on.png"))
        self.pushM_85 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M85.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M85_on.png"))
        self.pushM_86 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M86.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M86_on.png"))
        self.pushM_87 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M87.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M87_on.png"))
        self.pushM_88 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M88.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M88_on.png"))
        self.pushM_89 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M89.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M89_on.png"))
        self.pushM_90 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M90.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M90_on.png"))
        self.pushM_91 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M91.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M91_on.png"))
        self.pushM_92 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M92.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M92_on.png"))
        self.pushM_93 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M93.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M93_on.png"))
        self.pushM_94 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M94.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M94_on.png"))
        self.pushM_95 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M95.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M95_on.png"))
        self.pushM_96 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M96.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M96_on.png"))
        self.pushM_97 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M97.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M97_on.png"))
        self.pushM_98 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M98.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M98_on.png"))
        self.pushM_99 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M99.png"),
                                  QtGui.QPixmap(self.sciezka + r"\images\icons\M99_on.png"))
        self.pushM_100 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M100.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M100_on.png"))
        self.pushM_101 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M101.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M101_on.png"))
        self.pushM_102 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M102.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M102_on.png"))
        self.pushM_103 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M103.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M103_on.png"))
        self.pushM_104 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M104.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M104_on.png"))
        self.pushM_105 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M105.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M105_on.png"))
        self.pushM_106 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M106.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M106_on.png"))
        self.pushM_107 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M107.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M107_on.png"))
        self.pushM_108 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M108.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M108_on.png"))
        self.pushM_109 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M109.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M109_on.png"))
        self.pushM_110 = PicButton(QtGui.QPixmap(self.sciezka + r"\images\icons\M110.png"),
                                   QtGui.QPixmap(self.sciezka + r"\images\icons\M110_on.png"))

        self.gridLayout.addWidget(self.pushM_1, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_2, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_3, 0, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_4, 0, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_5, 0, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_6, 0, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_7, 0, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_8, 0, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_9, 0, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_10, 0, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_11, 0, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_12, 1, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_13, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_14, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_15, 1, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_16, 1, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_17, 1, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_18, 1, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_19, 1, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_20, 1, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_21, 1, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_22, 1, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_23, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_24, 2, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_25, 2, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_26, 2, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_27, 2, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_28, 2, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_29, 2, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_30, 2, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_31, 2, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_32, 2, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_33, 2, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_34, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_35, 3, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_36, 3, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_37, 3, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_38, 3, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_39, 3, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_40, 3, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_41, 3, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_42, 3, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_43, 3, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_44, 3, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_45, 4, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_46, 4, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_47, 4, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_48, 4, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_49, 4, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_50, 4, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_51, 4, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_52, 4, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_53, 4, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_54, 4, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_55, 4, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_56, 5, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_57, 5, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_58, 5, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_59, 5, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_60, 5, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_61, 5, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_62, 5, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_63, 5, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_64, 5, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_65, 5, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_66, 5, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_67, 6, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_68, 6, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_69, 6, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_70, 6, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_71, 6, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_72, 6, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_73, 6, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_74, 6, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_75, 6, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_76, 6, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_77, 6, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_78, 7, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_79, 7, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_80, 7, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_81, 7, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_82, 7, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_83, 7, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_84, 7, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_85, 7, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_86, 7, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_87, 7, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_88, 7, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_89, 8, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_90, 8, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_91, 8, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_92, 8, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_93, 8, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_94, 8, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_95, 8, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_96, 8, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_97, 8, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_98, 8, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_99, 8, 10, 1, 1)

        self.gridLayout.addWidget(self.pushM_100, 9, 0, 1, 1)
        self.gridLayout.addWidget(self.pushM_101, 9, 1, 1, 1)
        self.gridLayout.addWidget(self.pushM_102, 9, 2, 1, 1)
        self.gridLayout.addWidget(self.pushM_103, 9, 3, 1, 1)
        self.gridLayout.addWidget(self.pushM_104, 9, 4, 1, 1)
        self.gridLayout.addWidget(self.pushM_105, 9, 5, 1, 1)
        self.gridLayout.addWidget(self.pushM_106, 9, 6, 1, 1)
        self.gridLayout.addWidget(self.pushM_107, 9, 7, 1, 1)
        self.gridLayout.addWidget(self.pushM_108, 9, 8, 1, 1)
        self.gridLayout.addWidget(self.pushM_109, 9, 9, 1, 1)
        self.gridLayout.addWidget(self.pushM_110, 9, 10, 1, 1)


app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
window.show()
window.setFixedSize(960, 520)
myStream = MyStream()
myStream.message.connect(window.on_myStream_message)
sys.stdout = myStream

app.exec()
