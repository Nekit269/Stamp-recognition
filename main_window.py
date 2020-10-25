import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage, qRgb
from PyQt5.QtCore import Qt

from imageio import imread
import cv2
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.preprocessing.image import *
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanva
# from matplotlib.figure import SubplotParams

from designs import Ui_MainWindow

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    IMG_SIZE=(2302,1632, 3)
    gray_color_table = [qRgb(i, i, i) for i in range(256)]

    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)

        self._img = None

        self._model = None
        self.load_custom_model()

        self.pushButton.clicked.connect(self.load_image)
        self.pushButton_2.clicked.connect(self.analyze)

    def toQImage(self, im, copy=False):
        print(im.dtype)
        if im is None:
            return QImage()

        # if im.dtype == np.uint8:
        if len(im.shape) == 2:
            print(np.max(im))
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(self.gray_color_table)
            return qim.copy() if copy else qim

        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim

        raise NotImplementedException

    def set_image(self, ar, label):
        image = self.toQImage(ar)
        pixmap = QPixmap(image)
        pixmap = pixmap.scaled(label.width(),label.height(), Qt.KeepAspectRatio)  
        label.setPixmap(pixmap)

    def make_model(self):
        inp = Input(self.IMG_SIZE)
        x = ZeroPadding2D(((1,1),(16,16)))(inp)
        skips=[]
        for n in [9,12,12]:
            skips.append(x)
            x = Conv2D(n, kernel_size=3,strides=2,activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
        for n in [9,12]:
            x = UpSampling2D(size=2)(x)
            x = concatenate([x, skips.pop()])
            x = Conv2DTranspose(n, kernel_size=3,strides=1,activation='relu',padding='same')(x)
            x = BatchNormalization()(x)
        x = UpSampling2D(size=2)(x)
        x = concatenate([x, skips.pop()])
        x = Conv2DTranspose(1, kernel_size=3,strides=1,activation='sigmoid',padding='same')(x)
        x = Cropping2D(((1,1),(16,16)))(x)
        x = Reshape((self.IMG_SIZE[0]*self.IMG_SIZE[1],1,))(x)
        return Model(inp, x)
    
    def load_custom_model(self):
        self._model = self.make_model()
        self._model.load_weights("./model/model003.hdf5")

    def load_image(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg, *.png)")
        
        if fname[0] == '':
            return
        source_img = imread(fname[0])
        img = source_img / 255
        img_shape = img.shape
        print(img_shape)
        if (img_shape[0] > self.IMG_SIZE[0] or img_shape[1] > self.IMG_SIZE[1]):
            self._img = cv2.resize(img, (self.IMG_SIZE[1], self.IMG_SIZE[0]))
        else:
            self._img = cv2.copyMakeBorder(img,
                                          (self.IMG_SIZE[0]-img.shape[0])//2,
                                          (self.IMG_SIZE[0]-img.shape[0])-(self.IMG_SIZE[0]-img.shape[0])//2,
                                          (self.IMG_SIZE[1]-img.shape[1])//2,
                                          (self.IMG_SIZE[1]-img.shape[1])-(self.IMG_SIZE[1]-img.shape[1])//2,
                                          cv2.BORDER_REFLECT)

        self.set_image(source_img, self.label)

    def analyze(self):
        if self._img is None:
            return

        predicted = self._model.predict(np.array([self._img]))
        img_pred = np.round(predicted).reshape((1,self.IMG_SIZE[0],self.IMG_SIZE[1]))
        img_scaled = (img_pred * 255.0).astype(np.uint8)
        self.set_image(img_scaled[0], self.label_2)
