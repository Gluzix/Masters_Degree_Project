from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex, QMutexLocker
from PyQt6.QtGui import QPixmap, QImage
import cv2
import dlib
import numpy as np


class OpenCvWorker(QObject):
    pixmap_ready = pyqtSignal(QPixmap)
    black_pixmap_ready = pyqtSignal(QPixmap)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'E:/Projekt Magisterski/pre_trained_data/shape_predictor_68_face_landmarks.dat')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.is_running = False
        self.mutex = QMutex()

    def __del__(self):
        self.stop()

    def stop(self):
        QMutexLocker(self.mutex)
        if self.is_running:
            self.is_running = False
            self.cap.release()

    @pyqtSlot()
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            self.run()

    def run(self):
        image_count = 0
        while self.is_running:
            QMutexLocker(self.mutex)
            ret, image = self.cap.read()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray_image, 1)
            height, width, channel = image.shape

            blank_image = np.zeros((height, width, 3), np.uint8)

            for rect in rects:
                shape = self.predictor(gray_image, rect)
                shape_numpy_arr = np.zeros((68, 2), dtype='int')
                for i in range(0, 68):
                    shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

                for i, (x, y) in enumerate(shape_numpy_arr):
                    cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
                    cv2.circle(blank_image, (x, y), 1, (255, 255, 255), -1)

            # cv2.imwrite(f"normal_{image_count}.png", image)
            # cv2.imwrite(f"black_{image_count}.png", blank_image)

            image_count += 1

            bytesPerLine = 3 * width
            qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
            qImage_black = QImage(blank_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

            qPixmap = QPixmap.fromImage(qImage)
            qPixmap_black = QPixmap.fromImage(qImage_black)

            self.pixmap_ready.emit(qPixmap)
            self.black_pixmap_ready.emit(qPixmap_black)
