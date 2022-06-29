from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex, QMutexLocker, QDir
from PyQt6.QtGui import QPixmap, QImage
import cv2
import dlib
import numpy as np


class OpenCvWorker(QObject):
    pixmap_ready = pyqtSignal(QPixmap)
    black_pixmap_ready = pyqtSignal(QPixmap)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat')

    def __init__(self, record=False, parent=None):
        super().__init__(parent)
        self.cap = None
        self.is_running = False
        self.mutex = QMutex()
        self.record = record
        self.fps_to_record = 15

    def __del__(self):
        self.stop()

    def stop(self):
        QMutexLocker(self.mutex)
        if self.is_running:
            self.is_running = False
            self.cap.release()

    def set_fps_to_record(self, fps_to_record):
        self.fps_to_record = fps_to_record

    @pyqtSlot()
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.cap = cv2.VideoCapture(0)
            self.run()

    def run(self):
        image_count = 0
        dir = QDir()
        path_images = dir.currentPath() + "/images/"
        while self.is_running:
            QMutexLocker(self.mutex)
            ret, image = self.cap.read()

            height, width, channel = image.shape

            if self.record:
                cv2.imwrite(f"{path_images}image_{image_count}.png", image)
                if image_count == self.fps_to_record:
                    self.stop()
                    break

            image_count += 1

            bytesPerLine = 3 * width
            qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

            qPixmap = QPixmap.fromImage(qImage)

            self.pixmap_ready.emit(qPixmap)
