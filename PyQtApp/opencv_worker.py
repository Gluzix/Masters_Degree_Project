from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMutex, QMutexLocker, QDir
from PyQt6.QtGui import QPixmap, QImage
import cv2
import dlib
from keras.models import load_model
from Pix2Pix.pix_2_pix_trainer import Pix2PixTrainer
import numpy as np


class OpenCvWorker(QObject):
    pixmap_ready = pyqtSignal(QPixmap)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        'E:/Projekt Magisterski/resources/pre_trained_data/shape_predictor_68_face_landmarks.dat')
    model = load_model("E:/Projekt Magisterski/resources/datasets_and_ready_models/duda_big_dataset_batch_10_around_4000_samples/model_192000.h5")
    face_cascade = cv2.CascadeClassifier("E:/Projekt Magisterski/resources/pre_trained_data/haarcascade_frontalface_default.xml")

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
        while self.is_running:
            QMutexLocker(self.mutex)
            ret, image = self.cap.read()

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)

            if len(faces) == 0:
                continue

            (x, y, w, h) = faces[0]

            cropped_image_gray = gray_image[y - 30:y + h + 30, x - 30:x + w + 30]
            cropped_image_color = image[y - 30:y + h + 30, x - 30:x + w + 30]

            try:
                resized_cropped_image_gray = cv2.resize(cropped_image_gray, (256, 256), interpolation=cv2.INTER_NEAREST)
                resized_cropped_image_color = cv2.resize(cropped_image_color, (256, 256),
                                                         interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                print(str(e))
                continue

            rects = self.detector(resized_cropped_image_gray, 1)

            if not rects:
                continue

            height, width, channel = resized_cropped_image_color.shape
            blank_image = np.zeros((height, width, 3), np.uint8)

            for rect in rects:
                shape = self.predictor(cropped_image_gray, rect)
                shape_numpy_arr = np.zeros((68, 2), dtype='int')
                for i in range(0, 68):
                    shape_numpy_arr[i] = (shape.part(i).x, shape.part(i).y)

                for i, (x, y) in enumerate(shape_numpy_arr):
                    cv2.circle(blank_image, (x, y), 1, (255, 255, 255), -1)

            realA = np.expand_dims(blank_image, axis=0)
            realA = (realA - 127.5) / 127.5

            X_fakeB, _ = Pix2PixTrainer.generate_fake_samples(self.model, realA, 1)
            X_fakeB = (X_fakeB + 1) / 2.0
            fake_image = cv2.cvtColor(X_fakeB[0], cv2.COLOR_BGR2RGB)

            bytesPerLine = 3 * width

            qImage = QImage(fake_image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)

            qPixmap = QPixmap.fromImage(qImage)

            self.pixmap_ready.emit(qPixmap)
