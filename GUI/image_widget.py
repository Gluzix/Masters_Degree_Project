from PyQt6.QtWidgets import QWidget, QGraphicsPixmapItem, QGraphicsScene
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSlot, QThread, QObject, PYQT_SIGNAL
from resources.image_widget_ui import Ui_Form
from GUI.opencv_worker import OpenCvWorker


class ImageWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_pixmap_item = QGraphicsPixmapItem(self)
        self.graphics_scene.addItem(self.graphics_pixmap_item)
        self.imageGraphicsView.setScene(self.graphics_scene)

        self.thread = QThread()
        self.open_cv_worker = OpenCvWorker()
        QObject.connect(self.open_cv_worker, PYQT_SIGNAL("pixmap_ready(QPixmap)"), self.set_pixmap)
        self.open_cv_worker.moveToThread(self.thread)

    @pyqtSlot(QPixmap)
    def set_pixmap(self, pixmap):
        self.graphics_pixmap_item.setPixmap(pixmap)
        self.imageGraphicsView.update()
