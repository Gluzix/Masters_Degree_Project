from PyQt6.QtWidgets import QWidget, QGraphicsPixmapItem, QGraphicsScene
from PyQt6.QtGui import QPixmap
from resources.qt_resources.image_widget_ui import Ui_Form
from PyQt6.QtCore import pyqtSlot


class ImageWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.graphics_scene = QGraphicsScene(self)
        self.graphics_pixmap_item = QGraphicsPixmapItem()
        self.graphics_scene.addItem(self.graphics_pixmap_item)
        self.imageGraphicsView.setScene(self.graphics_scene)

    @pyqtSlot(QPixmap)
    def set_pixmap(self, pixmap):
        self.graphics_pixmap_item.setPixmap(pixmap)
        self.imageGraphicsView.update()
