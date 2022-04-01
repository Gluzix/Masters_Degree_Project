from PyQt6.QtWidgets import QWidget
from resources.live_data_widget_ui import Ui_Form
from GUI.image_widget import ImageWidget
from PyQt6.QtCore import pyqtSlot, QThread
from GUI.opencv_worker import OpenCvWorker


class LiveDataWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.image_widget = ImageWidget(self)
        self.black_image_widget = ImageWidget(self)
        self.mainLayout.addWidget(self.image_widget)
        self.mainLayout.addWidget(self.black_image_widget)

        self.thread = QThread()
        self.open_cv_worker = OpenCvWorker()
        self.open_cv_worker.moveToThread(self.thread)

        self.thread.started.connect(self.open_cv_worker.run)
        self.open_cv_worker.pixmap_ready.connect(self.image_widget.set_pixmap)
        self.open_cv_worker.black_pixmap_ready.connect(self.black_image_widget.set_pixmap)

        self.thread.start()
