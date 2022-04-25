from PyQt6.QtWidgets import QWidget
from resources.live_data_widget_ui import Ui_Form
from image_widget import ImageWidget
from PyQt6.QtCore import pyqtSlot, QThread, pyqtSignal
from opencv_worker import OpenCvWorker


class LiveDataWidget(QWidget, Ui_Form):
    acquisition_start = pyqtSignal()
    acquisition_stop = pyqtSignal()

    def __init__(self, record=False, frames_to_record=15, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.image_widget = ImageWidget(self)
        self.black_image_widget = ImageWidget(self)
        self.mainLayout.addWidget(self.image_widget, 0, 0)
        self.mainLayout.addWidget(self.black_image_widget, 0, 1)

        self.thread = QThread()
        if record:
            self.open_cv_worker = OpenCvWorker(record)
            self.open_cv_worker.set_fps_to_record(frames_to_record)
        else:
            self.open_cv_worker = OpenCvWorker(record)
        self.open_cv_worker.moveToThread(self.thread)

        self.start_image_acquisition.clicked.connect(self.on_start_button_clicked)
        self.stop_image_acquisition.clicked.connect(self.on_stop_button_clicked)

        self.thread.started.connect(self.open_cv_worker.run)
        self.open_cv_worker.pixmap_ready.connect(self.image_widget.set_pixmap)
        self.open_cv_worker.black_pixmap_ready.connect(self.black_image_widget.set_pixmap)
        self.acquisition_start.connect(self.open_cv_worker.start)
        self.acquisition_stop.connect(self.open_cv_worker.stop)

        self.thread.start()

    @pyqtSlot()
    def on_start_button_clicked(self):
        self.acquisition_start.emit()

    @pyqtSlot()
    def on_stop_button_clicked(self):
        self.open_cv_worker.stop()
