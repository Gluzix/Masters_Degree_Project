from PyQt6.QtWidgets import QMainWindow, QWidget
from PyQt6.QtCore import pyqtSlot
from resources.main_window_ui import Ui_MainWindow
from live_data_widget import LiveDataWidget


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Projekt Magisterski")
        self.main_widget = QWidget()
        self.main_layout.addWidget(self.main_widget)
        self.live_button.clicked.connect(self.on_live_data_button_clicked)
        self.record_button.clicked.connect(self.on_record_data_button_clicked)

    @pyqtSlot()
    def on_live_data_button_clicked(self):
        live_data_widget = LiveDataWidget(self)
        self.main_layout.replaceWidget(self.main_widget, live_data_widget)
        self.main_widget = live_data_widget

    @pyqtSlot()
    def on_record_data_button_clicked(self):
        live_data_widget = LiveDataWidget(True, 500, self)
        self.main_layout.replaceWidget(self.main_widget, live_data_widget)
        self.main_widget = live_data_widget
