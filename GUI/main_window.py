from PyQt6.QtWidgets import QMainWindow
from resources.main_window_ui import Ui_MainWindow
from GUI.live_data_widget import LiveDataWidget


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Projekt Magisterski")
        self.live_data_widget = LiveDataWidget(self)
        self.main_layout.addWidget(self.live_data_widget)
