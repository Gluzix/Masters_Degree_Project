from PyQt6.QtWidgets import QWidget
from resources.live_data_widget_ui import Ui_Form
from GUI.image_widget import ImageWidget


class LiveDataWidget(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.image_widget = ImageWidget(self)
        self.mainLayout.addWidget(self.image_widget)

