from PyQt6.QtWidgets import QApplication
import sys
from GUI.main_window import MainWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec()
