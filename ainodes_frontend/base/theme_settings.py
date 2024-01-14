import os
from PyQt6 import QtWidgets, QtCore
from ainodes_frontend import singleton as gs
class ThemePreferencesDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Theme Preferences")
        self.setLayout(QtWidgets.QVBoxLayout())

        # Dropdown for themes
        self.themeComboBox = QtWidgets.QComboBox()
        self.layout().addWidget(self.themeComboBox)

        # Populate the dropdown with themes from the folder
        self.themeFolder = "ainodes_frontend/qss"
        self.populateThemes()

        # OK and Cancel buttons
        buttons = QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        self.buttonBox = QtWidgets.QDialogButtonBox(buttons)
        self.layout().addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.applyTheme)
        self.buttonBox.rejected.connect(self.reject)

    def populateThemes(self):
        for file in os.listdir(self.themeFolder):
            if file.endswith(".qss"):
                self.themeComboBox.addItem(file)

    def applyTheme(self):
        selectedTheme = self.themeComboBox.currentText()
        themePath = os.path.join(self.themeFolder, selectedTheme)

        gs.qss = f"qss/{selectedTheme}"

        with open(themePath, "r") as file:
            stylesheet = file.read()
            QtWidgets.QApplication.instance().setStyleSheet(stylesheet)
        self.accept()