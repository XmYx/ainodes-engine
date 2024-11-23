# settings_dialog.py
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import QSettings

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = QSettings("aiNodes", "Engine")  # Adjust these values

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Label and LineEdit for ComfyUI path
        self.comfyui_label = QLabel("ComfyUI Folder:")
        self.comfyui_line_edit = QLineEdit()
        self.comfyui_line_edit.setPlaceholderText("Select the ComfyUI folder")

        # Load existing path from settings
        comfyui_path = self.settings.value("comfyui_path", "")
        self.comfyui_line_edit.setText(comfyui_path)

        # Browse button
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_comfyui_folder)

        # Save button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)

        # Layout setup
        layout.addWidget(self.comfyui_label)
        layout.addWidget(self.comfyui_line_edit)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def browse_comfyui_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select ComfyUI Folder")
        if folder:
            self.comfyui_line_edit.setText(folder)

    def save_settings(self):
        import os
        import subprocess

        comfyui_path = self.comfyui_line_edit.text().strip()
        if not comfyui_path:
            QMessageBox.warning(self, "Warning", "Please select a valid ComfyUI folder.")
            return
        if not os.path.exists(comfyui_path):
            reply = QMessageBox.question(self, 'ComfyUI Not Found',
                                         "The specified ComfyUI folder does not exist. Do you want to clone ComfyUI from GitHub?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    subprocess.check_call(
                        ['git', 'clone', 'https://github.com/comfyanonymous/ComfyUI.git', comfyui_path])
                    QMessageBox.information(self, "ComfyUI Cloned", "ComfyUI has been cloned successfully.")
                except subprocess.CalledProcessError as e:
                    QMessageBox.critical(self, "Error", f"Failed to clone ComfyUI: {e}")
                    return
            else:
                return  # User chose not to clone; do not save settings
        self.settings.setValue("comfyui_path", comfyui_path)
        QMessageBox.information(self, "Settings Saved", "ComfyUI path saved successfully.")
        self.accept()
