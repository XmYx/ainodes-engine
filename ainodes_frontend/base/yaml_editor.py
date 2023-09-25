import functools

import yaml
from PyQt6.QtWidgets import QTabWidget, QKeySequenceEdit
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog, QComboBox, \
    QCheckBox, QHBoxLayout, QLayout


folder_options = [
    "checkpoints",
    "controlnet",
    "embeddings",
    "hypernetworks",
    "loras",
    "output",
    "t2i_adapter",
    "upscalers",
    "vae",
]

DEFAULT_KEYBINDINGS = {
    "save": "Ctrl+S",
    "open": "Ctrl+O",
    # ... add other defaults as needed
}

class YamlEditorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = {}
        self.default_options = self.load_default_options('config/default_options.yaml')

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.set_dark_theme()

        self.tab_widget = QTabWidget()
        self.settings_widget = QWidget()
        self.keybindings_widget = QWidget()

        self.tab_widget.addTab(self.settings_widget, "Settings")
        self.tab_widget.addTab(self.keybindings_widget, "Keybindings")

        self.layout.addWidget(self.tab_widget)

        self.settings_layout = QVBoxLayout(self.settings_widget)
        self.keybindings_layout = QVBoxLayout(self.keybindings_widget)

    def set_dark_theme(self):
        style = """
        QWidget {
            background-color: #2b2b2b;
            color: #b2b2b2;
        }
        QLineEdit, QComboBox, QPushButton, QLabel {
            background-color: #353535;
            color: #b2b2b2;
        }
        """
        self.setStyleSheet(style)
    def load_default_options(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            self.values = yaml.safe_load(file)

        if 'keybindings' not in self.values:
            self.values['keybindings'] = DEFAULT_KEYBINDINGS

        for key, value in self.values.items():
            if key == 'keybindings':
                continue
            label = QLabel(key)
            self.layout.addWidget(label)

            if key in self.default_options:  # Add QComboBox for keys found in default_options
                combo_box = QComboBox()
                combo_box.setObjectName(key)
                combo_box.addItems(self.default_options[key])
                combo_box.setCurrentText(str(value))
                self.layout.addWidget(combo_box)

            elif isinstance(value, bool):  # Add QCheckBox for boolean values
                checkbox = QCheckBox()
                checkbox.setObjectName(key)
                checkbox.setChecked(value)
                self.layout.addWidget(checkbox)

            else:  # Add QLineEdit and QPushButton for string values
                layout = QHBoxLayout()

                line_edit = QLineEdit(str(value))
                line_edit.setObjectName(key)
                layout.addWidget(line_edit)
                if key in folder_options:
                    browse_button = QPushButton('Browse...')
                    browse_button.clicked.connect(functools.partial(self.browse, line_edit))
                    layout.addWidget(browse_button)
                self.layout.addLayout(layout)

        # Load keybindings
        keybindings_label = QLabel("Keybindings")
        self.keybindings_layout.addWidget(keybindings_label)

        for key, value in self.values['keybindings'].items():
            layout = QHBoxLayout()

            label = QLabel(key)
            layout.addWidget(label)

            key_sequence_edit = QKeySequenceEdit(Qt.QKeySequence(value))
            key_sequence_edit.setObjectName(f"keybinding_{key}")
            layout.addWidget(key_sequence_edit)

            self.keybindings_layout.addLayout(layout)

        # Add save and cancel buttons to the main layout (not inside tabs)
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_yaml)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.cancel)

        self.layout.addWidget(save_button)
        self.layout.addWidget(cancel_button)
    def browse(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        if folder_path:
            line_edit.setText(folder_path)

    def save_yaml(self):
        self.save_widgets_from_layout(self.layout)

        # Update keybindings from the editor widgets
        for key in self.values['keybindings']:
            key_sequence_edit = self.findChild(QKeySequenceEdit, f"keybinding_{key}")
            if key_sequence_edit:
                self.values['keybindings'][key] = key_sequence_edit.keySequence().toString()

        file_path = "config/settings.yaml"

        with open(file_path, 'w') as file:
            yaml.safe_dump(self.values, file)
        QMessageBox.information(self, 'Success', 'YAML file saved successfully.')
        from ainodes_frontend.base import load_settings

        load_settings()
        self.close()

    def save_widgets_from_layout(self, layout):
        for i in range(layout.count()):
            item = layout.itemAt(i)
            widget = item.widget()

            if widget is not None:
                if isinstance(widget, QLineEdit):
                    key = widget.objectName()
                    value = widget.text()
                    self.values[key] = value

                elif isinstance(widget, QCheckBox):
                    key = widget.objectName()
                    value = widget.isChecked()
                    self.values[key] = value

                elif isinstance(widget, QComboBox):
                    key = widget.objectName()
                    value = widget.currentText()
                    self.values[key] = value

            elif isinstance(item, QLayout):
                self.save_widgets_from_layout(item.layout())
    def cancel(self):
        self.close()
