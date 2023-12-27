import functools
import os

import yaml
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QTabWidget, QKeySequenceEdit
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog, QComboBox, \
    QCheckBox, QHBoxLayout


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
    "save": {"name":"Save",
             "shortcut":"Ctrl+S"},
    "open": {"name":"Open",
             "shortcut":"Ctrl+O"},
    "run": {"name":"Run Graph",
             "shortcut":"End"},
    "run_loop": {"name":"Run Loop",
             "shortcut":"Ctlr+R"},
    "stop_loop": {"name":"Stop Graph",
             "shortcut":"Ctrl+T"},
    "search": {"name":"Search Nodes",
             "shortcut":"Home"},
    "node_list": {"name":"Node List",
             "shortcut":"Esc"},
    "console": {"name":"Console",
             "shortcut":"`"},
    "minimap": {"name":"Minimap",
             "shortcut":"F2"},
    "help": {"name":"Help",
             "shortcut":"F1"},
    "fullscreen": {"name":"Fullscreen",
             "shortcut":"F11"}
}
def get_yaml_files_in_user_dir():
    directory = 'config/user'

    os.makedirs(directory, exist_ok=True)

    return [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.yaml')]

class YamlEditorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = {}
        #self.default_options = self.load_default_options()

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

        # Create a dropdown (QComboBox) for the YAML files
        self.user_profiles_dropdown = QComboBox()
        self.user_profiles_dropdown.addItems(get_yaml_files_in_user_dir())
        self.layout.addWidget(self.user_profiles_dropdown)

        # Create a QLineEdit for the user to input a new profile name
        self.new_profile_name_edit = QLineEdit()
        self.new_profile_name_edit.setPlaceholderText("Enter new profile name")
        self.layout.addWidget(self.new_profile_name_edit)

        # Create a save button to save current settings as a new profile
        self.save_profile_button = QPushButton('Save as New Profile')
        self.save_profile_button.clicked.connect(self.save_as_new_profile)
        self.layout.addWidget(self.save_profile_button)
        self.user_profiles_dropdown.currentIndexChanged.connect(self.load_selected_profile)
        # Add save and cancel buttons to the main layout (not inside tabs)
        save_button = QPushButton('Save')
        save_button.clicked.connect(self.save_yaml)
        cancel_button = QPushButton('Cancel')
        cancel_button.clicked.connect(self.cancel)

        self.layout.addWidget(save_button)
        self.layout.addWidget(cancel_button)

    def load_selected_profile(self, index):
        if index == -1:  # No item selected
            return

        # Get the selected profile's file name
        selected_profile_name = self.user_profiles_dropdown.currentText()
        file_path = os.path.join('config/user', selected_profile_name)
        self.clear()

        # Load the selected YAML file
        self.load_yaml(file_path)

    def clear(self):
        # Clear current widgets from settings_layout

        while self.settings_layout.count():
            child = self.settings_layout.takeAt(0)
            print(child)
            if child.widget():
                child.widget().deleteLater()

        # Clear current widgets from keybindings_layout
        while self.keybindings_layout.count():
            print(child)
            child = self.keybindings_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def save_as_new_profile(self):
        new_profile_name = self.new_profile_name_edit.text().strip()
        if not new_profile_name:
            QMessageBox.warning(self, 'Warning', 'Please enter a profile name.')
            return
        file_path = os.path.join('config/user', new_profile_name + '.yaml')
        self.save_yaml(file_path)
        # with open(file_path, 'w') as file:
        #     yaml.safe_dump(self.values, file)
        QMessageBox.information(self, 'Success', f'Profile saved as {new_profile_name}.')
        # Refresh the dropdown
        self.user_profiles_dropdown.clear()
        self.user_profiles_dropdown.addItems(get_yaml_files_in_user_dir())
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
    def load_default_options(self, file_path=None):

        if file_path is None:
            # Try to get the last used config
            file_path = self.get_last_config()

        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            self.values = yaml.safe_load(file)

        if 'keybindings' not in self.values:
            self.values['keybindings'] = DEFAULT_KEYBINDINGS
        # Check for new or missing keybindings and add them from DEFAULT_KEYBINDINGS
        for key, value in DEFAULT_KEYBINDINGS.items():
            if key not in self.values['keybindings']:
                self.values['keybindings'][key] = value
        for key, value in self.values.items():
            if key == 'keybindings':
                continue
            label = QLabel(key)
            self.settings_layout.addWidget(label)

            # if key in self.default_options:  # Add QComboBox for keys found in default_options
            #     combo_box = QComboBox()
            #     combo_box.setObjectName(key)
            #     combo_box.addItems(self.default_options[key])
            #     combo_box.setCurrentText(str(value))
            #     self.settings_layout.addWidget(combo_box)

            if isinstance(value, bool):  # Add QCheckBox for boolean values
                checkbox = QCheckBox()
                checkbox.setObjectName(key)
                checkbox.setChecked(value)
                self.settings_layout.addWidget(checkbox)

            else:  # Add QLineEdit and QPushButton for string values
                layout = QHBoxLayout()

                line_edit = QLineEdit(str(value))
                line_edit.setObjectName(key)

                layout.addWidget(line_edit)
                if key in folder_options:
                    browse_button = QPushButton('Browse...')
                    browse_button.clicked.connect(functools.partial(self.browse, line_edit))
                    layout.addWidget(browse_button)
                self.settings_layout.addLayout(layout)

        # Load keybindings
        keybindings_label = QLabel("Keybindings")
        self.keybindings_layout.addWidget(keybindings_label)

        for key, value in self.values['keybindings'].items():
            layout = QHBoxLayout()

            label = QLabel(value['name'])
            layout.addWidget(label)

            key_sequence_edit = QKeySequenceEdit(QKeySequence(value['shortcut']))
            key_sequence_edit.setObjectName(f"keybinding_{key}")
            layout.addWidget(key_sequence_edit)

            self.keybindings_layout.addLayout(layout)

    def browse(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select a directory')
        if folder_path:
            line_edit.setText(folder_path)

    def save_yaml(self, file_path=None):
        # Save values from settings layout
        self.values["keybindings"] = {}

        self.save_widgets_from_layout(self.settings_layout)
        self.save_widgets_from_layout(self.keybindings_layout)

        # Save values from keybindings layout
        # for i in range(self.keybindings_layout.count()):
        #     item = self.keybindings_layout.itemAt(i)
        #     layout = item.layout()
        #     if layout:
        #         for j in range(layout.count()):
        #             widget = layout.itemAt(j).widget()
        #             if isinstance(widget, QKeySequenceEdit):
        #                 key = widget.objectName().replace("keybinding_", "")
        #                 new_shortcut = widget.keySequence().toString()
        #                 self.values["keybindings"] = {}
        #                 self.values["keybindings"][key] = {}
        #                 self.values["keybindings"][key]["shortcut"] = new_shortcut
        #                 self.values["keybindings"][key]["name"] = DEFAULT_KEYBINDINGS[key]["name"]

        # The rest of the saving process remains the same
        # Decide which path to save to


        profile_name = self.user_profiles_dropdown.currentText()

        if profile_name == "":
            #print("Empty Profile name, setting to: settings.yaml")
            profile_name = "settings.yaml"

        #print("INITIAL PRINT", file_path, profile_name)
        if not file_path:
            profile_name = f'{profile_name}.yaml' if not profile_name.endswith(".yaml") else profile_name
            file_path = os.path.join('config/user', profile_name)
        # if file_path == None:
        #     if profile_name != "":
        #         file_path = os.path.join('config/user', profile_name + '.yaml')
        #     else:
        #         file_path = "config/user/settings.yaml"
        # else:
        #     profile_name = f'{profile_name}.yaml' if not profile_name.endswith(".yaml") else profile_name
        #     file_path = os.path.join('config/user', profile_name)
        #print("saving to ", file_path)

        with open(file_path, 'w') as file:
            yaml.safe_dump(self.values, file)

        # with open(file_path, 'w') as file:
        #     yaml.safe_dump(self.values, file)

        QMessageBox.information(self, 'Success', 'YAML file saved successfully.')

        from ainodes_frontend.base.settings import load_settings

        #print("Loading settings from", file_path)
        self.set_last_config(file_path)

        load_settings(file_path)
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
                elif isinstance(widget, QKeySequenceEdit):
                    #print("found keysequence to save", widget.keySequence().toString())
                    key = widget.objectName().replace("keybinding_", "")
                    new_shortcut = widget.keySequence().toString()
                    self.values["keybindings"][key] = {}
                    self.values["keybindings"][key]["shortcut"] = new_shortcut
                    self.values["keybindings"][key]["name"] = DEFAULT_KEYBINDINGS[key]["name"]

            else:
                self.save_widgets_from_layout(item)

            # elif isinstance(item, QLayout):
            #     self.save_widgets_from_layout(item.layout())
    def set_last_config(self, file_path):
        """
        Store the given file path in last_config.yaml.
        """
        last_config_path = os.path.join('config', 'last_config.yaml')
        with open(last_config_path, 'w') as file:
            yaml.safe_dump({"last_config": file_path}, file)

    def get_last_config(self):
        """
        Retrieve the file path from last_config.yaml.
        """
        last_config_path = os.path.join('config', 'last_config.yaml')
        if os.path.exists(last_config_path):
            with open(last_config_path, 'r') as file:
                data = yaml.safe_load(file)
                return data.get('last_config')
        return "config/default_settings.yaml"
    def cancel(self):
        self.close()
