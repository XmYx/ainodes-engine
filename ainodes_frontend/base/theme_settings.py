"""Theme picker dialog and the menu actions behind it."""
import os

from qtpy import QtWidgets

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import theme


class ThemePreferencesDialog(QtWidgets.QDialog):
    """Pick a theme and apply it live."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theme")
        self.setMinimumWidth(320)
        layout = QtWidgets.QVBoxLayout(self)

        layout.addWidget(QtWidgets.QLabel("Colour theme"))
        self.themeComboBox = QtWidgets.QComboBox()
        self.themeComboBox.addItems(theme.available_themes() + ["system"])
        current = theme.current_theme()
        index = self.themeComboBox.findText(current)
        if index >= 0:
            self.themeComboBox.setCurrentIndex(index)
        layout.addWidget(self.themeComboBox)

        self.rememberCheckBox = QtWidgets.QCheckBox("Remember this choice")
        self.rememberCheckBox.setChecked(True)
        layout.addWidget(self.rememberCheckBox)

        note = QtWidgets.QLabel(
            "Switching re-colours the canvas as well as the window chrome."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(buttons)
        buttons.accepted.connect(self.applyTheme)
        buttons.rejected.connect(self.reject)

        # live preview while browsing the list
        self.themeComboBox.currentTextChanged.connect(self.previewTheme)
        self._original = current

    def previewTheme(self, name):
        theme.apply_theme(name, window=self.parent())

    def applyTheme(self):
        name = self.themeComboBox.currentText()
        theme.apply_theme(name, window=self.parent())
        if self.rememberCheckBox.isChecked():
            save_theme_preference(name)
        self.accept()

    def reject(self):
        theme.apply_theme(self._original, window=self.parent())
        super().reject()


def save_theme_preference(name):
    """Persist the theme into the active settings file."""
    try:
        from ainodes_frontend.base.settings import get_last_config, save_settings

        gs.prefs.theme = name
        gs.prefs.settings["theme"] = name
        destination = get_last_config() or os.path.join("config", "default_settings.yaml")
        save_settings(gs.prefs, destination)
    except Exception as exc:
        print(f"[theme] could not save the theme preference: {exc}")


def toggle_theme(window=None):
    """Flip between dark and light - what the toolbar/menu toggle calls."""
    name = "light" if theme.current_theme() == "dark" else "dark"
    theme.apply_theme(name, window=window)
    save_theme_preference(name)
    return name
