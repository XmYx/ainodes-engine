"""ainodes-engine main"""
#!/usr/bin/env python3
import datetime

import yaml
from qtpy.QtWidgets import QProgressBar

from qtpy.QtCore import QPropertyAnimation, QEasingCurve
import os, sys
sys.path.append(os.getcwd())
from ainodes_frontend.base.settings import Settings, save_settings, load_settings

start_time = datetime.datetime.now()
print(f"Start aiNodes, please wait. {start_time}")

import os
os.environ["QT_API"] = "pyqt6"
os.environ["FORCE_QT_API"] = "1"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

import platform
import sys

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplashScreen, QApplication




from qtpy import QtCore, QtGui
try:
    import tqdm
except:
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    import tqdm


from ainodes_frontend import singleton as gs
from ainodes_frontend.node_engine.utils import loadStylesheets
from ainodes_frontend.base.args import get_args
from ainodes_frontend.base.import_utils import update_all_nodes_req, import_nodes_from_subdirectories, \
    set_application_attributes

# Set environment variable QT_API to use PySide6
# Install Triton if running on Linux
if "Linux" in platform.platform():
    import subprocess
    subprocess.check_call(["pip", "install", "triton==2.0.0"])
if "Linux" in platform.platform():

    # try:
    #     try:
    #         from gi.repository import Gtk
    #     except:
    #         subprocess.check_call(["pip", "install", "pygobject"])
    #         from gi.repository import Gtk
    #
    #     # Get the current GTK settings
    #     settings = Gtk.Settings.get_default()
    #     gtk_theme = settings.get_property("gtk-theme-name")
    #     # Check the GTK theme
    #     if gtk_theme.endswith("dark"):
    #         qss_file = "ainodes_frontend/qss/nodeeditor-dark-linux.qss"
    #     else:
    #         qss_file = "ainodes_frontend/qss/nodeeditor.qss"
    # except:
    gs.qss = "ainodes_frontend/qss/nodeeditor-dark-linux.qss"
elif "Windows" in platform.platform():

    settings = QtCore.QSettings('HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize',
                         QtCore.QSettings.Format.NativeFormat)
    theme = settings.value('AppsUseLightTheme')
    if theme == 0:
        gs.qss = "ainodes_frontend/qss/nodeeditor-dark.qss"
    else:
        gs.qss = "ainodes_frontend/qss/nodeeditor.qss"
    import ctypes
    myappid = u'mycompany.myproduct.subproduct.version'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
else:
    gs.qss = "ainodes_frontend/qss/nodeeditor-dark.qss"

print("QSS SET", gs.qss)
def append_subfolders_to_syspath(base_path):
    """
    Append all first-level subfolders of the given base_path to sys.path.

    :param base_path: The path of the base folder.
    """
    for name in os.listdir(base_path):
        full_path = os.path.join(base_path, name)
        if os.path.isdir(full_path):
            print("Addint", full_path, "to path")
            sys.path.append(full_path)

# Assuming 'src' is in the current directory
append_subfolders_to_syspath('src')

if __name__ == "__main__":
    from ainodes_frontend.base import settings

    settings.init_globals()

    gs.args = get_args()

    # Set environment variables for Hugging Face cache if not using local cache
    if gs.args.local_hf:
        print("Using HF Cache in app dir")
        os.makedirs("hf_cache", exist_ok=True)
        os.environ["HF_HOME"] = "hf_cache"

    if gs.args.highdpi:
        print("Setting up Hardware Accelerated GUI")
        from qtpy.QtQuick import QSGRendererInterface

        # Set up high-quality QSurfaceFormat object with OpenGL 3.3 and 8x antialiasing
        qs_format = QtGui.QSurfaceFormat()
        qs_format.setVersion(3, 3)
        qs_format.setSamples(8)
        qs_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        QtGui.QSurfaceFormat.setDefaultFormat(qs_format)
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    set_application_attributes(QApplication, gs.args)


    # The main application
    class Application(QApplication):

        def __init__(self, args):
            QApplication.__init__(self, args)

        def cleanUp(self):
            # THIS ACTUALLY WORKS
            print('closing')
    # make app
    ainodes_qapp = Application(sys.argv)
    ainodes_qapp.aboutToQuit.connect(ainodes_qapp.cleanUp)

    from ainodes_frontend.icon import icon

    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(icon)
    appIcon = QtGui.QIcon(pixmap)
    ainodes_qapp.setWindowIcon(appIcon)

    splash_pix = QtGui.QPixmap(
        "ainodes_frontend/qss/icon.ico")  # Replace "splash.png" with the path to your splash screen image
    from qtpy.QtCore import QRect
    from qtpy.QtGui import QPainter


    class CustomSplashScreen(QSplashScreen):
        def __init__(self, pixmap):
            super(CustomSplashScreen, self).__init__(pixmap)
            self.progressBar = QProgressBar(self)

            custom_style = """
            QProgressBar {
                background-color: #3a3a3a;
                border: 1px solid #2e2e2e;
                border-radius: 4px;
                text-align: center;
                color: white;
            }

            QProgressBar::chunk {
                background-color: #FF8C00;
                border-radius: 2px;
            }
            """

            self.progressBar.setStyleSheet(custom_style)

            self.progressBar.setGeometry(10, pixmap.height() - 25, pixmap.width() - 20, 20)
            self.progressBar.setValue(0)




        def setProgress(self, value):
            self.progressBar.setValue(value)
            self.repaint()  # Ensures that the splash screen is redrawn

        def drawContents(self, painter: QPainter):
            super(CustomSplashScreen, self).drawContents(painter)

            # Set the font for the text
            font = painter.font()
            font.setPointSize(14)
            painter.setFont(font)

            # Compute text position (almost bottom center)
            text = "Loading aiNodes"
            metrics = painter.fontMetrics()
            text_width = metrics.width(text)
            text_height = metrics.height()
            from PyQt6.QtGui import QColor

            text_x = int((self.width() - text_width) / 2)
            text_y = int(self.height() - 2 * text_height) - 195  # Adjust this value as needed to position the text
            # Draw the text background with 25% opacity
            painter.setBrush(QColor(0, 0, 0, 64))  # RGBA: 25% opacity
            painter.setPen(Qt.NoPen)  # No border
            painter.drawRect(text_x - 5, text_y - 5, text_width + 10,
                             text_height + 10)  # A little padding around the text

            # Draw the text itself in white
            painter.setPen(Qt.white)
            painter.drawText(text_x, text_y + metrics.ascent(),
                             text)  # metrics.ascent() is used to vertically align the text

            rect = QRect(self.progressBar.geometry())
            self.progressBar.render(painter, rect.topLeft())


    splash = CustomSplashScreen(splash_pix)
    splash.show()

    load_settings()

    base_folder = 'ai_nodes'
    if gs.args.update:
        print("Updating node requirements")
        update_all_nodes_req()

    total_steps = 0


    valid_subdirectories = [folder for folder in os.listdir(base_folder)
                            if "__pycache__" not in folder
                            and "_nodes" in folder
                            and os.path.isdir(os.path.join(base_folder, folder))]

    total_steps += len(valid_subdirectories)

    # if os.path.isdir(base_folder):
    #     for folder in os.listdir(base_folder):
    #         folder_path = os.path.join(base_folder, folder)
    #         if "__pycache__" not in folder_path and "_nodes" in folder_path:
    #             if os.path.isdir(folder_path):
    #                 import_nodes_from_subdirectories(folder_path)
    current_step = 0
    #if os.path.isdir(base_folder):
    for folder in valid_subdirectories:
        folder_path = os.path.join(base_folder, folder)
        import_nodes_from_subdirectories(folder_path)
        percentage_complete = int((current_step / total_steps) * 100)
        splash.setProgress(percentage_complete)
        current_step += 1

    from ainodes_frontend.base import CalculatorWindow

    ainodes_qapp.setApplicationName("aiNodes - Engine")
    wnd = CalculatorWindow(ainodes_qapp)
    wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), gs.qss)

    loadStylesheets(
        os.path.join(gs.qss),
        wnd.stylesheet_filename
    )

    wnd.show()
    wnd.fade_in_animation()
    wnd.nodesListWidget.addMyItems()
    wnd.onFileNew()

    # def fade_out_animation():
    splash.setProgress(100)
    splash_fade_animation = QPropertyAnimation(splash, b"windowOpacity")
    splash_fade_animation.setDuration(500)  # Set the duration of the animation in milliseconds
    splash_fade_animation.setStartValue(1.0)  # Start with opacity 1.0 (fully visible)
    splash_fade_animation.setEndValue(0.0)  # End with opacity 0.0 (transparent)
    splash_fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)  # Apply easing curve to the animation
    splash_fade_animation.finished.connect(
        lambda: splash.finish(wnd))  # Close the splash screen when animation finishes
    splash_fade_animation.start()

    end_time = datetime.datetime.now()
    print(f"Initialization took: {end_time - start_time}")
    sys.exit(ainodes_qapp.exec())



