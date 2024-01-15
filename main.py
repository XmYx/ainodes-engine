"""ainodes-engine main"""
#!/usr/bin/env python3
import datetime

import torch
import yaml
from PyQt6.QtCore import QFile
from PyQt6.QtWidgets import QTextEdit
from qtpy.QtWidgets import QProgressBar

from qtpy.QtCore import QPropertyAnimation, QEasingCurve
import os, sys
sys.path.append(os.getcwd())
from ainodes_frontend.base.settings import Settings, save_settings, load_settings

start_time = datetime.datetime.now()
print(f"Start aiNodes, please wait. {start_time}")

import os
os.environ["QT_API"] = "pyqt6"
# os.environ["FORCE_QT_API"] = "1"
# os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

import logging
import os
logging.basicConfig(level='ERROR')

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
# if "Linux" in platform.platform():
#     import subprocess
#     subprocess.check_call(["pip", "install", "triton==2.0.0"])
if "linux" in platform.platform().lower():

    try:
        try:
            from gi.repository import Gtk
        except:
            subprocess.check_call(["pip", "install", "pygobject"])
            from gi.repository import Gtk

        # Get the current GTK settings
        settings = Gtk.Settings.get_default()
        gtk_theme = settings.get_property("gtk-theme-name")
        # Check the GTK theme
        if gtk_theme.endswith("dark"):
            qss_file = "ainodes_frontend/qss/nodeeditor-dark-linux.qss"
        else:
            qss_file = "ainodes_frontend/qss/nodeeditor.qss"
    except:
        gs.qss = "qss/nodeeditor-dark-linux.qss"
elif "Windows" in platform.platform():

    settings = QtCore.QSettings('HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize',
                         QtCore.QSettings.Format.NativeFormat)
    theme = settings.value('AppsUseLightTheme')
    if theme == 0:
        gs.qss = "qss/nodeeditor-dark.qss"
    else:
        gs.qss = "qss/nodeeditor.qss"
    import ctypes
    myappid = u'mycompany.myproduct.subproduct.version'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
else:
    gs.qss = "qss/nodeeditor-dark.qss"

print("QSS SET", gs.qss)


import subprocess
import os

def clone_and_install(repo_url, install_directory='src'):
    # Save the current working directory
    original_dir = os.getcwd()

    try:
        # Clone the repository
        if not os.path.exists(install_directory):
            os.makedirs(install_directory)

        repo_name = repo_url.split('/')[-1]
        clone_path = os.path.join(install_directory, "deforum")

        if not os.path.exists(clone_path):
            print(f"Cloning {repo_url} into {clone_path}...")
            subprocess.run(["git", "clone", repo_url, clone_path], check=True)
        else:
            print(f"Repository already cloned at {clone_path}")

        # Change directory to the cloned repository
        os.chdir(clone_path)

        # Install using pip
        print("Installing the package...")
        subprocess.run(["pip", "install", "-e", "."], check=True)

        print("Installation completed.")
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
        print("Returned to the original directory.")

# URL of the repository
repo_url = "https://github.com/XmYx/deforum-studio"
#clone_and_install(repo_url)


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
#append_subfolders_to_syspath('src')

# import torch, platform

def get_torch_device():
    if "macOS" in platform.platform():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            return torch.device(torch.cuda.current_device())
        else:
            return torch.device("cpu")

def hijack_comfy_paths():

    import folder_paths

    supported_pt_extensions = set(['.ckpt', '.pt', '.bin', '.pth', '.safetensors'])

    #folder_names_and_paths = {}

    base_path = os.path.dirname(os.path.realpath(__file__))
    models_dir = os.path.join(base_path, "models")
    folder_paths.folder_names_and_paths["checkpoints"] = ([os.path.join(models_dir, "checkpoints")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["configs"] = ([os.path.join(models_dir, "configs")], [".yaml"])

    folder_paths.folder_names_and_paths["loras"] = ([os.path.join(models_dir, "loras")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["vae"] = ([os.path.join(models_dir, "vae")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["clip"] = ([os.path.join(models_dir, "clip")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["unet"] = ([os.path.join(models_dir, "unet")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["clip_vision"] = ([os.path.join(models_dir, "clip_vision")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["style_models"] = ([os.path.join(models_dir, "style_models")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["embeddings"] = ([os.path.join(models_dir, "embeddings")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["diffusers"] = ([os.path.join(models_dir, "diffusers")], ["folder"])
    folder_paths.folder_names_and_paths["vae_approx"] = ([os.path.join(models_dir, "vae_approx")], supported_pt_extensions)

    folder_paths.folder_names_and_paths["controlnet"] = (
    [os.path.join(models_dir, "controlnet"), os.path.join(models_dir, "t2i_adapter")], supported_pt_extensions)
    folder_paths.folder_names_and_paths["gligen"] = ([os.path.join(models_dir, "gligen")], supported_pt_extensions)

    folder_paths.folder_names_and_paths["upscale_models"] = ([os.path.join(models_dir, "upscale_models")], supported_pt_extensions)

    folder_paths.folder_names_and_paths["custom_nodes"] = ([os.path.join(base_path, "custom_nodes")], [])

    folder_paths.folder_names_and_paths["hypernetworks"] = ([os.path.join(models_dir, "hypernetworks")], supported_pt_extensions)

    folder_paths.folder_names_and_paths["classifiers"] = ([os.path.join(models_dir, "classifiers")], {""})

    folder_paths.output_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    folder_paths.temp_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
    #folder_paths.input_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), "input")

    folder_paths.filename_list_cache = {}

    # from ainodes_frontend.base import modelmanagement_hijack
    # import comfy.model_management
    # comfy.model_management.unet_inital_load_device = modelmanagement_hijack.unet_inital_load_device


if __name__ == "__main__":
    from ainodes_frontend.base import settings

    settings.init_globals()

    gs.args = get_args()
    gs.device = get_torch_device()




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
            self.comfy_ui_process = None
            #self.startComfyUI()

        def startComfyUI(self):
            # Start the ComfyUI subprocess
            comfy_ui_path = os.path.join('src', 'deforum', 'src', 'ComfyUI', 'main.py')
            if os.path.exists(comfy_ui_path):
                self.comfy_ui_process = subprocess.Popen([sys.executable, comfy_ui_path, '--extra-model-paths-config', 'config/comfy_paths.yaml'])
            else:
                print(f"ComfyUI main.py not found at {comfy_ui_path}")

        def cleanUp(self):
            # Terminate the ComfyUI subprocess
            if self.comfy_ui_process:
                self.comfy_ui_process.terminate()
                self.comfy_ui_process.wait()  # Wait for the process to terminate
            print('Application closing')

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


    class CustomOutputStream:
        def __init__(self, update_text_func):
            self.update_text_func = update_text_func

        def write(self, text):
            self.update_text_func(text)

        def flush(self):
            pass  # This can be an empty method

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
            # Initialize the text edit for displaying output
            self.text_display = QTextEdit(self)
            self.text_display.setGeometry(10, pixmap.height() - 100, pixmap.width() - 20,
                                          75)  # Adjust geometry as needed
            self.text_display.setReadOnly(True)
            # Other initialization...

        def append_text(self, text):
            self.text_display.moveCursor(QtGui.QTextCursor.MoveOperation.End)
            self.text_display.insertPlainText(text)
            # Optionally, you can ensure the latest output is visible:
            self.text_display.ensureCursorVisible()

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

    res = ''
    file = QFile(os.path.join(os.path.dirname(__file__), 'ainodes_frontend',gs.qss))
    file.open(QFile.ReadOnly | QFile.Text)
    stylesheet = file.readAll()
    res += "\n" + str(stylesheet, encoding='utf-8')
    ainodes_qapp.setApplicationName("aiNodes - Engine")
    ainodes_qapp.setStyleSheet(res)
    splash = CustomSplashScreen(splash_pix)
    splash.show()

    sys.stdout = CustomOutputStream(splash.append_text)
    from deforum.generators.comfy_utils import ensure_comfy
    ensure_comfy('src/ComfyUI')
    hijack_comfy_paths()
    load_settings()
    from ainodes_frontend.comfy_fns.adapter_nodes import was_adapter_node

    base_folder = 'ainodes_frontend/nodes'
    if gs.args.update:
        print("Updating node requirements")
        update_all_nodes_req()

    total_steps = 0
    current_step = 0
    #print(os.path.join(base_folder, 'ainodes_frontend', 'nodes'))
    if os.path.isdir('ainodes_frontend/nodes'):
        valid_subdirectories = [folder for folder in os.listdir(os.path.join(base_folder))
                                if "__pycache__" not in folder
                                and "_nodes" in folder
                                and os.path.isdir(os.path.join(base_folder, folder))]
        print(valid_subdirectories)
        total_steps += len(valid_subdirectories)

        # if os.path.isdir(base_folder):
        #     for folder in os.listdir(base_folder):
        #         folder_path = os.path.join(base_folder, folder)
        #         if "__pycache__" not in folder_path and "_nodes" in folder_path:
        #             if os.path.isdir(folder_path):
        #                 import_nodes_from_subdirectories(folder_path)
        #if os.path.isdir(base_folder):
        for folder in valid_subdirectories:
            folder_path = os.path.join(base_folder, folder)
            import_nodes_from_subdirectories(folder_path)
            percentage_complete = int((current_step / total_steps) * 100)
            splash.setProgress(percentage_complete)
            current_step += 1

    from ainodes_frontend.base import CalculatorWindow

    wnd = CalculatorWindow(ainodes_qapp)
    #wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), gs.qss)

    # loadStylesheets(
    #     wnd.stylesheet_filename,
    #     wnd.stylesheet_filename
    # )

    wnd.show()

    #wnd.showFullScreen()
    wnd.showMaximized()
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
    print("Theme set to:", gs.qss)
    sys.exit(ainodes_qapp.exec())



