"""ainodes-engine main"""
#!/usr/bin/env python3
import datetime

from PyQt6.QtCore import QPropertyAnimation, QEasingCurve

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

from ainodes_frontend import singleton as gs
from ainodes_frontend.base.settings import load_settings, init_globals
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
    gs.qss = "ainodes_frontend/qss/nodeeditor-dark-linux.qss"
elif "Windows" in platform.platform():
    gs.qss = "ainodes_frontend/qss/nodeeditor-dark.qss"
    import ctypes
    myappid = u'mycompany.myproduct.subproduct.version'  # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
else:
    gs.qss = "ainodes_frontend/qss/nodeeditor-dark.qss"



init_globals()

gs.args = get_args()

# Set environment variables for Hugging Face cache if not using local cache
if not gs.args.local_hf:
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
    #QtQuick.QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGLRhi)


set_application_attributes(QApplication, gs.args)


# make app
ainodes_qapp = QApplication(sys.argv)
from ainodes_frontend.icon import icon
pixmap = QtGui.QPixmap()
pixmap.loadFromData(icon)
appIcon = QtGui.QIcon(pixmap)
ainodes_qapp.setWindowIcon(appIcon)

splash_pix = QtGui.QPixmap("ainodes_frontend/qss/icon.ico")  # Replace "splash.png" with the path to your splash screen image
splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
splash.show()


load_settings()
base_folder = 'custom_nodes'
if gs.args.update:
    update_all_nodes_req()


for folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, folder)
    if "__pycache__" not in folder_path and "_nodes" in folder_path:
        if os.path.isdir(folder_path):
            import_nodes_from_subdirectories(folder_path)



from ainodes_frontend.base import CalculatorWindow
ainodes_qapp.setApplicationName("aiNodes - Engine")
wnd = CalculatorWindow(ainodes_qapp)
wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), gs.qss)

loadStylesheets(
    os.path.join(os.path.dirname(__file__), gs.qss),
    wnd.stylesheet_filename
)

wnd.show()
wnd.fade_in_animation()
wnd.nodesListWidget.addMyItems()
wnd.onFileNew()


#def fade_out_animation():
splash_fade_animation = QPropertyAnimation(splash, b"windowOpacity")
splash_fade_animation.setDuration(1500)  # Set the duration of the animation in milliseconds
splash_fade_animation.setStartValue(1.0)  # Start with opacity 1.0 (fully visible)
splash_fade_animation.setEndValue(0.0)  # End with opacity 0.0 (transparent)
splash_fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)  # Apply easing curve to the animation
splash_fade_animation.finished.connect(lambda: splash.finish(wnd))  # Close the splash screen when animation finishes
splash_fade_animation.start()


#fade_out_animation()
#splash.finish(wnd)


end_time = datetime.datetime.now()
print(f"Initialization took: {end_time - start_time}")


if __name__ == "__main__":
    ainodes_qapp.exec()
