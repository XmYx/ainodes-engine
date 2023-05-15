"""ainodes-engine main"""
#!/usr/bin/env python3

import sys
import os
import subprocess
import platform
import argparse
from types import SimpleNamespace

from qtpy import QtCore, QtQuick
from qtpy.QtQuick import QSGRendererInterface
from qtpy.QtCore import QCoreApplication
from qtpy import QtGui
from qtpy.QtWidgets import QApplication
from ainodes_frontend import singleton as gs

import ainodes_frontend.qss.nodeeditor_dark_resources
from ainodes_frontend.base.settings import save_settings, load_settings
from ainodes_frontend.node_engine.utils import loadStylesheets

# Set environment variable QT_API to use PySide6
os.environ["QT_API"] = "pyside6"
# Install Triton if running on Linux
if "Linux" in platform.platform():
    print(platform.platform())
    subprocess.check_call(["pip", "install", "triton==2.0.0"])
if "Linux" in platform.platform():
    qss = "ainodes_frontend/qss/nodeeditor-dark-linux.qss"
else:
    qss = "ainodes_frontend/qss/nodeeditor-dark.qss"

# Initialize global variables
gs.obj = {}
gs.values = {}
gs.current = {}
gs.nodes = {}
gs.system = SimpleNamespace()
gs.busy = False

gs.models = {}
gs.token = ""
gs.use_deforum_loss = None
gs.highlight_sockets = True
gs.loaded_sd = ""
gs.current = {}
gs.loaded_vae = ""
gs.logging = None
gs.debug = None


gs.system.textual_inversion_dir = "models/embeddings"
try:
    import xformers
    gs.system.xformer = True
except:
    gs.system.xformer = False

gs.current["sd_model"] = None
gs.current["inpaint_model"] = None
gs.loaded_vae = ""


# Parse command line arguments
parser = argparse.ArgumentParser()
#Local Huggingface Hub Cache
parser.add_argument("--local_hf", action="store_true")
parser.add_argument("--skip_base_nodes", action="store_true")
parser.add_argument("--light", action="store_true")
parser.add_argument("--skip_update", action="store_true")
parser.add_argument("--torch2", action="store_true")
parser.add_argument("--no_console", action="store_true")
args = parser.parse_args()
gs.args = args

# Set environment variables for Hugging Face cache if not using local cache
if not args.local_hf:
    print("Using HF Cache in app dir")
    os.makedirs("hf_cache", exist_ok=True)
    os.environ["HF_HOME"] = "hf_cache"
# Set up high-quality QSurfaceFormat object with OpenGL 3.3 and 8x antialiasing
qs_format = QtGui.QSurfaceFormat()
qs_format.setVersion(3, 3)
qs_format.setSamples(8)
qs_format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
QtGui.QSurfaceFormat.setDefaultFormat(qs_format)
def eventListener(*args, **kwargs):
    print("EVENT")
#QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
#QtQuick.QQuickWindow.setGraphicsApi(QSGRendererInterface.OpenGLRhi)
if __name__ == "__main__":

    # make app
    app = QApplication(sys.argv)

    QCoreApplication.instance().aboutToQuit.connect(eventListener)

    # Load style sheet from a file
    """if not args.light:
        with open("ainodes_frontend/qss/nodeeditor-dark.qss", "r", encoding="utf-8") as f:
            style_sheet = f.read()
            app.setStyleSheet(style_sheet)"""
    icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
    app.setWindowIcon(icon)
    app.setApplicationName("aiNodes - engine")
    load_settings()
    from ainodes_frontend.base import CalculatorWindow
    # Create and show the main window
    wnd = CalculatorWindow(app)


    wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), qss)
    loadStylesheets(
        os.path.join(os.path.dirname(__file__), qss),
        wnd.stylesheet_filename
    )

    if not args.skip_base_nodes:
        wnd.node_packages.list_widget.setCurrentRow(0)
        if not os.path.isdir('custom_nodes/ainodes_engine_base_nodes'):
            wnd.node_packages.download_repository()
        else:
            wnd.node_packages.update_repository(args.skip_update)
    wnd.show()
    wnd.nodesListWidget.addMyItems()
    wnd.onFileNew()
    if args.torch2 == True:
        from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.sd_optimizations.sd_hijack import apply_optimizations
        apply_optimizations()
    sys.exit(app.exec())
