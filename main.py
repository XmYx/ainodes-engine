"""ainodes-engine main"""
#!/usr/bin/env python3

import sys
import os
import subprocess
import platform
import argparse
from qtpy import QtGui
from qtpy.QtWidgets import QApplication
from ainodes_frontend import singleton as gs
from ainodes_frontend.base import CalculatorWindow
import ainodes_frontend.qss.nodeeditor_dark_resources
from ainodes_frontend.node_engine.utils import loadStylesheets

# Set environment variable QT_API to use PySide6
os.environ["QT_API"] = "pyside6"
# Install Triton if running on Linux
if "Linux" in platform.platform():
    print(platform.platform())
    subprocess.check_call(["pip", "install", "triton==2.0.0"])

# Initialize global variables
gs.obj = {}
gs.values = {}
gs.current = {}
gs.nodes = {}
gs.current["sd_model"] = None
gs.current["inpaint_model"] = None

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--local_hf", action="store_true")
parser.add_argument("--whisper", action="store_true")
parser.add_argument("--skip_base_nodes", action="store_true")
parser.add_argument("--light", action="store_true")
args = parser.parse_args()

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

# make app
app = QApplication(sys.argv)
# Load style sheet from a file
"""if not args.light:
    with open("ainodes_frontend/qss/nodeeditor-dark.qss", "r", encoding="utf-8") as f:
        style_sheet = f.read()
        app.setStyleSheet(style_sheet)"""
icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
app.setWindowIcon(icon)
app.setApplicationName("aiNodes - engine")

# Create and show the main window
wnd = CalculatorWindow()
wnd.stylesheet_filename = os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss")
loadStylesheets(
    os.path.join(os.path.dirname(__file__), "ainodes_frontend/qss/nodeeditor-dark.qss"),
    wnd.stylesheet_filename
)

#sys.stdout = wnd.text_widget
#sys.stderr = wnd.text_widget
#sys.stdin = wnd.text_widget


if not args.skip_base_nodes:
    wnd.node_packages.list_widget.setCurrentRow(0)
    if not os.path.isdir('custom_nodes/ainodes_engine_base_nodes'):
        wnd.node_packages.download_repository()
    else:
        wnd.node_packages.update_repository()
wnd.show()


#import_nodes_from_directory("ainodes_frontend/nodes/exec_nodes")
#import_nodes_from_directory("ainodes_frontend/nodes/image_nodes")
#import_nodes_from_directory("ainodes_frontend/nodes/torch_nodes")
#import_nodes_from_directory("ainodes_frontend/nodes/video_nodes")
#if args.whisper:
#    import_nodes_from_directory("ainodes_frontend/nodes/audio_nodes")

wnd.nodesListWidget.addMyItems()


#wnd.setStyleSheet(style_sheet)
sys.exit(app.exec())
