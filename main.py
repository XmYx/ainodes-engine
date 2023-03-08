import os, sys
os.environ['QT_API'] = 'pyside6'
import platform
import subprocess

if "Linux" in platform.platform():
    print(platform.platform())
    subprocess.run(["pip", "install", "triton==2.0.0"])

from qtpy import QtOpenGL, QtGui
from qtpy.QtWidgets import QApplication
from ainodes_backend.singleton import Singleton

gs = Singleton()
gs.obj = {}
gs.values = {}

sys.path.insert(0, os.path.join( os.path.dirname(__file__), "..", ".." ))
os.makedirs('hf_cache', exist_ok=True)
os.environ['HF_HOME'] = 'hf_cache'
from ainodes_frontend.nodes.base.node_window import CalculatorWindow
# Create a high-quality QSurfaceFormat object with OpenGL 3.3 and 8x antialiasing
format = QtGui.QSurfaceFormat()
format.setVersion(3, 3)
format.setSamples(8)
format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
#format.setOption(QtGui.QSurfaceFormat.HighDpiScaling)
QtGui.QSurfaceFormat.setDefaultFormat(format)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Load your style sheet from a file
    with open("ainodes_frontend/qss/nodeeditor-dark.qss", "r") as f:
        style_sheet = f.read()
    icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
    app.setWindowIcon(icon)
    app.setApplicationName("aiNodes - engine")
    # Set the style sheet for your entire application

    # print(QStyleFactory.keys())
    #app.setStyle('Fusion')
    wnd = CalculatorWindow()
    wnd.show()
    wnd.setStyleSheet(style_sheet)
    sys.exit(app.exec_())
