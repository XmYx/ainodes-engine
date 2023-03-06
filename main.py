import os, sys

from qtpy import QtOpenGL, QtGui
from qtpy.QtWidgets import QApplication
from backend.singleton import Singleton

gs = Singleton()
gs.obj = {}
gs.values = {}

sys.path.insert(0, os.path.join( os.path.dirname(__file__), "..", ".." ))
os.makedirs('hf_cache', exist_ok=True)
os.environ['HF_HOME'] = 'hf_cache'
from nodes.base.node_window import CalculatorWindow
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
    with open("qss/nodeeditor-dark.qss", "r") as f:
        style_sheet = f.read()

    # Set the style sheet for your entire application

    # print(QStyleFactory.keys())
    #app.setStyle('Fusion')
    wnd = CalculatorWindow()
    wnd.show()
    wnd.setStyleSheet(style_sheet)
    sys.exit(app.exec_())
