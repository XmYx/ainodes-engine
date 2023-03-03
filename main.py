import os, sys
from qtpy.QtWidgets import QApplication
from singleton import Singleton

gs = Singleton()
gs.obj = {}
gs.values = {}

sys.path.insert(0, os.path.join( os.path.dirname(__file__), "..", ".." ))

from nodes.base.calc_window import CalculatorWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # print(QStyleFactory.keys())
    app.setStyle('Fusion')

    wnd = CalculatorWindow()
    wnd.show()

    sys.exit(app.exec_())
