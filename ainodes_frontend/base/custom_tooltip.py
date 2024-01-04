from qtpy.QtWidgets import QLabel, QDialog, QVBoxLayout, QApplication, QGraphicsOpacityEffect
from qtpy.QtGui import QPixmap, QFont
from qtpy.QtCore import Qt, QTimer, QPropertyAnimation
from qtpy import QtWidgets


class CustomTooltip(QDialog):
    def __init__(self, parent=None):
        super(CustomTooltip, self).__init__(parent)
        self.label_icon = QLabel(self)
        self.label_text = QLabel(self)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label_icon)
        layout.addWidget(self.label_text)
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        self.animation = QPropertyAnimation(self.effect, b"opacity")

    def show(self, icon, text, pos):
        self.label_icon.setPixmap(QPixmap(icon))
        self.label_text.setText(text)
        self.move(pos)
        super().show()


class MyTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, *args, **kwargs):
        super(MyTreeWidgetItem, self).__init__(*args, **kwargs)
        self.tooltip = CustomTooltip()
        self.timer = QTimer()
        self.timer.timeout.connect(self.showTooltip)

    def showTooltip(self):
        if QApplication.instance().activeWindow() is not None:
            self.tooltip.show(self.icon().pixmap(), self.text(), QApplication.instance().activeWindow().cursor().pos())

    def enterEvent(self, event):
        #print("SHOWING")

        self.timer.start(500) # wait 500 ms before showing the tooltip

    def leaveEvent(self, event):
        self.timer.stop()
        self.tooltip.hide()