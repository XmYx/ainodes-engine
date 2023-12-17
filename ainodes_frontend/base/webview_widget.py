import sys

from qtpy import QtWidgets#, QtWebEngineWidgets
# from qtpy.QtCore import QUrl
# from qtpy.QtGui import QIcon
# from qtpy.QtWidgets import QLineEdit,QPushButton, QToolBar
# from qtpy.QtWebEngineCore import QWebEnginePage
# from qtpy.QtWebEngineWidgets import QWebEngineView


class BrowserWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
# class BrowserWidget(QtWidgets.QWidget):
#
#     def __init__(self):
#         super().__init__()
#
#         self.setWindowTitle('PyQt6 WebEngineWidgets Example')
#
#         layout = QtWidgets.QVBoxLayout()
#         self.setLayout(layout)
#
#         self.toolBar = QToolBar()
#         layout.addWidget(self.toolBar)
#
#         self.backButton = QPushButton()
#         self.backButton.setIcon(QIcon.fromTheme("go-previous"))
#         self.backButton.clicked.connect(self.back)
#         self.toolBar.addWidget(self.backButton)
#
#         self.forwardButton = QPushButton()
#         self.forwardButton.setIcon(QIcon.fromTheme("go-next"))
#         self.forwardButton.clicked.connect(self.forward)
#         self.toolBar.addWidget(self.forwardButton)
#
#         self.addressLineEdit = QLineEdit()
#         self.addressLineEdit.returnPressed.connect(self.load)
#         self.toolBar.addWidget(self.addressLineEdit)
#
#         self.webEngineView = QWebEngineView()
#         self.webEngineView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.WebAttribute.WebGLEnabled, True)
#
#         self.webEngineView.load(QUrl("http://qt.io"))
#         self.webEngineView.page().titleChanged.connect(self.setWindowTitle)
#         self.webEngineView.page().urlChanged.connect(self.urlChanged)
#         layout.addWidget(self.webEngineView)
#
#         self.addressLineEdit.setText("http://qt.io")
#
#     def load(self):
#         url = QUrl.fromUserInput(self.addressLineEdit.text())
#         if url.isValid():
#             self.webEngineView.load(url)
#
#     def back(self):
#         self.webEngineView.page().triggerAction(QWebEnginePage.WebAction.Back)
#
#     def forward(self):
#         self.webEngineView.page().triggerAction(QWebEnginePage.WebAction.Forward)
#
#     def urlChanged(self, url):
#         self.addressLineEdit.setText(url.toString())