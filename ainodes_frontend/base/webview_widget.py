from qtpy.QtCore import Qt, QUrl
from qtpy.QtWebEngineWidgets import QWebEngineView
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget, QLineEdit, QPushButton, QHBoxLayout

class BrowserWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create layout for the browser widget
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create a horizontal layout for the toolbar
        toolbar_layout = QHBoxLayout()

        # Create a back button
        self.back_button = QPushButton("Back")
        toolbar_layout.addWidget(self.back_button)

        # Create a URL bar
        self.url_bar = QLineEdit()
        self.url_bar.returnPressed.connect(self.load_url)
        toolbar_layout.addWidget(self.url_bar)

        # Create a Go button
        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.load_url)
        toolbar_layout.addWidget(self.go_button)

        # Add the toolbar layout to the main layout
        layout.addLayout(toolbar_layout)

        # Create a QWebEngineView
        self.webview = QWebEngineView()

        self.back_button.clicked.connect(self.webview.back)


        layout.addWidget(self.webview)

        # Load www.google.com
        self.load_url()

    def load_url(self):
        url = self.url_bar.text()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        self.webview.load(QUrl(url))