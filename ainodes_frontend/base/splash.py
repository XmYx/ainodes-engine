"""Startup splash screen.

Split out of main.py, and fixed for Qt 6: ``QFontMetrics.width()`` was removed
(it is ``horizontalAdvance()`` now) and ``Qt.NoPen`` / ``Qt.white`` have to be
spelled with their scoped enum names.
"""
import os

from qtpy import QtCore, QtGui
from qtpy.QtCore import QPropertyAnimation, QEasingCurve, QRect, Qt
from qtpy.QtGui import QColor, QPainter, QPixmap
from qtpy.QtWidgets import QProgressBar, QSplashScreen, QTextEdit

from ainodes_frontend.base import theme


class SplashScreen(QSplashScreen):
    """Splash with a progress bar and a small log view."""

    def __init__(self, image_path=None, message="Loading aiNodes"):
        pixmap = QPixmap()
        if image_path and os.path.isfile(image_path):
            pixmap = QPixmap(image_path)
        if pixmap.isNull():
            pixmap = QPixmap(480, 260)
            pixmap.fill(theme.color("surface"))
        super().__init__(pixmap)

        self.message = message
        self._pixmap = pixmap

        accent = theme.color("accent").name()
        surface = theme.color("surface_alt").name()
        border = theme.color("border").name()
        text = theme.color("text").name()

        self.progressBar = QProgressBar(self)
        self.progressBar.setStyleSheet(
            f"QProgressBar {{ background-color: {surface};"
            f" border: 1px solid {border}; border-radius: 4px;"
            f" text-align: center; color: {text}; }}"
            f"QProgressBar::chunk {{ background-color: {accent};"
            f" border-radius: 3px; }}"
        )
        self.progressBar.setGeometry(10, pixmap.height() - 28,
                                     pixmap.width() - 20, 20)
        self.progressBar.setValue(0)

        self.text_display = QTextEdit(self)
        self.text_display.setGeometry(10, pixmap.height() - 104,
                                      pixmap.width() - 20, 72)
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet(
            f"background-color: {theme.color('console_bg').name()};"
            f" color: {theme.color('console_text').name()};"
            f" border: 1px solid {border}; font-family: monospace;"
        )

    def append_text(self, text):
        self.text_display.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        self.text_display.insertPlainText(text)
        self.text_display.ensureCursorVisible()

    def set_progress(self, value, message=None):
        self.progressBar.setValue(int(value))
        if message:
            self.append_text(message + "\n")
        self.repaint()

    # Qt calls this; keep the painting cheap.
    def drawContents(self, painter: QPainter):
        super().drawContents(painter)

        font = painter.font()
        font.setPointSize(14)
        painter.setFont(font)

        metrics = painter.fontMetrics()
        text_width = metrics.horizontalAdvance(self.message)
        text_height = metrics.height()

        text_x = int((self.width() - text_width) / 2)
        text_y = max(int(self.height() * 0.45), text_height)

        painter.setBrush(QColor(0, 0, 0, 96))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(text_x - 6, text_y - 6, text_width + 12, text_height + 12)

        painter.setPen(theme.color("text"))
        painter.drawText(text_x, text_y + metrics.ascent(), self.message)

        self.progressBar.render(painter, QRect(self.progressBar.geometry()).topLeft())

    def fade_out(self, window, duration=400):
        animation = QPropertyAnimation(self, b"windowOpacity")
        animation.setDuration(duration)
        animation.setStartValue(1.0)
        animation.setEndValue(0.0)
        animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        animation.finished.connect(lambda: self.finish(window))
        animation.start()
        # keep a reference so the animation is not garbage collected mid-flight
        self._fade_animation = animation
        return animation
