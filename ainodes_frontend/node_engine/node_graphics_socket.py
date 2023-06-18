# -*- coding: utf-8 -*-
"""
A module containing Graphics representation of a :class:`~node_engine.node_socket.Socket`
"""
from qtpy import QtGui, QtCore
from qtpy.QtCore import Qt, QRectF
from qtpy.QtGui import QColor, QBrush, QPen
from qtpy.QtWidgets import QGraphicsItem

from ainodes_frontend import singleton as gs

SOCKET_COLORS = [
    QColor("#FFFF7700"),
    QColor("#FF52e220"),
    QColor("#FF0056a6"),
    QColor("#FFa86db1"),
    QColor("#FFb54747"),
    QColor("#FFdbe220"),
    QColor("#FF888888"),
    QColor("#FFFF7700"),
    QColor("#FF52e220"),
    QColor("#66FF99"),
    QColor("#0099CC"), #VAE
    QColor("#333300"),
    QColor("#993366"),
    QColor("#FF888888"),
]


model_colors = {0:SOCKET_COLORS[9],
                1:SOCKET_COLORS[10],
                2:SOCKET_COLORS[11],
                3:SOCKET_COLORS[12],
                4:SOCKET_COLORS[13],
}

class QDMGraphicsSocket(QGraphicsItem):
    """Class representing Graphic `Socket` in ``QGraphicsScene``"""
    def __init__(self, socket:'Socket'):
        """
        :param socket: reference to :class:`~node_engine.node_socket.Socket`
        :type socket: :class:`~nodeeditor.node_socket.Socket`
        """
        super().__init__(socket.node.grNode)

        self.socket = socket
        self.isHighlighted = False
        self.setZValue(-1)
        self.radius = 8
        self.outline_width = 2
        self.initAssets()

    @property
    def socket_type(self):
        return self.socket.socket_type

    def getSocketColor(self, key):
        """Returns the ``QColor`` for this ``key``"""

        print(self.socket.name)
        print(model_colors[0])
        if self.socket.name == "UNET":
            return model_colors[0]
        elif self.socket.name == "CLIP":
            return model_colors[1]
        elif self.socket.name == "VAE":
            return model_colors[2]
        elif self.socket.name == "CONTROLNET":
            return model_colors[3]


        if type(key) == int: return SOCKET_COLORS[key]
        elif type(key) == str: return QColor(key)
        return Qt.transparent

    def changeSocketType(self):
        """Change the Socket Type"""


        self._color_background = self.getSocketColor(self.socket_type)
        self._brush = QBrush(self._color_background)
        # print("Socket changed to:", self._color_background.getRgbF())
        self.update()

    def initAssets(self):
        """Initialize ``QObjects`` like ``QColor``, ``QPen`` and ``QBrush``"""

        # determine socket color
        self._color_background = self.getSocketColor(self.socket_type)
        self._color_outline = QColor("#FF000000")
        self._color_highlight = QColor("#FF37A6FF")

        self._pen = QPen(self._color_outline)
        self._pen.setWidthF(self.outline_width)
        self._pen_highlight = QPen(self._color_highlight)
        self._pen_highlight.setWidthF(2.0)
        self._brush = QBrush(self._color_background)

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):

        #if gs.highlight_sockets:
        mode = self.socket.node.scene.getView().mode
        dragged_socket = self.socket.node.scene.getView().dragging.drag_start_socket
        if mode == 2:
            #self.radius = 12
            #self.isHighlighted = True
            if self.socket.node != dragged_socket.node:
                if dragged_socket.is_input:
                    if not self.socket.is_input:
                        if self.socket.socket_type == dragged_socket.socket_type:
                            self.radius = 12
                            self.isHighlighted = True
                else:
                    if not dragged_socket.is_input:
                        if self.socket.is_input:
                            if self.socket.socket_type == dragged_socket.socket_type:
                                self.radius = 12
                                self.isHighlighted = True
        else:
            self.radius = 8
            self.isHighlighted = False

        painter.setBrush(self._brush)
        painter.setPen(self._pen if not self.isHighlighted else self._pen_highlight)
        """Painting a circle"""
        #print(self.socket.node, self.socket.is_input)
        # Add text next to the ellipse
        #text = "Test Text"
        #font = QtGui.QFont("Monospace", 12)
        #painter.setFont(font)
        #text_width = painter.fontMetrics().width(text) * 1.13
        #text_height = painter.fontMetrics().height() - 29
        text_width = 71.2
        text_height = -12
        if self.socket.is_input:
            painter.drawEllipse(-self.radius - 15, -self.radius, 2 * self.radius, 2 * self.radius)
            # Set the background color to dark green
            bg_color = QtGui.QColor('darkgreen')
            painter.setBrush(bg_color)
            text_color = QtGui.QColor('white')
            painter.setPen(text_color)
            # Define the position and size of the background rectangle
            rect = QtCore.QRectF(QtCore.QPointF(int(self.radius), int(-text_height / 2) + 3),
                                 QtCore.QSizeF(text_width + 10, text_height - 7))
            # Fill the rectangle with the background color
            painter.fillRect(rect, bg_color)
            painter.drawRoundedRect(rect, 3, 3)

            painter.drawText(QtCore.QPoint(int(self.radius + 5), int(-text_height/2)), f"{self.socket.name}")
        else:
            painter.setBrush(self._brush)
            painter.setPen(self._pen if not self.isHighlighted else self._pen_highlight)
            painter.drawEllipse(-self.radius + 15, -self.radius, 2 * self.radius, 2 * self.radius)

            # Set the background color to dark green
            bg_color = QtGui.QColor('darkgreen')
            painter.setBrush(bg_color)
            text_color = QtGui.QColor('white')
            painter.setPen(text_color)
            # Define the position and size of the background rectangle
            rect = QtCore.QRectF(QtCore.QPointF(int(self.radius - text_width - 20), int(-text_height / 2) + 3),
                                 QtCore.QSizeF(text_width + 10, text_height - 7))
            # Fill the rectangle with the background color
            painter.fillRect(rect, bg_color)
            painter.drawRoundedRect(rect, 3, 3)
            painter.drawText(QtCore.QPoint(int(self.radius - text_width - 15), int(-text_height / 2)), f"{self.socket.name}")

    def boundingRect(self) -> QRectF:
        """Defining Qt' bounding rectangle"""
        if self.socket.is_input:
            return QRectF(
                - self.radius - 10 - self.outline_width,
                - self.radius - self.outline_width,
                2 * (self.radius + self.outline_width),
                2 * (self.radius + self.outline_width),
            )
        else:
            return QRectF(
                - self.radius + 10 - self.outline_width,
                - self.radius - self.outline_width,
                2 * (self.radius + self.outline_width),
                2 * (self.radius + self.outline_width),
            )
    def wheelEvent(self, event):
        event.ignore()
