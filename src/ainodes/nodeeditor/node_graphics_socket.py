# -*- coding: utf-8 -*-
"""
A module containing Graphics representation of a :class:`~nodeeditor.node_socket.Socket`
"""
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QGraphicsItem
from qtpy.QtGui import QColor, QBrush, QPen
from qtpy.QtCore import Qt, QRectF

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
    QColor("#FF0056a6"),
    QColor("#FFa86db1"),
    QColor("#FFb54747"),
    QColor("#FFdbe220"),
    QColor("#FF888888"),
]

class QDMGraphicsSocket(QGraphicsItem):
    """Class representing Graphic `Socket` in `QGraphicsScene`"""
    def __init__(self, socket: 'Socket'):
        """
        :param socket: reference to :class:`~nodeeditor.node_socket.Socket`
        :type socket: :class:`~nodeeditor.node_socket.Socket`
        """
        super().__init__(socket.node.grNode)

        self.socket = socket

        self.isHighlighted = False
        self.radius = 6
        self.outline_width = 1
        self.hovered = False  # Track hover state
        self.setAcceptHoverEvents(True)  # Enable hover events
        self.initAssets()

    @property
    def socket_type(self):
        return self.socket.socket_type

    def getSocketColor(self, key):
        """Returns the `QColor` for this `key`"""
        if type(key) == int:
            return SOCKET_COLORS[key]
        elif type(key) == str:
            return QColor(key)
        return Qt.transparent

    def changeSocketType(self):
        """Change the Socket Type"""
        self._color_background = self.getSocketColor(self.socket_type)
        self._brush = QBrush(self._color_background)
        self.update()

    def initAssets(self):
        """Initialize `QObjects` like `QColor`, `QPen` and `QBrush`"""

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
        mode = self.socket.node.scene.getView().mode
        dragged_socket = self.socket.node.scene.getView().dragging.drag_start_socket
        if mode == 2:
            if self.socket.node != dragged_socket.node:
                if dragged_socket.is_input:
                    if not self.socket.is_input:
                        if self.socket.name == dragged_socket.name:
                            self.radius = 12
                            self.isHighlighted = True
                else:
                    if not dragged_socket.is_input:
                        if self.socket.is_input:
                            if self.socket.name == dragged_socket.name:
                                self.radius = 12
                                self.isHighlighted = True
        else:
            self.radius = 8
            self.isHighlighted = False

        if self.socket_type == 1:
            # Draw a rounded green execution arrow
            painter.setBrush(QColor("green"))
            painter.setPen(self._pen if not self.isHighlighted else self._pen_highlight)

            path = QtGui.QPainterPath()
            arrow_size = self.radius * 2
            offset = 10
            if self.socket.is_input:
                # Draw arrow pointing right, offset to the left
                path.moveTo(-arrow_size + offset, 0)  # Offset the start of the arrow to the left
                path.lineTo(-2 * arrow_size + offset, -arrow_size)  # Top point of the arrowhead
                path.lineTo(-2 * arrow_size + offset, arrow_size)  # Bottom point of the arrowhead
                path.lineTo(-arrow_size + offset, 0)  # Back to the starting point to close the arrow
            else:
                # Draw arrow pointing right
                path.moveTo(arrow_size + offset, 0)
                path.lineTo(0 + offset, -arrow_size)
                path.lineTo(0 + offset, arrow_size)
                path.lineTo(arrow_size + offset, 0)

            path = path.simplified()
            painter.drawPath(path)

            # Optionally draw a highlight border when hovered
            if self.hovered:
                highlight_pen = QPen(QColor("yellow"))
                highlight_pen.setWidthF(3.0)
                painter.setPen(highlight_pen)
                painter.drawPath(path)

        else:
            # Default behavior: Draw circle with text
            painter.setBrush(self._brush)
            painter.setPen(self._pen if not self.isHighlighted else self._pen_highlight)
            # Draw hover highlight if applicable
            if self.hovered:
                highlight_pen = QPen(QColor("yellow"))
                highlight_pen.setWidthF(3.0)
                painter.setPen(highlight_pen)
                if self.socket.is_input:
                    painter.drawEllipse(-self.radius - 15, -self.radius, 2 * self.radius, 2 * self.radius)
                else:
                    painter.drawEllipse(-self.radius + 15, -self.radius, 2 * self.radius, 2 * self.radius)

            text_width = 75
            proposed_width = painter.fontMetrics().width(f"{self.socket.name}")
            if proposed_width > text_width:
                text_width = proposed_width
            text_height = -12

            if self.socket.is_input:
                painter.drawEllipse(-self.radius - 15, -self.radius, 2 * self.radius, 2 * self.radius)
                bg_color = QtGui.QColor('darkgreen')
                painter.setBrush(bg_color)
                text_color = QtGui.QColor('white')
                painter.setPen(text_color)
                rect = QtCore.QRectF(QtCore.QPointF(int(self.radius), int(-text_height / 2) + 3),
                                     QtCore.QSizeF(text_width + 10, text_height - 7))
                painter.fillRect(rect, bg_color)
                painter.drawRoundedRect(rect, 3, 3)

                painter.drawText(QtCore.QPoint(int(self.radius + 5), int(-text_height/2)), f"{self.socket.name}")
            else:
                painter.setBrush(self._brush)
                painter.setPen(self._pen if not self.isHighlighted else self._pen_highlight)
                painter.drawEllipse(-self.radius + 15, -self.radius, 2 * self.radius, 2 * self.radius)

                bg_color = QtGui.QColor('darkgreen')
                painter.setBrush(bg_color)
                text_color = QtGui.QColor('white')
                painter.setPen(text_color)
                rect = QtCore.QRectF(QtCore.QPointF(int(self.radius - text_width - 20), int(-text_height / 2) + 3),
                                     QtCore.QSizeF(text_width + 10, text_height - 7))
                painter.fillRect(rect, bg_color)
                painter.drawRoundedRect(rect, 3, 3)
                painter.drawText(QtCore.QPoint(int(self.radius - text_width - 15), int(-text_height / 2)), f"{self.socket.name}")

    def boundingRect(self) -> QRectF:
        """Defining Qt's bounding rectangle"""
        if self.socket_type == 1:
            # Adjust the bounding rectangle to accommodate the arrow shape
            if self.socket.is_input:
                return QRectF(-2 * self.radius - 10, -self.radius - self.outline_width,
                              3 * (self.radius + self.outline_width), 2 * (self.radius + self.outline_width))
            else:
                return QRectF(-self.radius - self.outline_width + 10, -self.radius - self.outline_width,
                              2 * (self.radius + self.outline_width) + 10, 2 * (self.radius + self.outline_width))
        else:
            # Default bounding rectangle for circular sockets
            if self.socket.is_input:
                return QRectF(-self.radius - 15 - self.outline_width,
                              -self.radius - self.outline_width,
                              2 * (self.radius + self.outline_width),
                              2 * (self.radius + self.outline_width))
            else:
                return QRectF(-self.radius + 15 - self.outline_width,
                              -self.radius - self.outline_width,
                              2 * (self.radius + self.outline_width),
                              2 * (self.radius + self.outline_width))

    def hoverEnterEvent(self, event):
        """Handle hover enter event"""
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        """Handle hover leave event"""
        self.hovered = False
        self.update()
