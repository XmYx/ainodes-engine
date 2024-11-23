# -*- coding: utf-8 -*-
"""
A module containing Graphics representation of :class:`~nodeeditor.node_node.Node` with resizing functionality
"""
from qtpy.QtWidgets import QGraphicsItem, QWidget, QGraphicsTextItem, QStyleOptionGraphicsItem
from qtpy.QtGui import QFont, QColor, QPen, QBrush, QPainterPath, QCursor
from qtpy.QtCore import Qt, QRectF, QPointF
from enum import Enum


class ResizeDirection(Enum):
    NO_RESIZE = 0
    TOP_LEFT = 1
    TOP = 2
    TOP_RIGHT = 3
    RIGHT = 4
    BOTTOM_RIGHT = 5
    BOTTOM = 6
    BOTTOM_LEFT = 7
    LEFT = 8


class QDMGraphicsNode(QGraphicsItem):
    """Class describing Graphics representation of :class:`~nodeeditor.node_node.Node` with resizing functionality"""
    def __init__(self, node: 'Node', parent: QWidget = None):
        super().__init__(parent)
        self.node = node

        # init our flags
        self.hovered = False
        self._was_moved = False
        self._last_selected_state = False

        self.initSizes()
        self.initAssets()
        self.initUI()

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        # Resizing
        self.resizing = False
        self.resize_direction = ResizeDirection.NO_RESIZE
        self.resize_start_pos = QPointF()
        self.resize_handle_size = 10.0  # Size of the resize handle area
        self.minimum_width = 80
        self.minimum_height = 80

    @property
    def content(self):
        """Reference to `Node Content`"""
        return self.node.content if self.node else None

    @property
    def title(self):
        """title of this `Node`

        :getter: current Graphics Node title
        :setter: stores and make visible the new title
        :type: str
        """
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self.title_item.setPlainText(self._title)

    def initUI(self):
        """Set up this ``QGraphicsItem``"""
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setAcceptHoverEvents(True)

        # init title
        self.initTitle()
        self.title = self.node.title

        self.initContent()

    def initSizes(self):
        """Set up internal attributes like `width`, `height`, etc."""
        self.width = 180
        self.height = 240
        self.edge_roundness = 10.0
        self.edge_padding = 25.0
        self.title_height = 24
        self.title_horizontal_padding = 4.0
        self.title_vertical_padding = 4.0

    def initAssets(self):
        """Initialize ``QObjects`` like ``QColor``, ``QPen`` and ``QBrush``"""
        self._title_color = Qt.white
        self._title_font = QFont("Ubuntu", 10)

        self._color = QColor("#7F000000")
        self._color_selected = QColor("#FFFFA637")
        self._color_hovered = QColor("#FF37A6FF")

        self._pen_default = QPen(self._color)
        self._pen_default.setWidthF(2.0)
        self._pen_selected = QPen(self._color_selected)
        self._pen_selected.setWidthF(2.0)
        self._pen_hovered = QPen(self._color_hovered)
        self._pen_hovered.setWidthF(3.0)

        self._brush_title = QBrush(QColor("#FF313131"))
        self._brush_background = QBrush(QColor("#E3212121"))

    def onSelected(self):
        """Our event handling when the node was selected"""
        self.node.scene.grScene.itemSelected.emit()

    def doSelect(self, new_state=True):
        """Safe version of selecting the `Graphics Node`. Takes care about the selection state flag used internally

        :param new_state: ``True`` to select, ``False`` to deselect
        :type new_state: ``bool``
        """
        self.setSelected(new_state)
        self._last_selected_state = new_state
        if new_state:
            self.onSelected()

    def mousePressEvent(self, event):
        """Handle mouse press events for resizing"""
        if event.button() == Qt.LeftButton:
            self.resize_direction = self.checkResizeArea(event.pos())
            if self.resize_direction != ResizeDirection.NO_RESIZE:
                self.resizing = True
                self.resize_start_pos = event.pos()
                self._was_moved = False
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move events for moving and resizing"""
        if self.resizing:
            self.performResize(event.pos())
            event.accept()
            return

        super().mouseMoveEvent(event)

        # Optimize me! Just update the selected nodes
        for node in self.scene().scene.nodes:
            if node.grNode.isSelected():
                node.updateConnectedEdges()
        self._was_moved = True

    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if self.resizing:
            self.resizing = False
            self.resize_direction = ResizeDirection.NO_RESIZE
            self.node.scene.history.storeHistory("Node resized", setModified=True)
            event.accept()
            return

        super().mouseReleaseEvent(event)

        # handle when grNode moved
        if self._was_moved:
            self._was_moved = False
            self.node.scene.history.storeHistory("Node moved", setModified=True)

            self.node.scene.resetLastSelectedStates()
            self.doSelect()  # also trigger itemSelected when node was moved

            # we need to store the last selected state, because moving does also select the nodes
            self.node.scene._last_selected_items = self.node.scene.getSelectedItems()

            # now we want to skip storing selection
            return

        # handle when grNode was clicked on
        if self._last_selected_state != self.isSelected() or self.node.scene._last_selected_items != self.node.scene.getSelectedItems():
            self.node.scene.resetLastSelectedStates()
            self._last_selected_state = self.isSelected()
            self.onSelected()

    def mouseDoubleClickEvent(self, event):
        """Overriden event for doubleclick. Resend to `Node::onDoubleClicked`"""
        self.node.onDoubleClicked(event)

    def hoverEnterEvent(self, event):
        """Handle hover enter events"""
        self.hovered = True
        self.update()

    def hoverLeaveEvent(self, event):
        """Handle hover leave events"""
        self.hovered = False
        self.update()

    def hoverMoveEvent(self, event):
        """Handle hover move events for changing the cursor"""
        if self.resizing:
            return

        direction = self.checkResizeArea(event.pos())
        cursor = self.getCursorForResizeDirection(direction)
        if cursor:
            self.setCursor(cursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        super().hoverMoveEvent(event)

    def checkResizeArea(self, pos):
        """Check which resize area the mouse is in"""
        x = pos.x()
        y = pos.y()

        left = x <= self.resize_handle_size
        right = x >= self.width - self.resize_handle_size
        top = y <= self.resize_handle_size
        bottom = y >= self.height - self.resize_handle_size

        if left and top:
            return ResizeDirection.TOP_LEFT
        elif right and top:
            return ResizeDirection.TOP_RIGHT
        elif left and bottom:
            return ResizeDirection.BOTTOM_LEFT
        elif right and bottom:
            return ResizeDirection.BOTTOM_RIGHT
        elif top:
            return ResizeDirection.TOP
        elif bottom:
            return ResizeDirection.BOTTOM
        elif left:
            return ResizeDirection.LEFT
        elif right:
            return ResizeDirection.RIGHT
        else:
            return ResizeDirection.NO_RESIZE

    def getCursorForResizeDirection(self, direction):
        """Return the appropriate cursor for the resize direction"""
        if direction == ResizeDirection.TOP_LEFT or direction == ResizeDirection.BOTTOM_RIGHT:
            return Qt.SizeFDiagCursor
        elif direction == ResizeDirection.TOP_RIGHT or direction == ResizeDirection.BOTTOM_LEFT:
            return Qt.SizeBDiagCursor
        elif direction == ResizeDirection.TOP or direction == ResizeDirection.BOTTOM:
            return Qt.SizeVerCursor
        elif direction == ResizeDirection.LEFT or direction == ResizeDirection.RIGHT:
            return Qt.SizeHorCursor
        else:
            return None

    def performResize(self, pos):
        """Perform the resizing of the node based on mouse movement"""
        delta = pos - self.resize_start_pos
        dx = delta.x()
        dy = delta.y()
        old_rect = QRectF(0, 0, self.width, self.height)

        if self.resize_direction == ResizeDirection.RIGHT or self.resize_direction == ResizeDirection.TOP_RIGHT or self.resize_direction == ResizeDirection.BOTTOM_RIGHT:
            new_width = max(self.minimum_width, self.width + dx)
            self.prepareGeometryChange()
            self.width = new_width
            self.update()
            self.resize_start_pos.setX(pos.x())

        if self.resize_direction == ResizeDirection.BOTTOM or self.resize_direction == ResizeDirection.BOTTOM_LEFT or self.resize_direction == ResizeDirection.BOTTOM_RIGHT:
            new_height = max(self.minimum_height, self.height + dy)
            self.prepareGeometryChange()
            self.height = new_height
            self.update()
            self.resize_start_pos.setY(pos.y())

        if self.resize_direction == ResizeDirection.LEFT or self.resize_direction == ResizeDirection.TOP_LEFT or self.resize_direction == ResizeDirection.BOTTOM_LEFT:
            new_width = max(self.minimum_width, self.width - dx)
            diff = self.width - new_width
            self.prepareGeometryChange()
            self.width = new_width
            self.setPos(self.pos().x() + diff, self.pos().y())
            self.update()
            self.resize_start_pos.setX(pos.x())

        if self.resize_direction == ResizeDirection.TOP or self.resize_direction == ResizeDirection.TOP_LEFT or self.resize_direction == ResizeDirection.TOP_RIGHT:
            new_height = max(self.minimum_height, self.height - dy)
            diff = self.height - new_height
            self.prepareGeometryChange()
            self.height = new_height
            self.setPos(self.pos().x(), self.pos().y() + diff)
            self.update()
            self.resize_start_pos.setY(pos.y())

        self.updateContent()
        self.node.update_all_sockets()

    def updateContent(self):
        """Update the node's content after resizing"""
        if self.content is not None:
            self.content.setGeometry(
                int(self.edge_padding),
                int(self.title_height + self.edge_padding),
                int(self.width - 4 * self.edge_padding),
                int(self.height - 2 * self.edge_padding - self.title_height - 40)
            )
        self.title_item.setTextWidth(
            self.width - 2 * self.title_horizontal_padding
        )

    def boundingRect(self) -> QRectF:
        """Defining Qt's bounding rectangle"""
        return QRectF(
            0,
            0,
            self.width,
            self.height
        ).normalized()

    def initTitle(self):
        """Set up the title Graphics representation: font, color, position, etc."""
        self.title_item = QGraphicsTextItem(self)
        self.title_item.node = self.node
        self.title_item.setDefaultTextColor(self._title_color)
        self.title_item.setFont(self._title_font)
        self.title_item.setPos(self.title_horizontal_padding, 0)
        self.title_item.setTextWidth(
            self.width - 2 * self.title_horizontal_padding
        )

    def initContent(self):
        """Set up the `grContent` - ``QGraphicsProxyWidget`` to have a container for `Graphics Content`"""
        if self.content is not None:
            self.content.setGeometry(
                self.edge_padding,
                self.title_height + self.edge_padding,
                self.width - 2 * self.edge_padding,
                self.height - 2 * self.edge_padding - self.title_height
            )

        # get the QGraphicsProxyWidget when inserted into the grScene
        self.grContent = self.node.scene.grScene.addWidget(self.content)
        self.grContent.node = self.node
        self.grContent.setParentItem(self)

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget=None):
        """Painting the rounded rectangular `Node`"""
        # Title
        path_title = QPainterPath()
        path_title.setFillRule(Qt.WindingFill)
        path_title.addRoundedRect(0, 0, self.width, self.title_height, self.edge_roundness, self.edge_roundness)
        path_title.addRect(0, self.title_height - self.edge_roundness, self.edge_roundness, self.edge_roundness)
        path_title.addRect(self.width - self.edge_roundness, self.title_height - self.edge_roundness, self.edge_roundness, self.edge_roundness)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._brush_title)
        painter.drawPath(path_title.simplified())

        # Content
        path_content = QPainterPath()
        path_content.setFillRule(Qt.WindingFill)
        path_content.addRoundedRect(0, self.title_height, self.width, self.height - self.title_height, self.edge_roundness, self.edge_roundness)
        path_content.addRect(0, self.title_height, self.edge_roundness, self.edge_roundness)
        path_content.addRect(self.width - self.edge_roundness, self.title_height, self.edge_roundness, self.edge_roundness)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._brush_background)
        painter.drawPath(path_content.simplified())

        # Outline
        path_outline = QPainterPath()
        path_outline.addRoundedRect(-1, -1, self.width + 2, self.height + 2, self.edge_roundness, self.edge_roundness)
        painter.setBrush(Qt.NoBrush)
        if self.hovered:
            painter.setPen(self._pen_hovered)
            painter.drawPath(path_outline.simplified())
            painter.setPen(self._pen_default)
            painter.drawPath(path_outline.simplified())
        else:
            painter.setPen(self._pen_default if not self.isSelected() else self._pen_selected)
            painter.drawPath(path_outline.simplified())

        # Resize Handles (optional visual representation)
        if self.hovered or self.resizing:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 50)))
            size = int(self.resize_handle_size)
            # Top-left corner
            painter.drawRect(0, 0, size, size)
            # Top-right corner
            painter.drawRect(int(self.width - size), 0, size, size)
            # Bottom-left corner
            painter.drawRect(0, int(self.height - size), size, size)
            # Bottom-right corner
            painter.drawRect(int(self.width - size), int(self.height - size), size, size)
            # Left edge
            painter.drawRect(0, size, size, int(self.height - 2 * size))
            # Right edge
            painter.drawRect(int(self.width - size), size, size, int(self.height - 2 * size))
            # Top edge
            painter.drawRect(size, 0, int(self.width - 2 * size), size)
            # Bottom edge
            painter.drawRect(size, int(self.height - size), int(self.width - 2 * size), size)
