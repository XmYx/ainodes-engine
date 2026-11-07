"""Paintable canvas node (issue #12).

Draw scribbles, masks or pose-ish guides straight on the node and feed the
result to a ControlNet / img2img chain, instead of leaving the app to make a
PNG somewhere else.

The undo history is local to the node and bound to Ctrl+Z / Ctrl+Shift+Z
*while the canvas has focus*, so it no longer removes the node itself - that
was the complaint on the original issue - and the buttons remain for people
who prefer them.
"""
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import AiNode, CalcGraphicsNode, get_next_opcode, register_node
from ainodes_frontend.base.qimage_ops import pixmap_to_tensor, tensor_image_to_pixmap
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

OP_NODE_CANVAS = get_next_opcode()

MAX_UNDO = 32


class PaintCanvas(QtWidgets.QWidget):
    """A bitmap you can draw on with the mouse."""

    changed = QtCore.Signal()

    def __init__(self, width=512, height=512, parent=None):
        super().__init__(parent)
        self.setMinimumSize(256, 256)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)

        self.brush_color = QtGui.QColor("#ffffff")
        self.brush_size = 12
        self.erasing = False

        self._image = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
        self._image.fill(QtGui.QColor(0, 0, 0, 255))
        self._undo = []
        self._redo = []
        self._last_point = None

    # ------------------------------------------------------------- history
    def push_undo(self):
        self._undo.append(self._image.copy())
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self):
        if not self._undo:
            return False
        self._redo.append(self._image.copy())
        self._image = self._undo.pop()
        self.update()
        self.changed.emit()
        return True

    def redo(self):
        if not self._redo:
            return False
        self._undo.append(self._image.copy())
        self._image = self._redo.pop()
        self.update()
        self.changed.emit()
        return True

    # --------------------------------------------------------------- image
    def clear(self, color=QtGui.QColor(0, 0, 0, 255)):
        self.push_undo()
        self._image.fill(color)
        self.update()
        self.changed.emit()

    def set_background(self, pixmap):
        """Replace the canvas with an incoming image, keeping its size."""
        if pixmap is None or pixmap.isNull():
            return
        self.push_undo()
        self._image = pixmap.toImage().convertToFormat(
            QtGui.QImage.Format.Format_RGBA8888)
        self.update()
        self.changed.emit()

    def resize_canvas(self, width, height):
        if (width, height) == (self._image.width(), self._image.height()):
            return
        self.push_undo()
        scaled = self._image.scaled(width, height, Qt.IgnoreAspectRatio,
                                    Qt.SmoothTransformation)
        self._image = scaled.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        self.update()
        self.changed.emit()

    def pixmap(self):
        return QtGui.QPixmap.fromImage(self._image)

    # -------------------------------------------------------------- events
    def _to_image(self, position):
        if self.width() == 0 or self.height() == 0:
            return QtCore.QPoint(0, 0)
        scale_x = self._image.width() / self.width()
        scale_y = self._image.height() / self.height()
        return QtCore.QPoint(int(position.x() * scale_x), int(position.y() * scale_y))

    def mousePressEvent(self, event):
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return super().mousePressEvent(event)
        self.push_undo()
        self.erasing = event.button() == Qt.RightButton
        self._last_point = self._to_image(event.position())
        self._draw_to(self._last_point)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._last_point is None:
            return super().mouseMoveEvent(event)
        point = self._to_image(event.position())
        self._draw_to(point)
        self._last_point = point
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._last_point is not None:
            self._last_point = None
            self.changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _draw_to(self, point):
        painter = QtGui.QPainter(self._image)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        color = QtGui.QColor(0, 0, 0, 255) if self.erasing else self.brush_color
        pen = QtGui.QPen(color, self.brush_size, Qt.SolidLine, Qt.RoundCap,
                         Qt.RoundJoin)
        painter.setPen(pen)
        if self._last_point is None or self._last_point == point:
            painter.drawPoint(point)
        else:
            painter.drawLine(self._last_point, point)
        painter.end()
        self.update()

    def keyPressEvent(self, event):
        """Node-local undo/redo.

        The canvas has keyboard focus while you draw, so swallowing these here
        stops the graph-wide undo from deleting the node (issue #12).
        """
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Z and not (event.modifiers() & Qt.ShiftModifier):
                self.undo()
                event.accept()
                return
            if event.key() == Qt.Key_Y or (event.key() == Qt.Key_Z and
                                           event.modifiers() & Qt.ShiftModifier):
                self.redo()
                event.accept()
                return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(self.rect(), self._image, self._image.rect())
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 40), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()


class CanvasWidget(QDMNodeContentWidget):
    eval_signal = QtCore.Signal()
    set_image_signal = QtCore.Signal(object)

    def initUI(self):
        self.canvas = PaintCanvas()

        self.width_box = self.create_spin_box("Width", 64, 4096, 512, 8,
                                              accessible_name="canvas_width")
        self.height_box = self.create_spin_box("Height", 64, 4096, 512, 8,
                                               accessible_name="canvas_height")
        self.brush_box = self.create_spin_box("Brush", 1, 256, 12, 1,
                                              accessible_name="canvas_brush")
        self.color_button = QtWidgets.QPushButton("Colour")
        self.undo_button = QtWidgets.QPushButton("Undo")
        self.redo_button = QtWidgets.QPushButton("Redo")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.create_button_layout([self.color_button, self.undo_button,
                                   self.redo_button, self.clear_button])

        canvas_row = QtWidgets.QHBoxLayout()
        canvas_row.addWidget(self.canvas)
        self.widget_list.append(canvas_row)

        self.create_main_layout(grid=1)

        self.color_button.clicked.connect(self.pick_color)
        self.undo_button.clicked.connect(self.canvas.undo)
        self.redo_button.clicked.connect(self.canvas.redo)
        self.clear_button.clicked.connect(lambda: self.canvas.clear())
        self.brush_box.valueChanged.connect(self.set_brush)
        self.width_box.valueChanged.connect(self.apply_size)
        self.height_box.valueChanged.connect(self.apply_size)
        self.canvas.changed.connect(self.on_canvas_changed)
        self.set_image_signal.connect(self.canvas.set_background)
        self.update_color_button()

    def pick_color(self):
        color = QtWidgets.QColorDialog.getColor(self.canvas.brush_color, self,
                                                "Brush colour")
        if color.isValid():
            self.canvas.brush_color = color
            self.update_color_button()

    def update_color_button(self):
        color = self.canvas.brush_color
        text_color = "#000000" if color.lightness() > 140 else "#ffffff"
        self.color_button.setStyleSheet(
            f"background-color: {color.name()}; color: {text_color};")

    def set_brush(self, value):
        self.canvas.brush_size = int(value)

    def apply_size(self):
        self.canvas.resize_canvas(self.width_box.value(), self.height_box.value())

    def on_canvas_changed(self):
        if self.node is not None:
            self.node.markDirty(True)

    # The bitmap is part of the node's state, so it has to survive a save.
    def serialize(self, exec=False):
        data = super().serialize(exec=exec)
        if not exec:
            data["canvas_png"] = _image_to_base64(self.canvas.pixmap())
        return data

    def deserialize(self, data, hashmap={}, restore_id: bool = True):
        result = super().deserialize(data, hashmap, restore_id)
        encoded = data.get("canvas_png")
        if encoded:
            pixmap = _base64_to_pixmap(encoded)
            if pixmap is not None:
                self.canvas.set_background(pixmap)
        return result


def _image_to_base64(pixmap):
    import base64

    buffer = QtCore.QBuffer()
    buffer.open(QtCore.QBuffer.OpenModeFlag.ReadWrite)
    pixmap.save(buffer, "PNG")
    return base64.b64encode(bytes(buffer.data())).decode("ascii")


def _base64_to_pixmap(encoded):
    import base64

    try:
        raw = base64.b64decode(encoded)
    except Exception:
        return None
    pixmap = QtGui.QPixmap()
    return pixmap if pixmap.loadFromData(raw, "PNG") else None


@register_node(OP_NODE_CANVAS)
class CanvasNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/input_image.png"
    op_code = OP_NODE_CANVAS
    op_title = "Canvas"
    content_label_objname = "canvas_node"
    category = "base/image"
    NodeContent_class = CanvasWidget
    GraphicsNode_class = CalcGraphicsNode
    dim = (420, 640)
    custom_input_socket_name = ["IMAGE", "EXEC"]
    custom_output_socket_name = ["IMAGE", "EXEC"]

    help_text = (
        "Draw directly on the node and send the result downstream as an IMAGE.\n\n"
        "Left mouse paints, right mouse erases to black.\n"
        "Ctrl+Z / Ctrl+Shift+Z undo and redo the strokes (only while the\n"
        "canvas has focus, so the graph's own undo is left alone).\n"
        "Connect an IMAGE to the input to draw on top of it."
    )

    def __init__(self, scene):
        super().__init__(scene, inputs=[5, 1], outputs=[5, 1])
        self.last_input = None

    def evalImplementation_thread(self, index=0):
        incoming = self.getInputData(0)
        if incoming is not None and incoming is not self.last_input:
            # a new upstream image becomes the canvas background
            self.last_input = incoming
            self.content.set_image_signal.emit(tensor_image_to_pixmap(incoming))

        tensor = pixmap_to_tensor(self.content.canvas.pixmap())
        return [tensor]

    def onWorkerFinished(self, result, exec=True):
        self.busy = False
        self.markDirty(False)
        if result:
            self.setOutput(0, result[0])
        if exec:
            self.executeChild(1)
