from PyQt5.QtGui import QPixmap, QIcon, QDrag
from PyQt5.QtCore import QSize, Qt, QByteArray, QDataStream, QMimeData, QIODevice, QPoint
from PyQt5.QtWidgets import QListWidget, QAbstractItemView, QListWidgetItem

from node_core.node_register import NODE_CLASSES, LISTBOX_MIMETYPE
from nodeeditor.utils import dumpException


class QDMDragListbox(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # init
        self.setIconSize(QSize(32, 32))
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)

        self.addMyItems()


    def addMyItems(self):
        self.clear()
        class_names = list(NODE_CLASSES.keys())
        class_names.sort()
        for class_name in class_names:
            node_class = NODE_CLASSES[class_name]
            self.addMyItem(node_class.op_title, node_class.icon, class_name)

    def addMyItem(self, name, icon=None, class_name=""):
        item = QListWidgetItem(name, self)  # Can be (icon, text, parent, <int>type)
        pixmap = QPixmap(icon if icon is not None else ".")
        item.setIcon(QIcon(pixmap))
        item.setSizeHint(QSize(32, 32))

        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)

        # Setup data
        item.setData(Qt.UserRole, pixmap)
        item.setData(Qt.UserRole + 1, class_name)

    def startDrag(self, *args, **kwargs):
        try:
            item = self.currentItem()
            class_name = item.data(Qt.UserRole + 1)

            pixmap = QPixmap(item.data(Qt.UserRole))

            itemData = QByteArray()
            dataStream = QDataStream(itemData, QIODevice.WriteOnly)
            dataStream << pixmap
            dataStream.writeQString(class_name)
            dataStream.writeQString(item.text())

            mimeData = QMimeData()
            mimeData.setData(LISTBOX_MIMETYPE, itemData)

            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)

        except Exception as e:
            dumpException(e)