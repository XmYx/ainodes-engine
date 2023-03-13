import os

from qtpy import QtWidgets
from qtpy.QtGui import QPixmap, QIcon, QDrag
from qtpy.QtCore import QSize, Qt, QByteArray, QDataStream, QMimeData, QIODevice, QPoint
from qtpy.QtWidgets import QListWidget, QAbstractItemView, QListWidgetItem

import ainodes_frontend.base.node_config
from ainodes_frontend.base.node_config import CALC_NODES, get_class_from_opcode, LISTBOX_MIMETYPE, node_categories
from ainodes_frontend.node_engine.utils import dumpException, loadStylesheets

class QDMDragListbox(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # init
        self.setIconSize(QSize(32, 32))
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setDragEnabled(True)

        self.addMyItems()
        """self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss")
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )"""

    def addMyItems(self):
        self.clear()
        categories = {category: [] for category in node_categories}

        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys:
            node = get_class_from_opcode(key)
            if node.category not in node_categories:
                node_categories.append(node.category)
                categories[node.category] = []

            categories[node.category].append((node.op_title, node.icon, node.op_code))
        new_list = []
        for category, items in categories.items():

            parent = QtWidgets.QTreeWidgetItem(self)
            parent.setText(0, category.capitalize())
            items.sort(key=lambda item: item[0])
            for name, icon, op_code in items:
                item = QtWidgets.QTreeWidgetItem(parent)
                item.setText(0, name)
                pixmap = QPixmap(icon if icon is not None else ".")
                item.setIcon(0, QIcon(pixmap))
                item.setSizeHint(0, QSize(32, 32))
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)

                # setup data
                item.setData(0, Qt.UserRole, pixmap)
                item.setData(0, Qt.UserRole + 1, op_code)
        self.sortItems(0, Qt.AscendingOrder)



    def startDrag(self, *args, **kwargs):
        try:
            item = self.currentItem()
            op_code = item.data(0, Qt.UserRole + 1)

            pixmap = QPixmap(item.data(0, Qt.UserRole))


            itemData = QByteArray()
            dataStream = QDataStream(itemData, QIODevice.WriteOnly)
            dataStream << pixmap
            dataStream.writeInt8(op_code)
            dataStream.writeQString(item.text(0))

            mimeData = QMimeData()
            mimeData.setData(LISTBOX_MIMETYPE, itemData)

            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)

        except Exception as e: dumpException(e)
