import os

from qtpy import QtWidgets
from qtpy.QtCore import QSize, Qt, QByteArray, QDataStream, QMimeData, QIODevice, QPoint
from qtpy.QtGui import QPixmap, QIcon, QDrag
from qtpy.QtWidgets import QAbstractItemView

from ainodes_frontend.base.node_config import CALC_NODES, get_class_from_opcode, LISTBOX_MIMETYPE, node_categories
from ainodes_frontend.node_engine.utils import dumpException
from ainodes_frontend import singleton as gs


class QDMDragListbox(QtWidgets.QWidget):  # subclassing from QWidget now
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # Create a vertical layout
        self.layout = QtWidgets.QVBoxLayout(self)

        # Search box
        self.search_box = QtWidgets.QLineEdit(self)
        self.search_box.setPlaceholderText("Search...")
        self.search_box.textChanged.connect(self.filter_nodes)
        self.layout.addWidget(self.search_box)  # Adding at the top of the layout

        # Tree widget
        self.tree_widget = QtWidgets.QTreeWidget(self)
        self.tree_widget.setIconSize(QSize(32, 32))
        self.tree_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_widget.setDragEnabled(True)
        self.tree_widget.header().hide()
        self.layout.addWidget(self.tree_widget)

        self.setLayout(self.layout)

    def filter_nodes(self):
        query = self.search_box.text().lower()

        def collapse_all_items(item):
            for i in range(item.childCount()):
                child = item.child(i)
                collapse_all_items(child)
            item.setExpanded(False)

        def recursive_filter(item):
            matches = False
            for i in range(item.childCount()):
                child = item.child(i)
                if recursive_filter(child):  # Recursively check children
                    matches = True
            if item.text(0).lower().find(query) != -1:  # Check the item itself
                matches = True
            item.setHidden(not matches)

            # Expand the item if matches
            if matches:
                item.setExpanded(True)
                parent = item.parent()
                while parent:  # also expand all parent items
                    parent.setExpanded(True)
                    parent = parent.parent()
            return matches

        # Make all items visible first
        for i in range(self.tree_widget.topLevelItemCount()):
            topLevelItem = self.tree_widget.topLevelItem(i)
            topLevelItem.setHidden(False)
            for j in range(topLevelItem.childCount()):
                child = topLevelItem.child(j)
                child.setHidden(False)

        # Collapse all items
        for i in range(self.tree_widget.topLevelItemCount()):
            topLevelItem = self.tree_widget.topLevelItem(i)
            collapse_all_items(topLevelItem)

        # If query is not empty, apply the filtering and expand matching items
        if query:
            for i in range(self.tree_widget.topLevelItemCount()):
                topLevelItem = self.tree_widget.topLevelItem(i)
                recursive_filter(topLevelItem)

    def addMyItems(self):
        self.tree_widget.clear()
        categories = {}
        gs.node_dict = {}
        self.addSubgraphItems(categories)
        keys = list(CALC_NODES.keys())
        keys.sort()
        for key in keys:
            node = get_class_from_opcode(key)
            self.add_to_categories(categories, node.category, (node.op_title, node.icon, node.op_code, node.help_text))
            gs.node_dict[node.op_title] = node.op_code
        self.populate_tree(categories)


        self.tree_widget.sortItems(0, Qt.AscendingOrder)

    def add_to_categories(self, categories, category, value):
        parts = category.split('/')
        if len(parts) == 1:
            if category not in categories:
                categories[category] = {"_items": [], "_type": "category"}
            categories[category]["_items"].append(value)
        else:
            if parts[0] not in categories:
                categories[parts[0]] = {"_items": [], "_type": "category"}
            self.add_to_categories(categories[parts[0]], '/'.join(parts[1:]), value)

    def populate_tree(self, categories, parent=None):
        for category, items_or_subcategories in sorted(categories.items(), key=self.custom_sort_key):
            if category == "_items" or category == "_type":
                continue

            if parent is None:
                current_parent = QtWidgets.QTreeWidgetItem(self.tree_widget)
            else:
                current_parent = QtWidgets.QTreeWidgetItem(parent)

            current_parent.setText(0, category.capitalize())

            if "_items" in items_or_subcategories:
                items = items_or_subcategories["_items"]
                items.sort(key=lambda item: item[0])
                for name, icon, op_code, help_text in items:
                    item = QtWidgets.QTreeWidgetItem(current_parent)
                    item.setText(0, name)
                    pixmap = QPixmap(icon)
                    item.setIcon(0, QIcon(pixmap))
                    item.setSizeHint(0, QSize(32, 32))
                    item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
                    item.setToolTip(0, help_text)
                    item.setData(0, Qt.UserRole, pixmap)
                    item.setData(0, Qt.UserRole + 1, op_code)

            if isinstance(items_or_subcategories, dict):
                self.populate_tree(items_or_subcategories, current_parent)

    def custom_sort_key(self, item):
        key, val = item
        type_val = val.get("_type", "item") if isinstance(val, dict) else "item"
        if type_val == 'category':
            # categories are sorted before items
            return (0, key)
        else:
            # items are sorted after categories
            return (1, key)

    def addSubgraphItems(self, categories):
        # Add subgraphs category and files
        subgraph_category = "Subgraphs"
        subgraph_folder = "subgraphs"
        if os.path.isdir(subgraph_folder):
            subgraph_files = [f for f in os.listdir(subgraph_folder) if f.endswith(".json")]
            categories[subgraph_category] = {"_items": []}
            if subgraph_files:
                icon = "ainodes_frontend/icons/base_nodes/v2/load_subgraph.png"
                for file in subgraph_files:
                    categories[subgraph_category]["_items"].append(
                        (file, icon, gs.nodes["subgraph_node"]['op_code'], "Subgraph Nodes"))
        else:
            os.makedirs(subgraph_folder, exist_ok=True)
    def startDrag(self, *args, **kwargs):
        try:
            item = self.tree_widget.currentItem()
            op_code = item.data(0, Qt.UserRole + 1)
            #if op_code is not None:
            pm = False
            itemData = QByteArray()
            dataStream = QDataStream(itemData, QIODevice.WriteOnly)
            if item.data(0, Qt.UserRole) is not None:
                pixmap = QPixmap(item.data(0, Qt.UserRole)).scaled(256, 256, aspectRatioMode=Qt.KeepAspectRatio)
                pm = True
                dataStream << pixmap
            #print("OPCODE WILL BE", op_code)
            #if op_code:
                #try:
                #    op_code = int(op_code)
                #except:
                #    op_code = 99
            #print("TRIED", op_code)
            dataStream.writeInt(int(op_code))

            dataStream.writeQString(item.text(0))
            # Include JSON file data if available
            json_file = None
            if item.parent() and item.parent().text(0).lower() == "subgraphs":
                json_file = item.text(0)

            mimeData = QMimeData()
            mimeData.setData(LISTBOX_MIMETYPE, itemData)
            mimeData.setProperty("filename", json_file)
            drag = QDrag(self)
            drag.setMimeData(mimeData)
            if pm == True:
                drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
                drag.setPixmap(pixmap)

            drag.exec(Qt.MoveAction)

        except Exception as e: dumpException(e)
