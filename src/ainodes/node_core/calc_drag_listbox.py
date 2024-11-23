from PyQt5.QtGui import QPixmap, QIcon, QDrag
from PyQt5.QtCore import QSize, Qt, QByteArray, QDataStream, QMimeData, QIODevice, QPoint
from PyQt5.QtWidgets import (
    QListWidget, QAbstractItemView, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
    QWidget, QVBoxLayout, QLineEdit, QMessageBox
)

from node_core.node_register import NODE_CLASSES, LISTBOX_MIMETYPE
from nodeeditor.utils import dumpException

class DraggableTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Ensure drag is enabled
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)

    def startDrag(self, supportedActions):
        try:
            item = self.currentItem()
            if not item:
                return

            # Ensure it's a leaf node (i.e., a node class, not a category)
            if item.childCount() > 0:
                return  # Do not allow dragging categories

            class_name = item.data(0, Qt.UserRole + 1)
            pixmap = QPixmap(item.data(0, Qt.UserRole))

            itemData = QByteArray()
            dataStream = QDataStream(itemData, QIODevice.WriteOnly)
            dataStream << pixmap
            dataStream.writeQString(class_name)
            dataStream.writeQString(item.text(0))

            mimeData = QMimeData()
            mimeData.setData(LISTBOX_MIMETYPE, itemData)

            drag = QDrag(self)
            drag.setMimeData(mimeData)
            drag.setHotSpot(QPoint(pixmap.width() // 2, pixmap.height() // 2))
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)
        except Exception as e:
            dumpException(e)

class QDMDragListbox(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        # Create the main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Create and add the search bar
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Search nodes...")
        self.layout.addWidget(self.search_bar)

        # Create and add the tree widget
        self.tree_widget = DraggableTreeWidget()
        self.tree_widget.setHeaderHidden(True)  # Hide the header for a cleaner look
        self.tree_widget.setIconSize(QSize(32, 32))
        self.tree_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree_widget.setRootIsDecorated(True)  # Show expand/collapse indicators

        self.layout.addWidget(self.tree_widget)

        # Connect the search bar signal to the filter function
        self.search_bar.textChanged.connect(self.filter_nodes)

        # Populate the tree with nodes
        self.addMyItems()

    def addMyItems(self):
        self.tree_widget.clear()
        class_names = list(NODE_CLASSES.keys())
        class_names.sort()
        for class_name in class_names:
            node_class = NODE_CLASSES[class_name]
            category_path = getattr(node_class, 'category', '')
            # Handle empty category paths by defaulting to 'Uncategorized'
            if not category_path:
                category_path = 'Uncategorized'
            self.addMyItem(category_path, node_class.op_title, node_class.icon, class_name)

    def addMyItem(self, category_path, name, icon=None, class_name=""):
        # Split the category path by '/' and filter out empty strings
        categories = [cat for cat in category_path.split('/') if cat]
        if not categories:
            categories = ['Uncategorized']  # Default category if none provided

        parent = self.tree_widget.invisibleRootItem()  # Start from the root

        # Traverse or create the category hierarchy
        for category in categories:
            # Check if the category already exists under the current parent
            found = False
            for i in range(parent.childCount()):
                child = parent.child(i)
                if child.text(0) == category:
                    parent = child
                    found = True
                    break
            if not found:
                # Create a new category item
                category_item = QTreeWidgetItem(parent)
                category_item.setText(0, category)
                # Optionally, set a default icon for categories
                category_item.setIcon(0, QIcon(QPixmap(".")))  # Replace with a category icon if available
                category_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                parent = category_item

        # Now add the actual node as a child of the last category
        node_item = QTreeWidgetItem(parent)
        pixmap = QPixmap(icon) if icon else QPixmap(".")
        node_item.setIcon(0, QIcon(pixmap))
        node_item.setText(0, name)

        print("ADDED", name)

        node_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)

        # Store class_name as data for drag operation
        node_item.setData(0, Qt.UserRole, pixmap)
        node_item.setData(0, Qt.UserRole + 1, class_name)

    # Remove startDrag from this class, as it's now in DraggableTreeWidget

    def filter_nodes(self, text):
        """
        Filters the tree widget based on the search text.
        Shows only the nodes that match the search text and their parent categories.
        """
        text = text.lower()
        self.tree_widget.collapseAll()

        def filter_item(item):
            match = text in item.text(0).lower()
            child_matches = False
            for i in range(item.childCount()):
                child = item.child(i)
                if filter_item(child):
                    child_matches = True
            if match or child_matches:
                item.setHidden(False)
                if child_matches:
                    self.expand_parents(item)
                return True
            else:
                item.setHidden(True)
                return False

        root = self.tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            child = root.child(i)
            filter_item(child)

    def expand_parents(self, item):
        """
        Expands all parent categories of the given item.
        """
        parent = item.parent()
        while parent:
            self.tree_widget.expandItem(parent)
            parent = parent.parent()
