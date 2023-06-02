import os
import subprocess
import sys
import threading
from subprocess import run

import requests
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt, QSignalMapper
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QGraphicsView
from qtpy.QtWidgets import QMdiArea, QDockWidget, QAction, QMessageBox, QFileDialog

from ainodes_frontend.base import CalcGraphicsNode
from ainodes_frontend.base.ai_nodes_listbox import QDMDragListbox
from ainodes_frontend.base.node_config import CALC_NODES, import_nodes_from_file, import_nodes_from_subdirectories
from ainodes_frontend.base.node_sub_window import CalculatorSubWindow
from ainodes_frontend.base.settings import load_settings, save_settings, save_error_log
from ainodes_frontend.base.webview_widget import BrowserWidget
from ainodes_frontend.base.worker import Worker
from ainodes_frontend.node_engine.node_edge import Edge
from ainodes_frontend.node_engine.node_edge_validators import (
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node,
    edge_cannot_connect_input_and_output_of_different_type
)
from ainodes_frontend.node_engine.node_editor_window import NodeEditorWindow
from ainodes_frontend.node_engine.utils_no_qt import dumpException, pp

#Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_different_type)

from pyqtgraph.console import ConsoleWidget

# images for the dark skin
DEBUG = False
from ainodes_frontend import singleton as gs

load_settings()

gs.loaded_models = {}
gs.models = {}

class StdoutTextEdit(QtWidgets.QPlainTextEdit):
    signal = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set the maximum block count to 1000
        self.setMaximumBlockCount(1000)
        self.signal.connect(self.write_function)

    def write(self, text):
        self.signal.emit(text)

    def flush(self):
        pass  # no-op, since we're not buffering

    @QtCore.Slot(str)
    def write_function(self, text):
        # Split the text into lines
        lines = text.splitlines()

        # Append the lines without adding extra line breaks
        cursor = self.textCursor()
        for line in lines:
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(line)
            cursor.insertBlock()

        # Scroll to the bottom
        self.ensureCursorVisible()

def remove_empty_lines(file_path):
    with open(file_path, "r+") as f:
        content = f.readlines()
        if not content:
            # if the file is empty, add the base node as the first line
            f.write("XmYx/ainodes_engine_base_nodes\n")
        else:
            if not content[0].strip().startswith("XmYx/ainodes_engine_base_nodes"):
                # if the first line doesn't start with the base node, add it as the first line
                f.seek(0)
                f.write("XmYx/ainodes_engine_base_nodes\n")
                f.writelines(line for line in content if line.strip())
                f.truncate()
            else:
                # if the first line is already the base node, remove empty lines
                f.seek(0)
                f.writelines(line for line in content if line.strip())
                f.truncate()


class ParameterDock(QtWidgets.QDockWidget):
    def __init__(self):
        super().__init__()
        self.main_widget = QtWidgets.QWidget(self)

        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.setWidget(self.main_widget)
        self.setLayout(self.layout)



class MemoryWidget(QtWidgets.QDockWidget):
    def __init__(self):
        super().__init__()
        main_widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QHBoxLayout(main_widget)
        self.setWidget(main_widget)

        self.treeWidget = QtWidgets.QTreeWidget()
        self.setLayout(layout)
        layout.addWidget(self.treeWidget)

        self.populate_tree()

        self.treeWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeWidget.customContextMenuRequested.connect(self.show_context_menu)

        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        layout.addWidget(refresh_button)

    def populate_tree(self):
        self.treeWidget.clear()
        for key, value in gs.models.items():
            item = QtWidgets.QTreeWidgetItem(gs.models[key])
            self.treeWidget.addTopLevelItem(item)

            if isinstance(value, dict):
                self.add_subitems(item, value)

    def add_subitems(self, parent, dict_items):
        for key, value in dict_items.items():
            item = QtWidgets.QTreeWidgetItem([key])
            parent.addChild(item)

            if isinstance(value, dict):
                self.add_subitems(item, value)

    def show_context_menu(self, pos):
        item = self.treeWidget.currentItem()

        if item is not None:
            menu = QtWidgets.QMenu(self)

            delete_action = QAction("Delete", self)
            delete_action.triggered.connect(lambda: self.delete_item(item))
            menu.addAction(delete_action)

            menu.exec_(self.treeWidget.mapToGlobal(pos))

    def delete_item(self, item):
        parent = item.parent()

        if parent is None:
            self.treeWidget.takeTopLevelItem(self.treeWidget.indexOfTopLevelItem(item))
        else:
            parent.removeChild(item)

    def refresh(self):
        self.populate_tree()

class GitHubRepositoriesDialog(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Node Packages")
        main_widget = QtWidgets.QWidget(self)
        layout = QtWidgets.QHBoxLayout(main_widget)
        self.setWidget(main_widget)
        self.list_widget = QtWidgets.QListWidget()
        layout.addWidget(self.list_widget)
        repository_widget = QtWidgets.QWidget()
        repository_layout = QtWidgets.QVBoxLayout(repository_widget)
        repository_label = QtWidgets.QLabel("Node Packages")
        self.repository_name_label = QtWidgets.QLabel()
        self.repository_url_label = QtWidgets.QLabel()
        self.repository_icon_label = QtWidgets.QLabel()
        self.download_button = QtWidgets.QPushButton("Download")
        self.download_button.clicked.connect(self.download_repository)
        self.update_button = QtWidgets.QPushButton("Update / Import")
        self.update_button.clicked.connect(self.update_repository)
        self.update_button.hide()
        repository_layout.addWidget(repository_label)
        repository_layout.addWidget(self.repository_icon_label)
        repository_layout.addWidget(self.repository_name_label)
        repository_layout.addWidget(self.repository_url_label)
        repository_layout.addWidget(self.download_button)
        repository_layout.addWidget(self.update_button)
        layout.addWidget(repository_widget)
        self.list_widget.currentRowChanged.connect(self.show_repository_details)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.right_click_menu)
        self.add_repository_action = QtWidgets.QAction("Add Repository", self)
        self.add_repository_action.triggered.connect(self.add_repository)
        self.delete_repository_action = QtWidgets.QAction("Delete Repository", self)
        self.delete_repository_action.triggered.connect(self.delete_repository)

        self.context_menu = QtWidgets.QMenu(self)
        self.context_menu.addAction(self.add_repository_action)
        self.context_menu.addAction(self.delete_repository_action)

        self.repository_icon_label.setFixedSize(128, 128)
        self.repolist = "repositories.txt"



    def load_repositories(self):
        remove_empty_lines(self.repolist)
        with open("repositories.txt") as f:
            repositories = f.read().splitlines()
        for repository in repositories:
            item = QtWidgets.QListWidgetItem(repository)
            folder = repository.split("/")[1]
            if os.path.isdir(f"custom_nodes/{folder}"):
                item.setBackground(Qt.darkGreen)
                item.setForeground(Qt.white)
                self.update_button.setVisible(True)
                self.download_button.setVisible(False)
            else:
                item.setForeground(Qt.black)
                item.setBackground(Qt.darkYellow)
            self.list_widget.addItem(item)

    def show_repository_details(self, row):
        repository = self.list_widget.item(row).text()
        folder = repository.split("/")[1]
        self.repository_name_label.setText(repository)
        url = f"https://github.com/{repository}"
        self.repository_url_label.setText(url)
        icon_url = f"https://raw.githubusercontent.com/{repository}/main/icon.png"
        icon_pixmap = QtGui.QPixmap()
        icon_pixmap.loadFromData(requests.get(icon_url).content)
        self.repository_icon_label.setPixmap(icon_pixmap)
        self.repository_icon_label.setScaledContents(True)
        if os.path.isdir(f"./custom_nodes/{folder}"):
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.show()
            self.download_button.hide()
        else:
            self.list_widget.currentItem().setBackground(Qt.darkYellow)
            self.update_button.hide()
            self.download_button.show()

    def download_repository(self):
        worker = Worker(self.download_repository_thread)
        worker.signals.result.connect(self.download_repository_finished)
        self.parent.threadpool.start(worker)
    def download_repository_thread(self, progress_callback=None):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        command = f"git clone https://github.com/{repository} ./custom_nodes/{folder} && pip install -r ./custom_nodes/{folder}/requirements.txt"
        result = run(command, shell=True, stdout=self.parent.text_widget, stderr=self.parent.text_widget)
        return result
    @QtCore.Slot(object)
    def download_repository_finished(self, result):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        if result.returncode == 0:
            import_nodes_from_subdirectories(f"./custom_nodes/{folder}")
            self.parent.nodesListWidget.addMyItems()
            QtWidgets.QMessageBox.information(self, "Download Complete", f"{repository} was downloaded successfully.")
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.hide()
        else:
            QtWidgets.QMessageBox.critical(self, "Download Failed",
            f"An error occurred while downloading {repository}:\n{result.stderr.decode()}")

    def update_repository(self, skip_update=False):
        self.skip_update = skip_update
        worker = Worker(self.update_repository_thread)
        worker.signals.result.connect(self.update_repository_finished)
        self.parent.threadpool.start(worker)

    def update_repository_thread(self, progress_callback=None):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        #command = f"git -C ./custom_nodes/{folder} stash && git -C ./custom_nodes/{folder} pull && pip install -r ./custom_nodes/{folder}/requirements.txt"
        if self.skip_update == False:
            command = f"git -C ./custom_nodes/{folder} pull && pip install -r ./custom_nodes/{folder}/requirements.txt"
            result = run(command, shell=True, stdout=self.parent.text_widget, stderr=self.parent.text_widget)
        else:
            result = None
        return result
    @QtCore.Slot(object)
    def update_repository_finished(self, result):
        repository = self.repository_name_label.text()
        folder = repository.split("/")[1]
        if result != None:
            if result.returncode == 0:
                import_nodes_from_subdirectories(f"./custom_nodes/{folder}")
                self.parent.nodesListWidget.addMyItems()
                QtWidgets.QMessageBox.information(self, "Update Complete", f"{repository} was updated successfully.")
                self.list_widget.currentItem().setBackground(Qt.darkGreen)
                self.update_button.hide()

            else:
                QtWidgets.QMessageBox.critical(self, "Update Failed",
                                     f"An error occurred while updating {repository}:\n{result.stderr.decode()}")
        elif result == None:
            import_nodes_from_subdirectories(f"./custom_nodes/{folder}")
            self.parent.nodesListWidget.addMyItems()
            #QtWidgets.QMessageBox.information(self, "Import Complete", f"{repository} was imported successfully.")
            self.list_widget.currentItem().setBackground(Qt.darkGreen)
            self.update_button.hide()

    def add_repository(self):
        remove_empty_lines(self.repolist)
        text, ok = QtWidgets.QInputDialog.getText(self, "Add Repository",
                                                  "Enter the repository name (e.g. owner/repo):")
        if ok and text:
            with open("repositories.txt", "r") as f:
                lines = f.read().splitlines()
                if text in lines:
                    QtWidgets.QMessageBox.warning(self, "Duplicate Repository", f"{text} already exists in the list.")
                    return
            with open("repositories.txt", "a") as f:
                if lines:
                    f.write("\n")
                f.write(text)
            item = QtWidgets.QListWidgetItem(text)
            item.setForeground(Qt.black)
            item.setBackground(Qt.darkYellow)
            self.list_widget.addItem(item)

    def delete_repository(self):
        remove_empty_lines(self.repolist)
        row = self.list_widget.currentRow()
        if row != -1:
            repository = self.list_widget.item(row).text()
            if repository == "XmYx/ainodes_engine_base_nodes":
                QtWidgets.QMessageBox.warning(self, "Default Repository", "The default repository cannot be deleted.")
                return
            with open("repositories.txt", "r") as f:
                lines = f.readlines()
            with open("repositories.txt", "w") as f:
                for line in lines:
                    if line.strip() != repository:
                        f.write(line)
            self.list_widget.takeItem(row)
    def right_click_menu(self, position):
        self.context_menu.exec_(self.list_widget.mapToGlobal(position))

    def import_base_repositories(self):
        base_repo = 'ainodes_engine_base_nodes'
        import_nodes_from_subdirectories(f"custom_nodes/{base_repo}")
        if os.path.isdir('custom_nodes/ainodes_engine_deforum_nodes'):
            deforum_repo = 'ainodes_engine_deforum_nodes'
            import_nodes_from_subdirectories(f"custom_nodes/{deforum_repo}")

class StreamRedirect(QtCore.QObject):
    text_written = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.stdout_lock = threading.Lock()
        self.stderr_lock = threading.Lock()
        sys.stdout = self
        sys.stderr = self

    def write(self, text):
        self.text_written.emit(text)

    def flush(self):
        pass

    def fileno(self):
        return self.stdout.fileno()

    def decode_output(self, output_bytes):
        try:
            decoded_output = output_bytes.decode(sys.stdout.encoding)
        except UnicodeDecodeError:
            decoded_output = output_bytes.decode(sys.stdout.encoding, errors='replace')
        return decoded_output

    def write_stdout(self, output_bytes):
        with self.stdout_lock:
            decoded_output = self.decode_output(output_bytes)
            self.stdout.write(decoded_output)

    def write_stderr(self, output_bytes):
        with self.stderr_lock:
            decoded_output = self.decode_output(output_bytes)
            self.stderr.write(decoded_output)
class NodesConsole(ConsoleWidget):
    def __init__(self):
        super().__init__()
        stylesheet = '''
        QWidget#Form {
            background-color: black;
        }

        QPlainTextEdit#output {
            background-color: black;
            color: white;
            font-family: Monospace;
        }

        QLineEdit#input {
            background-color: black;
            color: white;
            font-family: Monospace;
            border: none;
        }

        QPushButton#historyBtn,
        QPushButton#exceptionBtn,
        QPushButton#clearExceptionBtn,
        QPushButton#catchAllExceptionsBtn,
        QPushButton#catchNextExceptionBtn {
            background-color: black;
            color: white;
            border: none;
        }

        QCheckBox#onlyUncaughtCheck,
        QCheckBox#runSelectedFrameCheck {
            color: white;
        }

        QListWidget#historyList,
        QListWidget#exceptionStackList {
            background-color: black;
            color: white;
            font-family: Monospace;
            border: none;
        }

        QGroupBox#exceptionGroup {
            border: 1px solid white;
        }

        QLabel#exceptionInfoLabel,
        QLabel#label {
            color: white;
        }

        QLineEdit#filterText {
            background-color: black;
            color: white;
            border: none;
        }

        QSplitter::handle {
            background-color: white;
        }

        QSplitter::handle:vertical {
            height: 6px;
        }

        QSplitter::handle:pressed {
            background-color: #888888;
        }
        '''

        # Apply the stylesheet to the application
        self.setStyleSheet(stylesheet)
    def write(self, strn, html=False, scrollToBottom=True):
        """Write a string into the console.

        If scrollToBottom is 'auto', then the console is automatically scrolled
        to fit the new text only if it was already at the bottom.
        """
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return

        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom

        self.output.moveCursor(QtGui.QTextCursor.MoveOperation.End)
        if html:
            self.output.textCursor().insertHtml(strn)
        else:
            if self.inCmd:
                self.inCmd = False
                self.output.textCursor().insertHtml("</div><br><div style='font-weight: normal; background-color: #FFF; color: black'>")
            self.output.insertPlainText(strn)

        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

class ColorEditor(QtWidgets.QDialog):
    def __init__(self, colors, names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Editor")
        self.colors = colors
        self.names = names

        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout()

        self.list_widget = QtWidgets.QListWidget()
        for index, color in enumerate(self.colors):
            item = QtWidgets.QListWidgetItem()
            item.setBackground(color)
            item_name = self.names.get(index, "UNUSED")
            item.setText(item_name)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        edit_btn = QtWidgets.QPushButton("Edit Color")
        edit_btn.clicked.connect(self.edit_color)
        layout.addWidget(edit_btn)

        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        layout.addWidget(save_btn)

        self.setLayout(layout)

    def edit_color(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            initial_color = current_item.background().color()
            new_color = QtWidgets.QColorDialog.getColor(initial_color, self)
            if new_color.isValid():
                current_item.setBackground(new_color)

    def get_updated_colors(self):
        updated_colors = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            updated_colors.append(item.background().color())
        return updated_colors
class CalculatorWindow(NodeEditorWindow):
    file_open_signal = QtCore.Signal(object)
    base_repo_signal = QtCore.Signal()
    file_new_signal = QtCore.Signal(object)
    json_open_signal = QtCore.Signal(object)
    def __init__(self, parent=None):
        super(CalculatorWindow, self).__init__()


    def eventListener(self, *args, **kwargs):
        print("cleaning up")
        self.cleanup()
        save_settings()
        save_error_log()
    def initUI(self):
        #self.setup_defaults()


        self.name_company = 'aiNodes'
        self.name_product = 'AI Node Editor'

        #self.stylesheet_filename = os.path.join(os.path.dirname(__file__), gs.qss)
        #loadStylesheets(
        #    os.path.join(os.path.dirname(__file__), gs.qss),
        #    self.stylesheet_filename
        #)

        self.empty_icon = QIcon("")

        if DEBUG:
            print("Registered nodes:")
            pp(CALC_NODES)

        self.mdiArea = QMdiArea()
        #self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        #self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdiArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mdiArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.mdiArea.setViewMode(QMdiArea.TabbedView)
        self.mdiArea.setDocumentMode(True)
        self.mdiArea.setTabsClosable(True)
        self.mdiArea.setTabsMovable(True)
        self.setCentralWidget(self.mdiArea)

        self.mdiArea.subWindowActivated.connect(self.updateMenus)
        self.windowMapper = QSignalMapper(self)
        self.windowMapper.mappedInt[int].connect(self.setActiveSubWindow)

        # Connect the signal mapper to the map slot using a lambda function
        self.windowMapper.mappedInt[int].connect(lambda id: self.setActiveSubWindow(self.subWindowList()[id]))
        self.createNodesDock()
        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()
        self.updateMenus()
        self.readSettings()
        self.setWindowTitle("aiNodes - Engine")
        #self.show_github_repositories()
        if not gs.args.no_console:
            self.create_console_widget()
            #self.tabifyDockWidget(self.node_packages, self.console)
        #self.threadpool = QtCore.QThreadPool()

        #self.parameter_dock = ParameterDock()
        #self.addDockWidget(Qt.LeftDockWidgetArea, self.parameter_dock)

        self.file_open_signal.connect(self.fileOpen)
        self.file_new_signal.connect(self.onFileNew_subgraph)
        self.json_open_signal.connect(self.onJsonOpen_subgraph)
        self.base_repo_signal.connect(self.import_base_repos)
        icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
        self.setWindowIcon(icon)

        print("aiNodes Ready.")
        print("------------------")
        print(f"|{len(CALC_NODES)} Nodes loaded.|")
        print("------------------")

        # Create a QDockWidget
        self.bdock_widget = QDockWidget("Embedded Browser", self)

        # Set the BrowserWidget as the central widget of the QDockWidget
        browser_widget = BrowserWidget()
        self.bdock_widget.setWidget(browser_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.bdock_widget)
        self.bdock_widget.setVisible(False)
        self.subgraph = None
    def cleanup(self):
        try:
            self.training_thread.terminate_process()
            self.training_thread.join()
            del self.training_thread
        except:
            pass
    def import_base_repos(self):
        """base_repo = 'ainodes_engine_base_nodes'
        import_nodes_from_subdirectories(f"custom_nodes/{base_repo}")
        if os.path.isdir('custom_nodes/ainodes_engine_deforum_nodes'):
            deforum_repo = 'ainodes_engine_deforum_nodes'
            import_nodes_from_subdirectories(f"custom_nodes/{deforum_repo}")"""

        base_folder = 'custom_nodes'
        for folder in os.listdir(base_folder):
            folder_path = os.path.join(base_folder, folder)
            if "__pycache__" not in folder_path and "_nodes" in folder_path:
                if os.path.isdir(folder_path):
                    import_nodes_from_subdirectories(folder_path)


    def edit_colors(self):
        editor = ColorEditor(gs.SOCKET_COLORS, gs.socket_names)
        result = editor.exec_()
        if result == QtWidgets.QDialog.Accepted:
            gs.SOCKET_COLORS = editor.get_updated_colors()
    def toggleDockWidgets(self):
        # Get the current visibility state of the dock widgets
        if not gs.args.no_console:
            consoleVisible = self.console.isVisible()
            self.console.setVisible(not consoleVisible)

        packagesVisible = self.node_packages.isVisible()

        # Toggle the visibility of the dock widgets

        self.node_packages.setVisible(not packagesVisible)
    def toggleNodesDock(self):
        # Get the current visibility state of the dock widgets
        consoleVisible = self.nodesDock.isVisible()

        # Toggle the visibility of the dock widgets
        self.nodesDock.setVisible(not consoleVisible)
    def toggleFullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:

        #super().keyPressEvent(event)
        if event.key() == 96:
            self.toggleDockWidgets()
        elif event.key() == 16777274:
            self.toggleFullscreen()
        elif event.key() == 16777216:
            self.toggleNodesDock()

    def create_console_widget(self):
        # Create a text widget for stdout and stderr
        self.text_widget = NodesConsole()
        # Set up the StreamRedirect objects
        self.stdout_redirect = StreamRedirect()
        self.stderr_redirect = StreamRedirect()
        #self.stdin_redirect = StreamRedirect()
        # Connect the text_written signal to the text_widget's append method
        self.stdout_redirect.text_written.connect(self.text_widget.write)
        #self.stdin_redirect.text_written.connect(self.text_widget.write)
        self.stderr_redirect.text_written.connect(self.text_widget.write)
        # Redirect stdout and stderr to the StreamRedirect objects
        sys.stdout = self.stdout_redirect
        #sys.stdin = self.stdin_redirect
        sys.stderr = self.stderr_redirect

        self.console = QDockWidget()
        self.console.setWindowTitle("Console")
        self.console.setAllowedAreas(Qt.BottomDockWidgetArea)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(5,5,5,5)
        layout.addWidget(self.text_widget)
        self.console.setWidget(widget)
        #layout.addWidget(self.text_widget2)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.console)

    def show_memory_widget(self):
        self.mem_widget = MemoryWidget()
        self.mem_widget.show()
    def show_github_repositories(self):
        self.node_packages = GitHubRepositoriesDialog(self)
        self.node_packages.load_repositories()
        self.addDockWidget(Qt.BottomDockWidgetArea, self.node_packages)
        #dialog.show()

    def closeEvent(self, event):
        self.mdiArea.closeAllSubWindows()
        if self.mdiArea.currentSubWindow():
            event.ignore()
        else:
            self.writeSettings()
            event.accept()
            # hacky fix for PyQt 5.14.x
            import sys
            sys.exit(0)


    def createActions(self):
        super().createActions()
        self.actNode = QAction('&Add Node', self, shortcut='Ctrl+L', statusTip="Open new node", triggered=self.onNodeOpen)
        self.actNodePacks = QAction('&Node Packages', self, shortcut='Ctrl+K', statusTip="Download Nodes", triggered=self.show_github_repositories)


        self.actClose = QAction("Cl&ose", self, statusTip="Close the active window", triggered=self.mdiArea.closeActiveSubWindow)
        self.actCloseAll = QAction("Close &All", self, statusTip="Close all the windows", triggered=self.mdiArea.closeAllSubWindows)
        self.actTile = QAction("&Tile", self, statusTip="Tile the windows", triggered=self.mdiArea.tileSubWindows)
        self.actCascade = QAction("&Cascade", self, statusTip="Cascade the windows", triggered=self.mdiArea.cascadeSubWindows)
        self.actNext = QAction("Ne&xt", self, shortcut=QKeySequence.NextChild, statusTip="Move the focus to the next window", triggered=self.mdiArea.activateNextSubWindow)
        self.actPrevious = QAction("Pre&vious", self, shortcut=QKeySequence.PreviousChild, statusTip="Move the focus to the previous window", triggered=self.mdiArea.activatePreviousSubWindow)

        self.actSeparator = QAction(self)
        self.actSeparator.setSeparator(True)

        self.actAbout = QAction("&About", self, statusTip="Show the application's About box", triggered=self.about)
        self.actBrowser = QAction("&Browser", self, statusTip="Show the application's Browser", triggered=self.browser)
        self.actTraining = QAction("&Training", self, statusTip="Start Kohya", triggered=self.training_gui)
        self.actColors = QAction("&Change Colors", self, statusTip="Change socket / route color palette", triggered=self.edit_colors)
        self.actRClickMenu = QAction("&Alternate context menu", self, statusTip="Change context menu type", triggered=self.toggle_menu)
        # Create a checkable QAction
        self.actRClickMenu.setCheckable(True)

    def training_gui(self):
        if hasattr(self, "training_thread"):
            self.cleanup()
        from custom_nodes.ainodes_engine_base_nodes.ainodes_backend.training_thread import TrainingThread
        gs.threads['lora'] = TrainingThread(self.stdout_redirect)
        gs.threads['lora'].start()
    def browser(self):
        self.bdock_widget.setVisible(not self.bdock_widget.isVisible())
    def toggle_menu(self):
        widget = self.getCurrentNodeEditorWidget()
        if widget is not None:
            widget.context_menu_style = 'classic' if widget.context_menu_style == 'modern' else 'modern'
    def getCurrentNodeEditorWidget(self):
        """ we're returning NodeEditorWidget here... """
        activeSubWindow = self.mdiArea.activeSubWindow()
        if activeSubWindow:
            return activeSubWindow.widget()
        return None
    def onNodeOpen(self):
        """Handle File Open operation"""
        fname, filter = QFileDialog.getOpenFileName(None, 'Open graph from file', f"{self.getFileDialogDirectory()}/ainodes_frontend/custom_nodes", 'Python Files (*.py)')
        if fname != '' and os.path.isfile(fname):
            import_nodes_from_file(fname)
            self.nodesListWidget.addMyItems()

    def onFileNew(self):
        try:
            subwnd = self.createMdiChild()
            subwnd.widget().fileNew()
            icon = QtGui.QIcon("ainodes_frontend/qss/icon.ico")
            subwnd.setWindowIcon(icon)
            subwnd.show()

            # Install event filter on the subwnd object
            subwnd.installEventFilter(self)


        except Exception as e: dumpException(e)

    @QtCore.Slot(object)
    def onFileNew_subgraph(self, node):
        try:
            subwnd = self.createMdiChild()
            subwnd.widget().fileNew()
            icon = QtGui.QIcon("ainodes_frontend/qss/icon.ico")
            subwnd.setWindowIcon(icon)
            subwnd.setWindowTitle(str(node.name))
            node.graph_window = subwnd
            subwnd.subgraph = True
            subwnd.show()

            # Install event filter on the subwnd object
            subwnd.installEventFilter(self)

        except Exception as e:
            dumpException(e)
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Close:
            if obj.subgraph:
                event.ignore()  # Ignore the close event
                return True
            else:
                nodes = obj.widget().scene.nodes
                for node in nodes:
                    if hasattr(node, "graph_window"):
                        node.graph_window.subgraph = None
                        node.graph_window.close()

        return super().eventFilter(obj, event)
    @QtCore.Slot(object)
    def onJsonOpen_subgraph(self, node):

        json_graph = node.graph_json
        json_name = node.name


        #fnames, filter = QFileDialog.getOpenFileNames(self, 'Open graph from file', f"{self.getFileDialogDirectory()}/graphs", self.getFileDialogFilter())

        try:
            existing = self.findMdiChild(json_name)
            if existing:
                self.mdiArea.setActiveSubWindow(existing)
            else:
                # we need to create new subWindow and open the file
                nodeeditor = CalculatorSubWindow()
                if nodeeditor.jsonLoad(json_graph, json_name):
                    self.statusBar().showMessage("File %s loaded" % json_name, 5000)
                    nodeeditor.setTitle()
                    subwnd = self.createMdiChild(nodeeditor)
                    icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
                    subwnd.setWindowIcon(icon)
                    node.graph_window = subwnd
                    subwnd.subgraph = True
                    subwnd.show()
                    # Install event filter on the subwnd object
                    subwnd.installEventFilter(self)

                else:
                    nodeeditor.close()
        except Exception as e: dumpException(e)


    def onFileOpen(self):
        fnames, filter = QFileDialog.getOpenFileNames(self, 'Open graph from file', f"{self.getFileDialogDirectory()}/graphs", self.getFileDialogFilter())

        try:
            for fname in fnames:
                if fname:
                    existing = self.findMdiChild(fname)
                    if existing:
                        self.mdiArea.setActiveSubWindow(existing)
                    else:
                        # we need to create new subWindow and open the file
                        nodeeditor = CalculatorSubWindow()
                        if nodeeditor.fileLoad(fname):
                            self.statusBar().showMessage("File %s loaded" % fname, 5000)
                            nodeeditor.setTitle()
                            subwnd = self.createMdiChild(nodeeditor)
                            icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
                            subwnd.setWindowIcon(icon)
                            subwnd.show()
                            # Install event filter on the subwnd object
                            subwnd.installEventFilter(self)


                        else:
                            nodeeditor.close()
        except Exception as e: dumpException(e)
    def onFileOpen_subgraph(self, graph):
        try:
            existing = self.findMdiChild(fname)
            if existing:
                self.mdiArea.setActiveSubWindow(existing)
            else:
                # we need to create new subWindow and open the file
                nodeeditor = CalculatorSubWindow()
                if nodeeditor.fileLoad(fname):
                    self.statusBar().showMessage("File %s loaded" % fname, 5000)
                    nodeeditor.setTitle()
                    subwnd = self.createMdiChild(nodeeditor)
                    icon = QtGui.QIcon("ainodes_frontend/qss/icon.png")
                    subwnd.setWindowIcon(icon)
                    subwnd.show()
                else:
                    nodeeditor.close()
        except Exception as e: dumpException(e)
    @QtCore.Slot(object)
    def fileOpen(self, file):

        print("OPENING")

        fnames = [file]
        try:
            for fname in fnames:
                if fname:
                    existing = self.findMdiChild(fname)
                    if existing:
                        self.mdiArea.setActiveSubWindow(existing)
                    else:
                        # we need to create new subWindow and open the file
                        nodeeditor = CalculatorSubWindow()
                        if nodeeditor.fileLoad(fname):
                            self.statusBar().showMessage("File %s loaded" % fname, 5000)
                            nodeeditor.setTitle()
                            subwnd = self.createMdiChild(nodeeditor)
                            subwnd.show()
                            # Install event filter on the subwnd object
                            subwnd.installEventFilter(self)

                        else:
                            nodeeditor.close()
        except Exception as e: dumpException(e)


    def about(self):
        QMessageBox.about(self, "About aiNodes engine",
                "<b>aiNodes engine</b> v0.1 "
                "for more information, please visit the GitHub Page"
                "<a href='https://www.github.com/XmYx/ainodes-engine'>aiNodes Engine GitHub</a>")

    def createMenus(self):
        super().createMenus()

        self.windowMenu = self.menuBar().addMenu("&Window")
        self.updateWindowMenu()
        self.windowMenu.aboutToShow.connect(self.updateWindowMenu)


        self.menuBar().addSeparator()

        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.actAbout)
        self.helpMenu.addAction(self.actTraining)
        self.helpMenu.addAction(self.actBrowser)


        self.editMenu.aboutToShow.connect(self.updateEditMenu)

        self.fileMenu.addAction(self.actNode)
        self.fileMenu.addAction(self.actNodePacks)
        # Get the index of the action in the fileMenu
        action_index = self.fileMenu.actions().index(self.actNode)

        # Insert the action at the new index
        new_index = max(0, action_index - 4)
        self.fileMenu.insertAction(self.fileMenu.actions()[new_index], self.actNode)
        # Get the index of the action in the fileMenu
        action_index = self.fileMenu.actions().index(self.actNodePacks)

        # Insert the action at the new index
        new_index = max(0, action_index - 4)
        self.fileMenu.insertAction(self.fileMenu.actions()[new_index], self.actNodePacks)
    def updateMenus(self):
        # print("update Menus")
        active = self.getCurrentNodeEditorWidget()
        hasMdiChild = (active is not None)

        self.actSave.setEnabled(hasMdiChild)
        self.actSaveAs.setEnabled(hasMdiChild)
        self.actClose.setEnabled(hasMdiChild)
        self.actCloseAll.setEnabled(hasMdiChild)
        self.actTile.setEnabled(hasMdiChild)
        self.actCascade.setEnabled(hasMdiChild)
        self.actNext.setEnabled(hasMdiChild)
        self.actPrevious.setEnabled(hasMdiChild)
        self.actSeparator.setVisible(hasMdiChild)

        self.updateEditMenu()

    def updateEditMenu(self):
        try:
            # print("update Edit Menu")
            active = self.getCurrentNodeEditorWidget()
            hasMdiChild = (active is not None)

            self.actPaste.setEnabled(hasMdiChild)

            self.actCut.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actCopy.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actDelete.setEnabled(hasMdiChild and active.hasSelectedItems())

            self.actUndo.setEnabled(hasMdiChild and active.canUndo())
            self.actRedo.setEnabled(hasMdiChild and active.canRedo())
        except Exception as e: dumpException(e)



    def updateWindowMenu(self):
        self.windowMenu.clear()

        toolbar_nodes = self.windowMenu.addAction("Nodes Toolbar")
        toolbar_nodes.setCheckable(True)
        toolbar_nodes.triggered.connect(self.onWindowNodesToolbar)
        toolbar_nodes.setChecked(self.nodesDock.isVisible())
        self.windowMenu.addAction(self.actColors)
        self.windowMenu.addAction(self.actRClickMenu)
        self.windowMenu.addSeparator()

        self.windowMenu.addAction(self.actClose)
        self.windowMenu.addAction(self.actCloseAll)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actTile)
        self.windowMenu.addAction(self.actCascade)
        self.windowMenu.addSeparator()
        self.windowMenu.addAction(self.actNext)
        self.windowMenu.addAction(self.actPrevious)
        self.windowMenu.addAction(self.actSeparator)

        windows = self.mdiArea.subWindowList()
        self.actSeparator.setVisible(len(windows) != 0)

        for i, window in enumerate(windows):
            child = window.widget()

            text = "%d %s" % (i + 1, child.getUserFriendlyFilename())
            if i < 9:
                text = '&' + text

            action = self.windowMenu.addAction(text)
            action.setCheckable(True)
            action.setChecked(child is self.getCurrentNodeEditorWidget())
            action.triggered.connect(self.windowMapper.map)
            self.windowMapper.setMapping(action, window)

    def onWindowNodesToolbar(self):
        if self.nodesDock.isVisible():
            self.nodesDock.hide()
        else:
            self.nodesDock.show()

    def createToolBars(self):
        pass

    def createNodesDock(self):
        self.nodesListWidget = QDMDragListbox()

        self.nodesDock = QDockWidget("Nodes")
        self.nodesDock.setWidget(self.nodesListWidget)
        self.nodesDock.setFloating(False)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.nodesDock)

    def createStatusBar(self):
        self.statusBar().showMessage("Ready")

    def createMdiChild(self, child_widget=None):
        nodeeditor = child_widget if child_widget is not None else CalculatorSubWindow()
        subwnd = self.mdiArea.addSubWindow(nodeeditor)
        #subwnd.setWindowIcon(self.empty_icon)
        icon = QtGui.QIcon("ainodes_frontend/qss/icon.ico")
        subwnd.setWindowIcon(icon)
        subwnd.subgraph = None

        # node_engine.scene.addItemSelectedListener(self.updateEditMenu)
        # node_engine.scene.addItemsDeselectedListener(self.updateEditMenu)
        #nodeeditor.scene.addItemSelectedListener(self.emitUIobjects)

        nodeeditor.scene.history.addHistoryModifiedListener(self.updateEditMenu)
        nodeeditor.addCloseEventListener(self.onSubWndClose)
        #subwnd.windowStateChanged.connect(partial(self.onSubWndFocusChanged, subwnd))

        return subwnd

    def onSubWndFocusChanged(self, subwnd, state, state_2):
        if state == Qt.WindowActive:
            view = subwnd.widget().scene.grScene.scene.getView()
            if state_2 == Qt.WindowState.WindowNoState:
                view.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)
                #print("STOPPED")
            else:
                #view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
                view.setViewportUpdateMode(QGraphicsView.MinimalViewportUpdate)
                #print("STARTED")
                #view.update()
        elif state == Qt.WindowMinimized:
            pass
    def emitUIobjects(self, item):

        # Remove any existing widgets from the layout
        self.clear_layout(self.parameter_dock.layout)

        if isinstance(item, CalcGraphicsNode):
            # Remove any existing widgets from the layout
            if hasattr(item, "content"):
                # Save the widgets and layouts in a list within the item.node
                item.node.saved_widgets_and_layouts = []

                for widget in item.content.widget_list:
                    print(widget)
                    widget.setParent(None)
                    item.node.saved_widgets_and_layouts.append(widget)
                    if isinstance(widget, QtWidgets.QHBoxLayout) or isinstance(widget, QtWidgets.QVBoxLayout):
                        self.parameter_dock.layout.addLayout(widget)
                    else:
                        self.parameter_dock.layout.addWidget(widget)

    def clear_layout(self, layout):
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                child_layout = item.layout()
                if child_layout is not None:
                    self.clear_layout(child_layout)
                    # Restore the parent of the child layout if it was saved in item.node
                    subwnd = self.getCurrentNodeEditorWidget()
                    for node_item in subwnd.scene.nodes:
                        if hasattr(node_item,
                                   "saved_widgets_and_layouts") and child_layout in node_item.saved_widgets_and_layouts:
                            node_item.saved_widgets_and_layouts.remove(child_layout)
                            node_item.saved_widgets_and_layouts.append(child_layout)
                            child_layout.setParent(node_item.content.main_layout)
                            break
    def onSubWndClose(self, widget, event):
        existing = self.findMdiChild(widget.filename)
        self.mdiArea.setActiveSubWindow(existing)

        if self.maybeSave():
            event.accept()
        else:
            event.ignore()


    def findMdiChild(self, filename):
        for window in self.mdiArea.subWindowList():
            if window.widget().filename == filename:
                return window
        return None


    def setActiveSubWindow(self, window):
        if window:
            self.mdiArea.setActiveSubWindow(window)
            #window.widget().setAttribute(Qt.WA_PaintOnScreen, True)

    def resumePaintEvents(self):
        # Resume paint events for all windows
        for window in self.mdiArea.subWindowList():
            window.widget().setAttribute(Qt.WA_PaintOnScreen, True)
            self.resumeSceneUpdates(window.scene)

    def pauseSceneUpdates(self, scene):

        print("Pausing rendering", scene)
        view = scene.grScene.scene.getView()  # Assuming there is only one view associated with the scene
        view.setViewportUpdateMode(QGraphicsView.NoViewportUpdate)

    def resumeSceneUpdates(self, scene):
        print("Resuming rendering", scene)

        view = scene.grScene.scene.getView()  # Assuming there is only one view associated with the scene
        view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        view.update()

