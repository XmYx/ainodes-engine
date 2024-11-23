import re
import sys
import threading

from pyqtgraph.console import ConsoleWidget
from qtpy import QtGui, QtCore


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

    def write_(self, strn, html=False, scrollToBottom=True):
        sys.__stdout__.write(strn)
        sb = self.output.verticalScrollBar()

        # Remove control characters used by tqdm
        filtered_strn = re.sub(r'\x1b\[.*?[@-~]', '', strn)

        self.output.insertPlainText(filtered_strn)
        sb.setValue(sb.maximum())

    def write(self, strn, style='output', scrollToBottom='auto'):
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return

        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.output.setTextCursor(cursor)

        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom

        # Check if the string contains the carriage return character, which is used by tqdm to refresh the progress bar
        if '\r' in strn:
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock,
                                QtGui.QTextCursor.MoveMode.KeepAnchor)

            cursor.removeSelectedText()  # Remove the last line
            strn = strn.replace('\r', '')  # Remove carriage return
            if strn.startswith('\n'):  # If there's a newline at the start after removing \r, remove it
                strn = strn[1:]
        # Insert the new text
        self.output.insertPlainText(strn)

        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

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
    def isatty(self):
        return False
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