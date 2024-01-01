from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics
from PyQt6.QtWidgets import QVBoxLayout, QDialog
from qtpy import QtCore, QtGui
from qtpy import QtWidgets
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

from pyflakes.reporter import Reporter
import ast
from textwrap import dedent
class CustomReporter(Reporter):
    def __init__(self):
        super().__init__(None, None)
        self.errors = []

    def unexpectedError(self, filename, msg):
        self.errors.append(msg)

    def syntaxError(self, filename, msg, lineno, offset, text):
        self.errors.append((filename, msg, lineno, offset, text))

    def flake(self, message):
        self.errors.append(message)

default_fn = """def customFunction(self):
    print("This is a susccesful test")
    return [None, None, None, None, None, None, None, None, None]"""


def get_lexer(parent=None):
    from PyQt6.Qsci import QsciScintilla, QsciLexerPython, QsciAPIs


    class PythonLexer(QsciLexerPython):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setDefaultColor(QColor('white'))
            self.setPaper(QColor("#d3d3d3"))

    lexer = PythonLexer(parent)

    return lexer

def get_scintilla():
    from PyQt6.Qsci import QsciScintilla, QsciLexerPython, QsciAPIs
    return QsciScintilla

class PythonCodeEditor(get_scintilla()):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setAutoCompletionSource(QsciScintilla.AutoCompletionSource.AcsAll)
        # self.setAutoCompletionThreshold(1)
        #self.setAutoFillBackground(True)
        from PyQt6.Qsci import QsciScintilla, QsciLexerPython, QsciAPIs

        self.setUtf8(True)
        self.setIndentationsUseTabs(False)
        self.setIndentationWidth(4)
        self.setTabWidth(4)
        self.setIndentationGuides(True)
        self.setBraceMatching(QsciScintilla.BraceMatch.NoBraceMatch)
        self.setAutoIndent(True)
        # Setup syntax highlighter

        lexer = QsciLexerPython(self)
        lexer.setDefaultColor(QColor('white'))
        self.setCaretForegroundColor(QColor('white'))
        #self.current_editor.setLexer(lexer)
        lexer.setPaper(QColor("#1e1f22"))
        lexer.setColor(QColor('#808080'), lexer.Comment)
        lexer.setColor(QColor('#FFA500'), lexer.Keyword)
        lexer.setColor(QColor('#ffffff'), lexer.ClassName)
        lexer.setFont(QFont('Consolas'))

        self.setLexer(lexer)

        self.setPaper(QColor("#1e1f22"))

        lexer.setPaper(QColor("#1e1f22"))
        lexer.setColor(QColor('#808080'), lexer.Comment)
        lexer.setColor(QColor('#FFA500'), lexer.Keyword)
        lexer.setColor(QColor('#00000'), lexer.ClassName)
        lexer.setColor(QColor("#FFFFFF"), lexer.Default)
        lexer.setFont(QFont('Consolas'))

        self.setTabWidth(4)
        self.setMarginLineNumbers(1, True)
        self.setMarginWidth(1, "#0000")
        left_margin_index = 0
        left_margin_width = 7
        self.setMarginsForegroundColor(QColor("#FFFFFF"))
        self.setMarginsBackgroundColor(QColor("#1e1f22"))
        font_metrics = QFontMetrics(self.font())
        left_margin_width_pixels = font_metrics.horizontalAdvance(' ') * left_margin_width
        self.SendScintilla(self.SCI_SETMARGINLEFT, left_margin_index, left_margin_width_pixels)
        self.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)
        self.setMarginSensitivity(2, True)
        #self.setFoldMarginColors(QColor("#1e1f22"), QColor("#1e1f22"))
        # Customizing brace matching
        self.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)

        # Set the background color for matched braces
        self.SendScintilla(QsciScintilla.SCI_STYLESETBACK, QsciScintilla.STYLE_BRACELIGHT, QColor("#808080"))

        # Optionally, set the foreground color for matched braces
        self.SendScintilla(QsciScintilla.SCI_STYLESETFORE, QsciScintilla.STYLE_BRACELIGHT, QColor("#000000"))
        #self.setCaretLineVisible(True)
        #self.setCaretLineBackgroundColor(QColor("#80d3d3d3"))
        self.setWrapMode(QsciScintilla.WrapMode.WrapWord)
        # self.setAutoCompletionThreshold(1)
        self.setBackspaceUnindents(True)
        self.setIndentationGuides(True)

        # Setup the QsciAPIs object
        api = QsciAPIs(lexer)

        # Add custom autocompletion
        for module_name in ["math", "os", "torch", "diffusers", "PIL", "numpy"]:  # Add as many modules as you like
            try:
                module = __import__(module_name)
                for attr_name in dir(module):
                    if not attr_name.startswith("_"):  # Ignore private attributes
                        api.add(module_name + "." + attr_name)
            except ImportError:
                pass  # Module not found, skip

        # Prepare the APIs and assign them to the lexer
        api.prepare()
        lexer.setAPIs(api)

OP_NODE_VIM = get_next_opcode()
class Dialog(QDialog):
    def __init__(self, child):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.child = child
        self.layout().addWidget(self.child)

class VimWidget(QDMNodeContentWidget):
    def initUI(self):

        self.create_widgets()
        self.create_main_layout(grid=1)

        #self.grid_layout.addWidget(self.editor)
        self.grid_layout.addWidget(self.dialog)
    def create_widgets(self):



        self.editor = PythonCodeEditor(parent=self.node.grNode)
        self.dialog = Dialog(self.editor)
        #self.dialog.show()
        if self.editor.text() == "":
            self.editor.setText(default_fn)

        self.run_button = QtWidgets.QPushButton("Run")
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.show_button = QtWidgets.QPushButton("Show Editor")
        self.create_button_layout([self.run_button, self.stop_button, self.show_button])

    def serialize(self) -> dict:
        res = super().serialize()
        res["code"] = str(self.editor.text())
        return res

    def deserialize(self, data, hashmap={}, restore_id:bool=True) -> bool:
        if "code" in data:
            self.editor.setText(data["code"])
        if self.editor.text() == "":
            self.editor.setText(default_fn)
        super().deserialize(data, hashmap, restore_id)
        return True


@register_node(OP_NODE_VIM)
class VimNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/code.png"
    op_code = OP_NODE_VIM
    op_title = "CodeEditor Node"
    content_label_objname = "code_editor_node"
    category = "base/code"
    help_text = "Code Editor Node\n\n" \

    def __init__(self, scene):
        super().__init__(scene, inputs=[9,8,7,6,5,4,3,2,1], outputs=[9,8,7,6,5,4,3,2,1])
        self.interrupt = False

    def initInnerClasses(self):
        self.content = VimWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.icon = self.icon
        self.grNode.thumbnail = QtGui.QImage(self.grNode.icon).scaled(64, 64, QtCore.Qt.KeepAspectRatio)
        self.grNode.height = 800
        self.grNode.width = 1024
        self.content.setMinimumWidth(1024)
        self.content.setMinimumHeight(500)
        self.content.eval_signal.connect(self.evalImplementation)
        self.content.run_button.clicked.connect(self.start)
        self.content.stop_button.clicked.connect(self.stop)
        self.content.show_button.clicked.connect(self.content.dialog.show)

    def onDoubleClicked(self, event):
        #print(self.content.isVisible())

        self.content.editor.setVisible(not self.content.editor.isVisible())

        if self.content.editor.isVisible():
            self.grNode.height = 600
            self.grNode.width = 1024
            self.content.setMinimumWidth(1024)
            self.content.setMinimumHeight(450)
            self.content.setMaximumWidth(1024)
            self.content.setMaximumHeight(450)

            self.update_all_sockets()
        else:
            self.grNode.height = 240
            self.grNode.width = 400
            self.content.setMinimumWidth(400)
            self.content.setMinimumHeight(180)
            self.content.setMaximumWidth(400)
            self.content.setMaximumHeight(100)
            self.update_all_sockets()

    def evalImplementation_thread(self, index=0, *args, **kwargs):
        self.gs = gs
        result = [None, None, None, None, None, None, None, None, None]

        function_string = dedent(self.content.editor.text())  # Get function string from the editor

        # Parse the function string into a Python function object
        function_definition = ast.parse(function_string, mode='exec')
        exec(compile(function_definition, filename="<ast>", mode="exec"), self.__dict__)
        if hasattr(self, "customFunction"):
            try:
                result = self.customFunction(self)
            except Exception as e:
                print(repr(e))
        else:
            print("Ran Python node, but it did not contain a customFunction")
        #self.origFunction = self.customFunction  # new_function is assumed to be the name of your function

        # Call the new function
        #return self.origFunction(self, *args, **kwargs)

        return result

        # function_string = dedent(self.content.editor.text())  # Get function string from the editor
        #
        # # Parse the function string into a Python function object
        # function_definition = ast.parse(function_string, mode='exec')
        # globals_ = {}
        # exec(compile(function_definition, filename="<ast>", mode="exec"), globals_)
        # self.origFunction = globals_['customFunction']  # new_function is assumed to be the name of your function
        #
        # # Call the new function
        # return self.origFunction(self, *args, **kwargs)

    def origFunction(self):
        return True

    # def onWorkerFinished(self, result, exec=True):
    #     self.busy = False
    #     assert isinstance(result, list), "Result is not a list"
    #     assert len(result) == 9, "Please make sure to return a list of all 4 elements [data:dict, conditionings:List[Torch.tensor]], images:List[QPixmap], latents:List[Torch.tensor], even if they are None."
    #
    #     for i in result:
    #
    #         print()
    #
    #         self.setOutput(result.index(i), i)
    #
    #     #self.setOutput(0, result[0])
    #     self.executeChild(4)

    def stop(self):
        print("Interrupting Execution of Graph")
        gs.should_run = None

    def start(self):
        gs.should_run = True
        self.content.eval_signal.emit()
