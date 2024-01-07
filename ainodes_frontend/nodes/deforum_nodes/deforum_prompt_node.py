import json
import re

from functools import partial

from qtpy import QtCore
from qtpy import QtWidgets

from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode, CalcGraphicsNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend.nodes.deforum_nodes.deforum_data_nodes import merge_dicts

#from ai_nodes.ainodes_engine_deforum_nodes.deforum_nodes.deforum_data_nodes import merge_dicts
# merge_dicts
OP_NODE_DEFORUM_PROMPT = get_next_opcode()


class DeforumPromptWidget(QDMNodeContentWidget):
    def initUI(self):
        self.createUI()

    def createUI(self):
        self.data = {}
        self.row_widgets = []

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.addButton = QtWidgets.QPushButton("Add Row")
        self.addButton.clicked.connect(self.add_row)
        self.layout.addWidget(self.addButton)

        self.loadButton = QtWidgets.QPushButton("Load Data")
        self.loadButton.clicked.connect(self.load_data)
        self.layout.addWidget(self.loadButton)

    def add_row(self):
        row_widget = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_widget)

        spinbox = QtWidgets.QSpinBox()
        spinbox.setRange(0, 10000)
        row_layout.addWidget(spinbox)

        textedit = QtWidgets.QTextEdit()
        row_layout.addWidget(textedit)

        removeButton = QtWidgets.QPushButton("Remove")
        removeButton.clicked.connect(lambda: self.remove_row(row_widget))
        row_layout.addWidget(removeButton)

        self.row_widgets.append(row_widget)
        self.layout.insertWidget(self.layout.count() - 1, row_widget)

    def serialize(self):
        res = self.get_values()
        return res

    def deserialize(self, data, hashmap={}, restore_id:bool=True):
        #self.clear_rows()


        for value, text in data.items():
            row_widget = QtWidgets.QWidget()
            row_layout = QtWidgets.QHBoxLayout(row_widget)
            spinbox = QtWidgets.QSpinBox()
            spinbox.setRange(0, 10000)
            spinbox.setValue(int(value))
            row_layout.addWidget(spinbox)

            textedit = QtWidgets.QTextEdit()
            textedit.setPlainText(text.replace("\"", ""))
            row_layout.addWidget(textedit)

            removeButton = QtWidgets.QPushButton("Remove")
            removeButton.clicked.connect(partial(self.remove_row, row_widget))
            row_layout.addWidget(removeButton)

            self.row_widgets.append(row_widget)
            self.layout.insertWidget(self.layout.count() - 1, row_widget)
        super().deserialize(data, hashmap={}, restore_id=True)

    def remove_row(self, row_widget):

        self.row_widgets.remove(row_widget)
        self.layout.removeWidget(row_widget)
        row_widget.deleteLater()

    def get_values(self):
        self.data = {}
        for i, row_widget in enumerate(self.row_widgets):
            spinbox = row_widget.layout().itemAt(0).widget()
            textedit = row_widget.layout().itemAt(1).widget()

            value = spinbox.value()
            text = textedit.toPlainText()

            # Encode the text in a JSON-friendly format
            encoded_text = json.dumps(text)

            self.data[str(value)] = encoded_text

        return self.data

    def clear_rows(self):
        for row_widget in self.row_widgets:
            self.layout.removeWidget(row_widget)
            row_widget.deleteLater()
        self.row_widgets = []

        #self.update_data()

    def get_values(self):
        self.data = {}
        for i, row_widget in enumerate(self.row_widgets):
            spinbox = row_widget.layout().itemAt(0).widget()
            textedit = row_widget.layout().itemAt(1).widget()

            value = spinbox.value()
            text = textedit.toPlainText()

            # Encode the text in a JSON-friendly format
            encoded_text = json.dumps(text)

            self.data[str(value)] = encoded_text

        return self.data

    def load_data(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Load Data")

        layout = QtWidgets.QVBoxLayout(dialog)

        textedit = QtWidgets.QTextEdit()
        layout.addWidget(textedit)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)

        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            json_data = textedit.toPlainText()
            data = json.loads(json_data)
            self.deserialize(data)


@register_node(OP_NODE_DEFORUM_PROMPT)
class DeforumPromptNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/deforum.png"
    op_code = OP_NODE_DEFORUM_PROMPT
    op_title = "Deforum Prompt Node"
    content_label_objname = "deforum_prompt_node"
    category = "base/deforum"

    dim = (600,800)

    make_dirty = True

    # output_socket_name = ["EXEC", "MODEL"]
    def __init__(self, scene):
        super().__init__(scene, inputs=[6, 1], outputs=[6, 1])

    def initInnerClasses(self):
        self.content = DeforumPromptWidget(self)
        self.grNode = CalcGraphicsNode(self)
        self.grNode.width = 400
        self.grNode.height = 800
        self.content.setMinimumWidth(400)
        self.content.setMinimumHeight(600)
        self.content.eval_signal.connect(self.evalImplementation)

    def evalImplementation_thread(self, index=0):
        self.busy = True

        input_data = self.getInputData(0)

        prompts = self.content.get_values()

        data = {"animation_prompts":prompts}

        if input_data is not None:
            data = merge_dicts(input_data, data)

        return [data]





def make_valid_json_string(json_like_string):
    # Remove whitespace and line breaks
    json_string = re.sub(r"\s", "", json_like_string)

    # Check if the string starts with '{' and ends with '}'
    if not json_string.startswith("{") or not json_string.endswith("}"):
        raise ValueError("Invalid JSON-like structure")

    # Remove outer curly braces
    json_string = json_string[1:-1]

    # Add double quotes around keys and values
    json_string = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'"\1":', json_string)
    json_string = re.sub(r":\s*([a-zA-Z_][a-zA-Z0-9_]*)", r':"\1"', json_string)

    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')

    return json_string