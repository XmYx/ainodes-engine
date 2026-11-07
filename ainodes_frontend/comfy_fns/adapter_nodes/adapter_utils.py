"""Build an aiNodes node from a ComfyUI node class.

The old version of this file reverse-engineered a node's ``__init__`` with a
regex, lowercased every keyword argument, swallowed all exceptions, and called
the node's function with the *aiNodes* node as ``self``.  It now goes through
:mod:`comfy_schema` for the input/output description and :mod:`comfy_invoke`
for the call, so V1 and V3 nodes, async functions and ``ui`` results all work
and failures are reported instead of dropped.
"""
import gc
import os
import random
import shutil

from qtpy import QtCore
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QLineEdit,
                            QPushButton, QSpinBox, QTextEdit, QCheckBox)

from ainodes_frontend import singleton as gs
from ainodes_frontend.base import AiNode, get_next_opcode, register_node_now
from ainodes_frontend.base import device
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget

from .comfy_invoke import NodeExecutionBlocked, invoke
from .comfy_schema import describe_node

INT_MIN = -2147483647
INT_MAX = 2147483647

SEED_CONTROLS = ["fixed", "increment", "decrement", "randomize"]

#: Registered descriptions, keyed by the ComfyUI node key.
REGISTERED = {}
#: ``{node_key: reason}`` for everything that could not be adapted.
FAILED = {}


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_int(value):
    return max(INT_MIN, min(INT_MAX, _as_int(value)))


def _input_dir():
    try:
        import folder_paths

        return folder_paths.get_input_directory()
    except Exception:
        return os.path.abspath("input")


class ComfyWidgetMixin:
    """Widget construction shared by every adapted node's content widget."""

    def build_comfy_widget(self, spec):
        """Create one Qt widget for a :class:`comfy_schema.ComfyInput`."""
        name = spec.name
        params = spec.params
        label = params.get("label") or name
        tooltip = params.get("tooltip") or ""

        if spec.kind == "combo":
            choices = [str(c) for c in spec.choices] or [""]
            widget = self.create_combo_box(choices, label, accessible_name=name)
            default = params.get("default")
            if default is not None and str(default) in choices:
                widget.setCurrentText(str(default))
            if spec.wants_upload:
                self._add_upload_button(name, widget)
        elif spec.type_name == "BOOLEAN":
            widget = self.create_check_box(label, checked=bool(params.get("default", False)),
                                           accessible_name=name)
        elif spec.type_name == "INT":
            widget = self.create_spin_box(
                label,
                min_val=_clamp_int(params.get("min", INT_MIN)),
                max_val=_clamp_int(params.get("max", INT_MAX)),
                default_val=_clamp_int(params.get("default", 0)),
                step=max(1, _as_int(params.get("step", 1), 1)),
                accessible_name=name,
            )
        elif spec.type_name in ("FLOAT", "NUMBER"):
            widget = self.create_double_spin_box(
                label,
                min_val=_as_float(params.get("min", -1e9), -1e9),
                max_val=_as_float(params.get("max", 1e9), 1e9),
                step=_as_float(params.get("step", 0.01), 0.01) or 0.01,
                default_val=_as_float(params.get("default", 0.0)),
                accessible_name=name,
            )
        elif params.get("multiline"):
            widget = self.create_text_edit(label, default=str(params.get("default", "")))
        else:
            widget = self.create_line_edit(
                label,
                default=str(params.get("default", "")),
                placeholder=str(params.get("placeholder", params.get("default", ""))),
                accessible_name=name,
            )

        if tooltip and hasattr(widget, "setToolTip"):
            widget.setToolTip(str(tooltip))
        setattr(self, name, widget)

        if spec.wants_seed_control:
            control = self.create_combo_box(SEED_CONTROLS, f"{label} control",
                                            accessible_name=f"{name}_control")
            setattr(self, f"{name}_control", control)
        return widget

    def _add_upload_button(self, name, combo):
        """A real file picker for ComfyUI's ``image_upload`` combos.

        In the browser this is an upload button that POSTs to ``/upload/image``;
        here the file is copied straight into ComfyUI's input directory, which
        has the same end result without needing a server.
        """
        button = QPushButton(f"Load {name}…")

        def pick():
            path, _ = QFileDialog.getOpenFileName(
                self, f"Choose a file for '{name}'", "",
                "Media (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.avif *.mp4 *.webm "
                "*.wav *.mp3 *.flac);;All files (*)",
            )
            if not path:
                return
            target_dir = _input_dir()
            os.makedirs(target_dir, exist_ok=True)
            target = os.path.join(target_dir, os.path.basename(path))
            if os.path.abspath(path) != os.path.abspath(target):
                shutil.copy2(path, target)
            basename = os.path.basename(target)
            if combo.findText(basename) < 0:
                combo.addItem(basename)
            combo.setCurrentText(basename)
            if self.node is not None:
                self.node.markDirty(True)

        button.clicked.connect(pick)
        self.create_button_layout([button])
        setattr(self, f"{name}_upload_button", button)
        return button

    def read_comfy_values(self, specs):
        """``{input_name: value}`` straight off the widgets."""
        values = {}
        for spec in specs:
            widget = getattr(self, spec.name, None)
            if widget is None:
                continue
            values[spec.name] = self.read_widget(widget, spec)
        return values

    @staticmethod
    def read_widget(widget, spec=None):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QSpinBox):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        if isinstance(widget, QTextEdit):
            return widget.toPlainText()
        if isinstance(widget, QLineEdit):
            text = widget.text()
            if spec is not None and spec.type_name == "INT":
                return _as_int(text)
            if spec is not None and spec.type_name in ("FLOAT", "NUMBER"):
                return _as_float(text)
            return text
        getter = getattr(widget, "value", None)
        return getter() if callable(getter) else None

    @QtCore.Slot()
    def mark_node_dirty(self, value=None):
        if self.node is not None:
            self.node.markDirty(True)


def create_node(desc, js_extra=None):
    """Register an aiNodes node class for ``desc`` and return it."""
    js_extra = list(js_extra or [])
    node_op_code = get_next_opcode()
    class_name = "".join(ch for ch in desc.title.title() if ch.isalnum()) or f"Comfy{node_op_code}"
    widget_specs = desc.widgets

    input_sockets, input_names = desc.socket_inputs()
    output_sockets, output_names = desc.socket_outputs()

    node_help = desc.description or (
        f"{desc.title} - adapted from the ComfyUI node '{desc.key}'.\n"
        f"Category: {desc.category}"
    )
    if js_extra:
        node_help += ("\n\nWidgets declared in this pack's JavaScript and offered "
                      "here as plain fields: "
                      + ", ".join(w["name"] for w in js_extra))

    class Widget(QDMNodeContentWidget, ComfyWidgetMixin):
        finished = Signal()

        def initUI(self):
            for spec in widget_specs:
                try:
                    self.build_comfy_widget(spec)
                except Exception as exc:  # one bad widget must not kill the node
                    print(f"[comfy] {desc.key}: could not build widget "
                          f"'{spec.name}' ({exc})")
            for extra in js_extra:
                self._build_js_widget(extra)
            self.create_main_layout(grid=1)
            self._connect_dirty_signals()

        def _build_js_widget(self, extra):
            name = extra["name"]
            if getattr(self, name, None) is not None:
                return
            label = f"{name} (js)"
            if extra["type"] == "INT":
                widget = self.create_spin_box(label, INT_MIN, INT_MAX, 0, 1,
                                              accessible_name=name)
            elif extra["type"] == "FLOAT":
                widget = self.create_double_spin_box(label, -1e9, 1e9, 0.01, 0.0,
                                                     accessible_name=name)
            elif extra["type"] == "BOOLEAN":
                widget = self.create_check_box(label, accessible_name=name)
            else:
                widget = self.create_line_edit(label, accessible_name=name)
            widget.setToolTip(
                f"'{name}' is added by {extra['pack']} in {extra['source']}.\n"
                "aiNodes offers it as a plain field; the JavaScript behaviour "
                "behind it is not reimplemented."
            )
            setattr(self, name, widget)

        def _connect_dirty_signals(self):
            for widget in getattr(self, "widget_list", []):
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.valueChanged.connect(self.mark_node_dirty)
                elif isinstance(widget, (QLineEdit, QTextEdit)):
                    widget.textChanged.connect(self.mark_node_dirty)
                elif isinstance(widget, QComboBox):
                    widget.currentIndexChanged.connect(self.mark_node_dirty)
                elif isinstance(widget, QCheckBox):
                    widget.stateChanged.connect(self.mark_node_dirty)

    class Node(AiNode):
        icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
        help_text = node_help
        op_code = node_op_code
        op_title = desc.title
        content_label_objname = f"comfy_{desc.key.lower().replace(' ', '_')}"
        category = desc.category
        NodeContent_class = Widget
        dim = (340, 180)
        custom_input_socket_name = input_names
        custom_output_socket_name = output_names

        comfy_key = desc.key
        comfy_description = desc
        comfy_js_widgets = js_extra

        # Loaders are expensive and deterministic, so do not re-run them on
        # every graph execution; everything else re-runs.
        make_dirty = "loader" not in desc.title.lower()

        def __init__(self, scene):
            super().__init__(scene, inputs=input_sockets, outputs=output_sockets)
            self.exec_port = len(self.outputs) - 1
            self.device = gs.device
            self.last_ui = None

            modifier = max(len(input_sockets), len(output_sockets))
            # start below the title bar instead of underneath it
            title_height = getattr(self.grNode, "title_height", 24)
            self.content.setGeometry(0, title_height + 4,
                                     self.content.geometry().width(),
                                     self.content.geometry().height())
            self.grNode.height = self.content.geometry().height() + (modifier * 30)
            self.update_all_sockets()

        # ------------------------------------------------------------------
        def _gather_ports(self):
            """Port values plus the names of required ports left unconnected."""
            values = {}
            missing = []
            for index, spec in enumerate(desc.ports):
                data = self.getInputData(index)
                if data is None:
                    if spec.required:
                        missing.append(spec.name)
                    continue
                values[spec.name] = data
            return values, missing

        def _apply_seed_control(self):
            for spec in widget_specs:
                if not spec.wants_seed_control:
                    continue
                widget = getattr(self.content, spec.name, None)
                control = getattr(self.content, f"{spec.name}_control", None)
                if widget is None or control is None:
                    continue
                mode = control.currentText()
                if mode == "randomize":
                    widget.setValue(random.randint(0, INT_MAX))
                elif mode == "increment":
                    widget.setValue(min(widget.maximum(), widget.value() + 1))
                elif mode == "decrement":
                    widget.setValue(max(widget.minimum(), widget.value() - 1))

        def _hidden_values(self):
            hidden = {}
            for name, type_name in desc.hidden.items():
                if type_name == "PROMPT":
                    hidden[name] = {}
                elif type_name == "EXTRA_PNGINFO":
                    hidden[name] = {"workflow": {}}
                elif type_name == "UNIQUE_ID":
                    hidden[name] = str(self.id)
            return hidden

        def evalImplementation_thread(self):
            port_values, missing = self._gather_ports()
            if missing:
                # issue #21: a disconnected required input used to fail silently
                self.markInvalid(True)
                print(f"[{desc.title}] cannot run: required input(s) not "
                      f"connected: {', '.join(missing)}")
                self.content.finished.emit()
                return False

            kwargs = dict(port_values)
            kwargs.update(self.content.read_comfy_values(widget_specs))
            for extra in js_extra:
                widget = getattr(self.content, extra["name"], None)
                if widget is not None:
                    kwargs[extra["name"]] = self.content.read_widget(widget)
            kwargs.update(self._hidden_values())

            if self.make_dirty:
                self.markDirty(True)

            if not self.isDirty():
                self.content.update()
                self.content.finished.emit()
                return True

            try:
                values, ui = invoke(desc.node_class, kwargs,
                                    function_name=desc.function_name,
                                    expected_outputs=len(desc.outputs))
            except NodeExecutionBlocked as blocked:
                print(f"[{desc.title}] execution blocked: {blocked.message}")
                self.markInvalid(True)
                self.content.finished.emit()
                return False
            except Exception as exc:
                from ainodes_frontend.base.settings import handle_ainodes_exception

                handle_ainodes_exception()
                print(f"[{desc.title}] failed: {exc}")
                self.markInvalid(True)
                self.content.finished.emit()
                return False

            for index in range(len(desc.outputs)):
                self.setOutput(index, values[index] if index < len(values) else None)

            self.last_ui = ui
            self._apply_seed_control()
            self.markDirty(False)
            self.markInvalid(False)
            self.content.update()
            self.content.finished.emit()
            return True

        def onWorkerFinished(self, result, exec=True):
            self.busy = False
            if result:
                self.markDirty(False)
                if exec:
                    self.executeChild(self.exec_port)

        def can_run(self):
            if not self.inputs:
                return True
            for socket in self.inputs:
                for edge in socket.edges:
                    for attribute in ("start_socket", "end_socket"):
                        other = getattr(edge, attribute, None)
                        if other is not None and getattr(other, "node", None) not in (None, self):
                            if other.node.isDirty():
                                return False
            return True

        def onInputChanged(self, socket=None):
            self.markDirty(True)

        def remove(self):
            for index in range(len(self.outputs)):
                object_name = self.getID(index)
                value = self._output_values.pop(object_name, None)
                if value is None:
                    continue
                try:
                    value.cpu()
                except Exception:
                    pass
                del value
            device.empty_cache()
            gc.collect()
            super().remove()

    Node.__name__ = class_name
    Node.__qualname__ = class_name
    register_node_now(node_op_code, Node)
    REGISTERED[desc.key] = desc
    return Node


def adapt_node(key, node_class, display_name=None, category_prefix="",
               js_extra=None):
    """Describe and register one ComfyUI node.  Returns the class or ``None``."""
    if key in REGISTERED:
        return None
    try:
        desc = describe_node(key, node_class, display_name=display_name,
                             category_prefix=category_prefix)
    except Exception as exc:
        FAILED[key] = f"schema: {exc}"
        return None
    if not desc.function_name:
        FAILED[key] = "no FUNCTION attribute"
        return None
    try:
        return create_node(desc, js_extra=js_extra)
    except Exception as exc:
        FAILED[key] = f"build: {exc}"
        return None
