"""Normalise a ComfyUI node class into something aiNodes can build a UI from.

ComfyUI's ``INPUT_TYPES()`` is loose by design: an entry can be a port, a
widget, a combo box, a hidden runtime value, a V1 tuple or a V3 schema object,
and ``RETURN_TYPES`` entries may be plain strings or ``comfy_api`` ``IO``
StrEnum members.  Everything downstream gets much simpler if that is parsed
once, here.
"""
from dataclasses import dataclass, field

#: aiNodes socket index per ComfyUI data type.  7 is the generic "any" socket.
SOCKET_FOR_TYPE = {
    "EXEC": 1,
    "LATENT": 2,
    "CONDITIONING": 3,
    "MODEL": 4,
    "CLIP": 4,
    "VAE": 4,
    "CONTROL_NET": 4,
    "CLIP_VISION": 4,
    "STYLE_MODEL": 4,
    "GLIGEN": 4,
    "UPSCALE_MODEL": 4,
    "IMAGE": 5,
    "MASK": 5,
    "STRING": 6,
    "INT": 6,
    "FLOAT": 6,
    "BOOLEAN": 6,
    "NUMBER": 6,
    "AUDIO": 6,
    "VIDEO": 6,
}
ANY_SOCKET = 7
EXEC_SOCKET = 1

#: Types that always get a widget rather than a socket when ComfyUI gives
#: them widget parameters.
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "NUMBER"}

#: ``hidden`` input names ComfyUI fills in at execution time.
HIDDEN_NAMES = {"PROMPT", "EXTRA_PNGINFO", "UNIQUE_ID", "DYNPROMPT",
                "AUTH_TOKEN_COMFY_ORG", "API_KEY_COMFY_ORG", "NODE_DEF"}


def type_name(value):
    """``IO.CONDITIONING`` / ``'CONDITIONING'`` -> ``'CONDITIONING'``."""
    value = getattr(value, "value", value)
    return str(value).upper()


@dataclass
class ComfyInput:
    name: str
    type_name: str = ""
    params: dict = field(default_factory=dict)
    required: bool = True
    #: ``port`` (socket), ``widget`` (int/float/string/bool), ``combo``
    #: (dropdown of literal choices) or ``hidden`` (runtime value).
    kind: str = "port"
    choices: list = field(default_factory=list)

    @property
    def socket(self):
        return SOCKET_FOR_TYPE.get(self.type_name, ANY_SOCKET)

    @property
    def wants_upload(self):
        return bool(self.params.get("image_upload") or
                    self.params.get("video_upload") or
                    self.params.get("audio_upload"))

    @property
    def wants_seed_control(self):
        return bool(self.params.get("control_after_generate"))


@dataclass
class ComfyOutput:
    name: str
    type_name: str
    is_list: bool = False

    @property
    def socket(self):
        return SOCKET_FOR_TYPE.get(self.type_name, ANY_SOCKET)


@dataclass
class NodeDescription:
    key: str                     # the NODE_CLASS_MAPPINGS key
    title: str                   # display name
    category: str
    node_class: type = None
    function_name: str = ""
    inputs: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    hidden: dict = field(default_factory=dict)
    description: str = ""
    output_node: bool = False

    @property
    def ports(self):
        return [i for i in self.inputs if i.kind == "port"]

    @property
    def widgets(self):
        return [i for i in self.inputs if i.kind in ("widget", "combo")]

    def socket_inputs(self):
        """(socket index list, socket name list) including the EXEC socket."""
        sockets = [i.socket for i in self.ports]
        names = [i.name.upper() for i in self.ports]
        return sockets + [EXEC_SOCKET], names + ["EXEC"]

    def socket_outputs(self):
        sockets = [o.socket for o in self.outputs]
        names = [o.type_name for o in self.outputs]
        return sockets + [EXEC_SOCKET], names + ["EXEC"]


def _parse_param(name, spec, required, hidden=False):
    """Turn one ``INPUT_TYPES`` entry into a :class:`ComfyInput`."""
    if hidden:
        return ComfyInput(name=name, type_name=type_name(spec), kind="hidden",
                          required=False)

    params = {}
    if isinstance(spec, (list, tuple)):
        head = spec[0]
        if len(spec) > 1 and isinstance(spec[1], dict):
            params = dict(spec[1])
    else:
        head = spec

    # A list/tuple head means "pick one of these literals" - a combo box.
    if isinstance(head, (list, tuple)):
        choices = [str(getattr(c, "value", c)) for c in head]
        return ComfyInput(name=name, type_name="COMBO", params=params,
                          required=required, kind="combo", choices=choices)

    tname = type_name(head)

    # ComfyUI's V3 schema can declare a combo through options/choices instead.
    options = params.get("options") or params.get("choices")
    if tname == "COMBO" and isinstance(options, (list, tuple)):
        return ComfyInput(name=name, type_name="COMBO", params=params,
                          required=required, kind="combo",
                          choices=[str(o) for o in options])

    if tname in WIDGET_TYPES:
        # forceInput turns a widget type into a socket
        if params.get("forceInput"):
            return ComfyInput(name=name, type_name=tname, params=params,
                              required=required, kind="port")
        return ComfyInput(name=name, type_name=tname, params=params,
                          required=required, kind="widget")

    return ComfyInput(name=name, type_name=tname, params=params,
                      required=required, kind="port")


def describe_node(key, node_class, display_name=None, category_prefix=""):
    """Build a :class:`NodeDescription`, or raise if the class is unusable."""
    input_types = node_class.INPUT_TYPES()
    if not isinstance(input_types, dict):
        raise TypeError(f"{key}: INPUT_TYPES() did not return a dict")

    inputs = []
    hidden = {}
    for section in ("required", "optional"):
        for name, spec in (input_types.get(section) or {}).items():
            inputs.append(_parse_param(name, spec, required=(section == "required")))
    for name, spec in (input_types.get("hidden") or {}).items():
        hidden[name] = type_name(spec)

    return_types = getattr(node_class, "RETURN_TYPES", ()) or ()
    if isinstance(return_types, str):
        return_types = (return_types,)
    return_names = getattr(node_class, "RETURN_NAMES", None) or ()
    output_is_list = getattr(node_class, "OUTPUT_IS_LIST", None) or ()

    outputs = []
    for index, rtype in enumerate(return_types):
        tname = type_name(rtype)
        name = str(return_names[index]) if index < len(return_names) else tname
        is_list = bool(output_is_list[index]) if index < len(output_is_list) else False
        outputs.append(ComfyOutput(name=name, type_name=tname, is_list=is_list))

    category = getattr(node_class, "CATEGORY", "") or "uncategorised"
    if category_prefix:
        category = f"{category_prefix}/{category}"

    return NodeDescription(
        key=key,
        title=str(display_name or key),
        category=category,
        node_class=node_class,
        function_name=getattr(node_class, "FUNCTION", "") or "",
        inputs=inputs,
        outputs=outputs,
        hidden=hidden,
        description=(getattr(node_class, "DESCRIPTION", "") or "").strip(),
        output_node=bool(getattr(node_class, "OUTPUT_NODE", False)),
    )
