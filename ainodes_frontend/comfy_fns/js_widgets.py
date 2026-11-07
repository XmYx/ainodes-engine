"""Discover widgets that ComfyUI custom nodes add from JavaScript.

### What this can and cannot do

A ComfyUI node's inputs normally live in ``INPUT_TYPES()``, and aiNodes builds
a real Qt widget for each one.  Some packs instead add widgets from a web
extension (``app.registerExtension`` + ``node.addWidget(...)``) and read them
back through ``extra_pnginfo``/``widgets_values`` - that information exists
only in the browser, so there is nothing on the Python side to introspect.

This module closes part of that gap by *reading the JavaScript*: it finds
``addWidget`` / ``ComfyWidgets.X`` calls and the node name they sit next to,
and reports them.  The adapter then offers those widgets in the Qt node and
passes their values to the Python function when its signature accepts them
(or takes ``**kwargs``).

Being honest about the limits:

* it is a static scan with a proximity heuristic, so a pack that computes
  widget names at runtime, or builds them from a server round-trip, will not
  be picked up;
* minified bundles (``assets/*-HASH.js``) often survive the scan but with
  mangled identifiers;
* widgets whose *behaviour* lives in JS (custom canvases, pose editors, image
  croppers) can be surfaced as a plain value field at best - the interactive
  part is a browser widget and is not reimplemented here.

``report()`` prints exactly what was found so the gap is visible rather than
silent.
"""
import json
import os
import re
import sys

# node.addWidget("number", "strength", 1.0, cb, {...})
_ADD_WIDGET = re.compile(
    r"""addWidget\s*\(\s*["'](?P<wtype>\w+)["']\s*,\s*["'](?P<name>[^"']+)["']""",
    re.VERBOSE,
)
# ComfyWidgets.STRING(node, "text", ["STRING", {...}], app)
_COMFY_WIDGETS = re.compile(
    r"""ComfyWidgets(?:\.|\[["'])(?P<wtype>[A-Z_]+)(?:["']\])?\s*\(\s*\w+\s*,\s*["'](?P<name>[^"']+)["']"""
)
# the many ways a JS extension says "this is the node I care about"
_NODE_NAME = re.compile(
    r"""(?:nodeData|nodeType|node|this)\s*(?:\.comfyClass|\.type|\.name)?\s*"""
    r"""(?:===?|==)\s*["'](?P<name>[A-Za-z0-9_\-. |]+)["']"""
)
_NODE_NAME_LIST = re.compile(
    r"""(?:nodeNames|NODE_TYPES|targets|TARGET_NODES)\s*=\s*\[(?P<body>[^\]]{0,2000})\]"""
)

#: how far back from an addWidget call we look for a node name
PROXIMITY = 2500

WIDGET_TYPE_MAP = {
    "number": "FLOAT",
    "slider": "FLOAT",
    "int": "INT",
    "INT": "INT",
    "FLOAT": "FLOAT",
    "text": "STRING",
    "string": "STRING",
    "STRING": "STRING",
    "customtext": "STRING",
    "toggle": "BOOLEAN",
    "boolean": "BOOLEAN",
    "BOOLEAN": "BOOLEAN",
    "combo": "COMBO",
    "COMBO": "COMBO",
    "button": "BUTTON",
}

#: widget names JS packs use for bookkeeping, never for user input
IGNORED_NAMES = {
    "control_after_generate", "controlafter_generate", "$$canvas-image-preview",
    "upload", "imageupload", "preview", "placeholder",
}


def web_directories(comfy_path=None):
    """Every directory a ComfyUI extension serves JavaScript from."""
    dirs = {}
    try:
        import nodes

        for name, path in getattr(nodes, "EXTENSION_WEB_DIRS", {}).items():
            if os.path.isdir(path):
                dirs[name] = path
    except Exception:
        pass

    comfy_path = comfy_path or _comfy_path()
    if comfy_path:
        builtin = os.path.join(comfy_path, "web", "extensions")
        if os.path.isdir(builtin):
            dirs.setdefault("comfy-core", builtin)
        custom = os.path.join(comfy_path, "custom_nodes")
        if os.path.isdir(custom):
            for pack in sorted(os.listdir(custom)):
                pack_dir = os.path.join(custom, pack)
                if not os.path.isdir(pack_dir):
                    continue
                for sub in ("js", "web", "web/js", "dist", "web_version"):
                    candidate = os.path.join(pack_dir, *sub.split("/"))
                    if os.path.isdir(candidate):
                        dirs.setdefault(pack, candidate)
                        break
    return dirs


def _comfy_path():
    try:
        from ainodes_frontend.comfy_fns import comfy_bootstrap

        return comfy_bootstrap.COMFY_PATH
    except Exception:
        return None


def _js_files(root, limit=400):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", ".git")]
        for filename in filenames:
            if filename.endswith(".js") or filename.endswith(".mjs"):
                out.append(os.path.join(dirpath, filename))
                if len(out) >= limit:
                    return out
    return out


def _names_before(text, position):
    """Node names mentioned in the PROXIMITY window before ``position``."""
    window = text[max(0, position - PROXIMITY):position]
    names = [m.group("name") for m in _NODE_NAME.finditer(window)]
    for match in _NODE_NAME_LIST.finditer(window):
        names += re.findall(r"""["']([A-Za-z0-9_\-. |]+)["']""", match.group("body"))
    return names


def scan(comfy_path=None, known_node_keys=None):
    """``{node_key: [{name, type, source}]}`` plus a summary dict.

    ``known_node_keys`` filters the result to node classes that actually
    exist, which throws away most of the heuristic's false positives.
    """
    known = set(known_node_keys or ())
    found = {}
    unattributed = []
    scanned_files = 0

    for pack, directory in web_directories(comfy_path).items():
        for path in _js_files(directory):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            scanned_files += 1
            if "addWidget" not in text and "ComfyWidgets" not in text:
                continue

            for pattern in (_ADD_WIDGET, _COMFY_WIDGETS):
                for match in pattern.finditer(text):
                    name = match.group("name")
                    if name.lower() in IGNORED_NAMES:
                        continue
                    wtype = WIDGET_TYPE_MAP.get(match.group("wtype"), "STRING")
                    entry = {"name": name, "type": wtype,
                             "source": os.path.relpath(path, directory),
                             "pack": pack}
                    targets = [n for n in _names_before(text, match.start())
                               if not known or n in known]
                    if targets:
                        for target in dict.fromkeys(targets):
                            bucket = found.setdefault(target, [])
                            if not any(w["name"] == name for w in bucket):
                                bucket.append(entry)
                    else:
                        unattributed.append(entry)

    summary = {
        "files_scanned": scanned_files,
        "nodes_with_js_widgets": len(found),
        "widgets": sum(len(v) for v in found.values()),
        "unattributed": len(unattributed),
    }
    return found, summary, unattributed


_cache = None


def get(comfy_path=None, known_node_keys=None, refresh=False):
    global _cache
    if _cache is None or refresh:
        _cache = scan(comfy_path, known_node_keys)
    return _cache


def report(stream=None, comfy_path=None, known_node_keys=None):
    """Print what the scan found; used by ``--report-js-widgets``."""
    stream = stream or sys.stdout
    found, summary, unattributed = get(comfy_path, known_node_keys, refresh=True)
    print(f"[js] scanned {summary['files_scanned']} JavaScript files in "
          f"{len(web_directories(comfy_path))} web directories", file=stream)
    print(f"[js] {summary['widgets']} JS-declared widgets on "
          f"{summary['nodes_with_js_widgets']} node classes; "
          f"{summary['unattributed']} could not be tied to a node", file=stream)
    for key in sorted(found):
        widgets = ", ".join(f"{w['name']}:{w['type']}" for w in found[key])
        print(f"  {key}: {widgets}", file=stream)
    return found, summary


def dump(path, comfy_path=None, known_node_keys=None):
    found, summary, unattributed = get(comfy_path, known_node_keys, refresh=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "nodes": found,
                   "unattributed": unattributed}, fh, indent=2)
    return path
