"""Call a ComfyUI node's Python function the way ComfyUI itself would.

ComfyUI has two node flavours in flight: the classic "V1" class with
``INPUT_TYPES`` + ``FUNCTION``, and the newer "V3" schema classes whose entry
point is a locked classmethod built by ``comfy_api.internal``.  Node functions
may also be ``async``, may return a bare tuple, a ``{"ui": ..., "result": ...}``
dict, a ``NodeOutput``, or an ``ExecutionBlocker``.

This mirrors ``execution._async_map_node_over_list`` closely enough that nodes
behave the same inside aiNodes, without aiNodes having to monkeypatch anything
in ComfyUI (which is exactly what broke the last time ComfyUI's executor was
refactored).
"""
import asyncio
import inspect
import logging

logger = logging.getLogger(__name__)

_instances = {}


class NodeExecutionBlocked(RuntimeError):
    """A node returned ``ExecutionBlocker``; downstream nodes must not run."""

    def __init__(self, message=None):
        super().__init__(message or "execution blocked by node")
        self.message = message


def _v3_internals():
    try:
        from comfy_api.internal import (_ComfyNodeInternal, _NodeOutputInternal,
                                        is_class, make_locked_method_func)

        return _ComfyNodeInternal, _NodeOutputInternal, is_class, make_locked_method_func
    except Exception:
        return None, None, None, None


def is_v3(node_class):
    internal, _, _, _ = _v3_internals()
    if internal is None:
        return False
    try:
        return isinstance(node_class, type) and issubclass(node_class, internal)
    except TypeError:
        return False


def get_instance(node_class, reuse=True):
    """A node object to call the function on.

    ComfyUI instantiates the class once per graph node and keeps it, because
    some nodes stash state (counters, output directories) on ``self``.
    """
    if not reuse:
        return node_class()
    key = id(node_class)
    instance = _instances.get(key)
    if instance is None:
        instance = node_class()
        _instances[key] = instance
    return instance


def resolve_callable(node_class, function_name=None, instance=None):
    """Return a zero-``self`` callable for ``node_class``'s entry point."""
    function_name = function_name or getattr(node_class, "FUNCTION", None)
    if not function_name:
        raise AttributeError(f"{node_class!r} has no FUNCTION")

    internal, _, is_class, make_locked_method_func = _v3_internals()
    if internal is not None and is_v3(node_class):
        node_class.VALIDATE_CLASS()
        class_clone = node_class.PREPARE_CLASS_CLONE(None)
        return make_locked_method_func(node_class, function_name, class_clone)

    instance = instance if instance is not None else get_instance(node_class)
    return getattr(instance, function_name)


def _await(result):
    if not inspect.isawaitable(result):
        return result
    from ainodes_frontend.comfy_fns import comfy_bootstrap

    loop = comfy_bootstrap._ensure_loop()
    if loop.is_running():  # pragma: no cover - aiNodes never runs the loop
        raise RuntimeError("cannot await a ComfyUI node from inside the loop")
    return loop.run_until_complete(result)


def normalise_result(result, expected_outputs=0):
    """``(values_tuple, ui_dict)`` from whatever the node handed back."""
    try:
        from comfy_execution.graph import ExecutionBlocker

        if isinstance(result, ExecutionBlocker):
            raise NodeExecutionBlocked(getattr(result, "message", None))
    except ImportError:
        pass

    ui = None

    # V3 NodeOutput
    if hasattr(result, "args") and hasattr(result, "ui") and hasattr(result, "result"):
        if getattr(result, "block_execution", None):
            raise NodeExecutionBlocked(result.block_execution)
        ui = _ui_to_dict(result.ui)
        values = tuple(result.args or ())
    elif isinstance(result, dict):
        ui = _ui_to_dict(result.get("ui"))
        values = result.get("result")
        if values is None:
            values = ()
        elif not isinstance(values, (list, tuple)):
            values = (values,)
        values = tuple(values)
    elif isinstance(result, (list, tuple)):
        values = tuple(result)
    elif result is None:
        values = ()
    else:
        values = (result,)

    if expected_outputs and len(values) < expected_outputs:
        values = values + (None,) * (expected_outputs - len(values))
    return values, ui


def _ui_to_dict(ui):
    if ui is None:
        return None
    if isinstance(ui, dict):
        return ui
    as_dict = getattr(ui, "as_dict", None)
    if callable(as_dict):
        try:
            return as_dict()
        except Exception:
            return None
    return None


def accepted_kwargs(fn):
    """Parameter names ``fn`` will accept, or ``None`` when it takes ``**kwargs``."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names = set()
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.KEYWORD_ONLY):
            names.add(name)
    names.discard("self")
    names.discard("cls")
    return names


def filter_kwargs(fn, kwargs):
    allowed = accepted_kwargs(fn)
    if allowed is None:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in allowed}


def invoke(node_class, kwargs, function_name=None, expected_outputs=0,
           drop_unknown=True):
    """Run the node and return ``(values, ui)``.

    ``drop_unknown`` throws away keyword arguments the function does not
    declare, which is what lets JS-declared widget values be offered
    optimistically without breaking nodes that never asked for them.
    """
    fn = resolve_callable(node_class, function_name)
    call_kwargs = filter_kwargs(fn, kwargs) if drop_unknown else dict(kwargs)
    dropped = set(kwargs) - set(call_kwargs)
    if dropped:
        logger.debug("%s: ignoring unsupported inputs %s",
                     getattr(node_class, "__name__", node_class), sorted(dropped))
    result = _await(fn(**call_kwargs))
    return normalise_result(result, expected_outputs)
