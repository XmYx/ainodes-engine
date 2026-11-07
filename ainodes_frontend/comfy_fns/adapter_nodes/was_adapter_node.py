"""Adapt every node ComfyUI has loaded into an aiNodes node.

Kept under the historical file name so saved graphs that reference it still
import, but it no longer does its work at import time - :func:`adapt_all` is
called from ``main.py`` once ComfyUI is on ``sys.path``.
"""
import logging

from .adapter_utils import FAILED, REGISTERED, adapt_node

logger = logging.getLogger(__name__)

#: Node keys that are pointless or actively harmful inside aiNodes.
SKIP_KEYS = {
    # aiNodes drives execution itself, so ComfyUI's own graph plumbing nodes
    # have nothing to act on.
    "PrimitiveNode",
    "Reroute",
    "Note",
    "MarkdownNote",
}


def adapt_all(category_prefix="ComfyUI", use_js_scan=True, verbose=True):
    """Register an aiNodes node for every ComfyUI node class.

    Returns the number of nodes registered.
    """
    try:
        import nodes
    except ImportError:
        logger.warning("ComfyUI is not importable; no ComfyUI nodes adapted")
        return 0

    mappings = dict(getattr(nodes, "NODE_CLASS_MAPPINGS", {}))
    display_names = dict(getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}))

    js_map = {}
    if use_js_scan:
        try:
            from ainodes_frontend.comfy_fns import js_widgets

            js_map, js_summary, _ = js_widgets.get(known_node_keys=set(mappings))
            if verbose:
                print(f"[comfy] JS scan: {js_summary['widgets']} widget(s) on "
                      f"{js_summary['nodes_with_js_widgets']} node class(es), "
                      f"{js_summary['unattributed']} not attributable "
                      f"(see --report-js-widgets)")
        except Exception as exc:
            logger.warning("JS widget scan failed: %s", exc)

    count = 0
    for key in sorted(mappings):
        if key in SKIP_KEYS:
            continue
        node_class = mappings[key]
        if adapt_node(key, node_class,
                      display_name=display_names.get(key, key),
                      category_prefix=category_prefix,
                      js_extra=js_map.get(key)) is not None:
            count += 1

    if verbose and FAILED:
        print(f"[comfy] {len(FAILED)} node class(es) could not be adapted")
        for key, reason in sorted(FAILED.items())[:10]:
            print(f"[comfy]   {key}: {reason}")
        if len(FAILED) > 10:
            print(f"[comfy]   ... and {len(FAILED) - 10} more")
    return count


def registered():
    return dict(REGISTERED)


def failures():
    return dict(FAILED)


# Backwards compatibility: this module used to expose the sampler preview
# registry, and node packs still import it from here.
from ainodes_frontend.comfy_fns.preview import Previewer, latent_preview  # noqa: E402,F401
