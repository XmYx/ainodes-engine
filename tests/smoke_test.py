#!/usr/bin/env python3
"""End-to-end smoke test: boot aiNodes offscreen and run a real graph.

    python tests/smoke_test.py            # needs a ComfyUI checkout
    python tests/smoke_test.py --no-comfy # engine only

It builds an EmptyLatentImage -> LatentUpscale chain out of *adapted ComfyUI
nodes*, executes it through aiNodes' own node runner, and checks the tensor
that comes out the far end.  No model weights are needed, so it is safe to run
anywhere, including CI.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("QT_API", "pyqt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

FAILURES = []


def check(condition, message):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)
    return condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-comfy", action="store_true")
    parser.add_argument("--comfy-path", default=None)
    options = parser.parse_args()

    from qtpy.QtWidgets import QApplication

    from ainodes_frontend import singleton as gs
    from ainodes_frontend.base import device, settings, theme
    from ainodes_frontend.base.args import get_args

    # the window reads gs.args, so give it the defaults
    gs.args = get_args([])

    print("device")
    settings.init_globals()
    device.init(None)
    gs.device = device.get_torch_device()
    check(device.backend() in ("cuda", "rocm", "xpu", "mps", "cpu"),
          f"detected backend {device.backend()} ({device.describe()})")
    check(device.autocast_dtype() is not None, "autocast dtype resolved")
    device.empty_cache()

    app = QApplication(sys.argv[:1])
    settings.load_settings()

    print("themes")
    for name in theme.available_themes():
        sheet = theme.stylesheet(name)
        check(len(sheet) > 1000, f"{name}: stylesheet renders ({len(sheet)} chars)")
        check("@" not in sheet.split("*/")[-1],
              f"{name}: no unresolved @token@ placeholders")
    check(theme.is_dark("dark") and not theme.is_dark("light"),
          "dark is dark and light is light")
    theme.apply_theme("dark", app=app)

    comfy_ready = False
    if not options.no_comfy:
        print("comfyui")
        from ainodes_frontend.comfy_fns import comfy_bootstrap

        try:
            path = comfy_bootstrap.ensure_comfy(path=options.comfy_path,
                                                load_custom_nodes=False)
            comfy_ready = check(bool(path), f"bootstrapped ComfyUI at {path}")
            import nodes as comfy_nodes

            check(len(comfy_nodes.NODE_CLASS_MAPPINGS) > 10,
                  f"{len(comfy_nodes.NODE_CLASS_MAPPINGS)} ComfyUI node classes")
        except Exception as exc:
            check(False, f"ComfyUI bootstrap failed: {exc}")

    print("node packs")
    from ainodes_frontend.base.import_utils import IMPORTED_MODULES, import_nodes_from_subdirectories

    base = os.path.join("ainodes_frontend", "nodes")
    for folder in sorted(os.listdir(base)):
        if folder.endswith("_nodes") and os.path.isdir(os.path.join(base, folder)):
            import_nodes_from_subdirectories(os.path.join(base, folder), verbose=False)
    check(len(IMPORTED_MODULES) > 5, f"{len(IMPORTED_MODULES)} node modules imported")

    adapted = 0
    if comfy_ready:
        from ainodes_frontend.comfy_fns.adapter_nodes import was_adapter_node

        adapted = was_adapter_node.adapt_all(verbose=False, use_js_scan=False)
        check(adapted > 10, f"{adapted} ComfyUI nodes adapted")

    print("window")
    from ainodes_frontend.base import CalculatorWindow

    window = CalculatorWindow(app)
    # CalculatorWindow redirects sys.stdout into its console dock, and without
    # an event loop those queued writes never arrive - put the real stdout back
    # so the rest of this test is visible.
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    window.nodesListWidget.addMyItems()
    window.resize(1280, 800)
    window.show()
    window.onFileNew()
    app.processEvents()
    editor = window.getCurrentNodeEditorWidget()
    check(editor is not None, "an editor window exists")

    if editor is not None and comfy_ready:
        print("graph execution")
        check(run_graph(app, editor), "EmptyLatentImage -> LatentUpscale produced "
                                      "a 2x4x128x128 latent")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


def run_graph(app, editor):
    """Build and execute a two-node ComfyUI graph inside aiNodes."""
    from ainodes_frontend.base.node_config import get_class_from_content_label_objname
    from ainodes_frontend.node_engine.node_edge import Edge

    empty_cls = get_class_from_content_label_objname("comfy_emptylatentimage")
    upscale_cls = get_class_from_content_label_objname("comfy_latentupscale")
    if empty_cls is None or upscale_cls is None:
        print("    (EmptyLatentImage / LatentUpscale are not registered)")
        return False

    scene = editor.scene
    source = empty_cls(scene)
    target = upscale_cls(scene)
    source.setPos(-400, 0)
    target.setPos(200, 0)

    source.content.width.setValue(512)
    source.content.height.setValue(512)
    source.content.batch_size.setValue(2)
    target.content.width.setValue(1024)
    target.content.height.setValue(1024)

    # LATENT is the first declared port on LatentUpscale, so socket index 0.
    Edge(scene, source.outputs[0], target.inputs[0], edge_type=2)

    source.markDirty(True)
    target.markDirty(True)
    source.evalImplementation_thread()
    app.processEvents()
    target.evalImplementation_thread()
    app.processEvents()

    result = target.getOutput(0)
    if not isinstance(result, dict) or "samples" not in result:
        print(f"    (unexpected output: {result!r})")
        return False
    shape = tuple(result["samples"].shape)
    print(f"    latent shape {shape}")
    return shape == (2, 4, 128, 128)


if __name__ == "__main__":
    sys.exit(main())
