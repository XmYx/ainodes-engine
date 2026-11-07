#!/usr/bin/env python3
"""aiNodes Engine - application entry point.

Run ``python main.py --help`` for the full flag list.  The important ones:

    --theme dark|light|system   pick a theme (default: follow the desktop)
    --device cuda|rocm|xpu|mps|cpu
    --comfy-path PATH           the ComfyUI checkout to borrow nodes from
    --headless --run GRAPH      execute a saved graph with no window
"""
import datetime
import logging
import os
import sys

START_TIME = datetime.datetime.now()

# The app loads node packs, qss and icons by relative path, so anchor the
# process at the repo root rather than wherever the user happened to be.
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
os.chdir(APP_ROOT)

# qtpy needs to know which binding to use before anything imports it.
os.environ.setdefault("QT_API", "pyqt6")

logging.basicConfig(level=os.environ.get("AINODES_LOG_LEVEL", "ERROR"))

from ainodes_frontend import singleton as gs  # noqa: E402
from ainodes_frontend.base.args import get_args  # noqa: E402


def _banner(args, device_line, comfy_line):
    print(f"aiNodes Engine - starting {START_TIME:%Y-%m-%d %H:%M:%S}")
    print(f"  device : {device_line}")
    print(f"  comfy  : {comfy_line}")
    print(f"  theme  : {gs.theme}")


def bootstrap_comfy(args):
    """Make ComfyUI's nodes importable.  Returns a status line for the log."""
    if args.no_comfy:
        return "disabled (--no-comfy)"
    from ainodes_frontend.base import device
    from ainodes_frontend.comfy_fns import comfy_bootstrap

    try:
        path = comfy_bootstrap.ensure_comfy(
            path=args.comfy_path,
            device_backend=device.backend(),
            vram_state=getattr(gs, "vram_state", None),
            load_custom_nodes=not args.no_comfy_custom_nodes,
            load_api_nodes=args.comfy_api_nodes,
            install=args.install_comfy,
        )
    except Exception as exc:
        print(f"[comfy] {exc}")
        print("[comfy] continuing without ComfyUI nodes")
        gs.comfy_available = False
        return f"unavailable - {exc}"

    gs.comfy_available = True
    gs.comfy_path = path
    version = comfy_bootstrap.comfy_version(path) or "unknown"
    return f"{version} at {path}"


def import_node_packs(progress=None):
    """Import every ``*_nodes`` package under ``ainodes_frontend/nodes``."""
    from ainodes_frontend.base.import_utils import import_nodes_from_subdirectories

    base_folder = os.path.join("ainodes_frontend", "nodes")
    if not os.path.isdir(base_folder):
        return []

    folders = sorted(
        folder
        for folder in os.listdir(base_folder)
        if folder.endswith("_nodes")
        and "__pycache__" not in folder
        and os.path.isdir(os.path.join(base_folder, folder))
    )
    for index, folder in enumerate(folders):
        import_nodes_from_subdirectories(os.path.join(base_folder, folder))
        if progress is not None:
            progress(int((index + 1) / max(len(folders), 1) * 90))
    return folders


def import_comfy_adapters():
    """Wrap ComfyUI's NODE_CLASS_MAPPINGS as aiNodes nodes."""
    if not getattr(gs, "comfy_available", False):
        return 0
    try:
        from ainodes_frontend.comfy_fns.adapter_nodes import was_adapter_node

        return was_adapter_node.adapt_all()
    except Exception:
        from ainodes_frontend.base.settings import handle_ainodes_exception

        handle_ainodes_exception()
        return 0


def main():
    from ainodes_frontend.base import device, settings, theme

    args = get_args()
    gs.args = args

    settings.init_globals()
    device.init(args.device)
    gs.device = device.get_torch_device()
    gs.device_backend = device.backend()

    if args.local_hf:
        os.makedirs("hf_cache", exist_ok=True)
        os.environ["HF_HOME"] = os.path.abspath("hf_cache")

    if args.headless:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from qtpy import QtCore, QtGui
    from qtpy.QtWidgets import QApplication

    from ainodes_frontend.base.import_utils import set_application_attributes
    from ainodes_frontend.base.splash import SplashScreen

    if args.highdpi:
        surface = QtGui.QSurfaceFormat()
        surface.setVersion(3, 3)
        surface.setSamples(8)
        surface.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        QtGui.QSurfaceFormat.setDefaultFormat(surface)
        QtCore.QCoreApplication.setAttribute(
            QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
        )

    set_application_attributes(QApplication, args)

    app = QApplication(sys.argv)
    app.setApplicationName("aiNodes - Engine")
    app.setApplicationDisplayName("aiNodes Engine")

    settings.load_settings()

    # --theme wins; otherwise the saved preference; otherwise the desktop's.
    wanted = args.theme or getattr(gs.prefs, "theme", None) or "system"
    if args.light:  # legacy flag
        wanted = "light"
    theme.apply_theme(wanted, app=app)

    from ainodes_frontend.icon import icon

    pixmap = QtGui.QPixmap()
    pixmap.loadFromData(icon)
    app.setWindowIcon(QtGui.QIcon(pixmap))

    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "ainodes.engine.desktop.1"
            )
        except Exception:
            pass

    splash = None
    if not args.headless and not args.no_splash:
        splash = SplashScreen(os.path.join("ainodes_frontend", "qss", "icon.png"))
        splash.show()
        app.processEvents()

    def progress(value, message=None):
        if splash is not None:
            splash.set_progress(value, message)
            app.processEvents()

    progress(5, "Looking for ComfyUI")
    comfy_line = bootstrap_comfy(args)

    if args.report_js_widgets:
        from ainodes_frontend.comfy_fns import js_widgets

        if splash is not None:
            splash.close()
        import nodes

        js_widgets.report(known_node_keys=set(nodes.NODE_CLASS_MAPPINGS))
        return 0

    progress(20, "Loading node packs")
    if not args.skip_base_nodes:
        import_node_packs(lambda value: progress(20 + value * 0.4))

    progress(62, "Adapting ComfyUI nodes")
    adapted = import_comfy_adapters()
    if adapted:
        print(f"[comfy] adapted {adapted} ComfyUI nodes")

    progress(85, "Building the window")
    from ainodes_frontend.base import CalculatorWindow

    window = CalculatorWindow(app)
    window.nodesListWidget.addMyItems()

    _banner(args, device.describe(), comfy_line)

    if args.headless:
        from ainodes_frontend.base.headless import run_headless

        return run_headless(app, window, args)

    if args.screenshot:
        # A screenshot run needs a deterministic, fully laid out window.
        window.resize(1920, 1080)
    window.show()
    if not args.screenshot:
        window.showMaximized()
    window.fade_in_animation()
    if args.open:
        window.fileOpen(args.open)
    else:
        window.onFileNew()

    progress(100, "Ready")
    if splash is not None:
        splash.fade_out(window)

    report = __import__("ainodes_frontend.base.import_utils", fromlist=["x"]) \
        .import_failure_report()
    if report:
        print(report)

    print(f"Initialization took: {datetime.datetime.now() - START_TIME}")

    if args.screenshot or args.quit_after:
        _schedule_exit(app, window, args)

    return app.exec()


def _schedule_exit(app, window, args):
    """Grab a screenshot and/or quit, used by the smoke tests."""
    from qtpy.QtCore import QTimer

    delay = args.quit_after or (3.0 if args.screenshot else 0.0)

    def finish():
        # the window owns sys.stdout by now, so report on the real one
        out = sys.__stdout__
        if args.screenshot:
            path = os.path.abspath(args.screenshot)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            pixmap = window.grab()
            if pixmap.save(path):
                print(f"[screenshot] {path} ({pixmap.width()}x{pixmap.height()})",
                      file=out, flush=True)
            else:
                print(f"[screenshot] failed to write {path}", file=out, flush=True)
        app.quit()

    QTimer.singleShot(int(delay * 1000), finish)


if __name__ == "__main__":
    sys.exit(main())
