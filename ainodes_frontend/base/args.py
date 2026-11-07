"""Command line arguments."""
import argparse


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ainodes",
        description="aiNodes Engine - a node based desktop UI for diffusion models.",
    )

    general = parser.add_argument_group("general")
    general.add_argument("--local_hf", action="store_true",
                         help="keep the Hugging Face cache inside the app directory")
    general.add_argument("--open", metavar="GRAPH",
                         help="open this graph file on startup")
    general.add_argument("--no-splash", dest="no_splash", action="store_true",
                         help="skip the splash screen")
    general.add_argument("--skip_base_nodes", action="store_true",
                         help="do not import the bundled node packs")
    general.add_argument("--update", action="store_true", default=False,
                         help="git pull + pip install every node pack before starting")
    general.add_argument("--skip_update", action="store_true",
                         help="kept for the old launchers; updates are opt-in now")

    hardware = parser.add_argument_group("hardware")
    hardware.add_argument("--device", metavar="DEVICE", default=None,
                          help="cuda, cuda:1, rocm, xpu, mps or cpu "
                               "(default: auto-detect; also reads $AINODES_DEVICE)")
    hardware.add_argument("--cpu", action="store_const", const="cpu", dest="device",
                          help="shorthand for --device cpu")
    hardware.add_argument("--vram", choices=["low", "medium", "high"], default=None,
                          help="how aggressively to keep models resident")

    comfy = parser.add_argument_group("ComfyUI")
    comfy.add_argument("--comfy-path", dest="comfy_path", metavar="PATH", default=None,
                       help="path to a ComfyUI checkout "
                            "(default: $AINODES_COMFYUI_PATH, config/comfy.yaml, "
                            "./src/ComfyUI, ../ComfyUI)")
    comfy.add_argument("--no-comfy", dest="no_comfy", action="store_true",
                       help="start without ComfyUI nodes at all")
    comfy.add_argument("--no-comfy-custom-nodes", dest="no_comfy_custom_nodes",
                       action="store_true",
                       help="load ComfyUI's built-in nodes but skip custom_nodes/")
    comfy.add_argument("--comfy-api-nodes", dest="comfy_api_nodes", action="store_true",
                       help="also load ComfyUI's hosted API nodes")
    comfy.add_argument("--install-comfy", dest="install_comfy", action="store_true",
                       help="clone ComfyUI into src/ComfyUI if none is found")
    comfy.add_argument("--report-js-widgets", dest="report_js_widgets",
                       action="store_true",
                       help="list the widgets custom node packs declare in "
                            "JavaScript, then exit")

    appearance = parser.add_argument_group("appearance")
    appearance.add_argument("--theme", choices=["dark", "light", "system"], default=None,
                            help="colour theme (default: the saved preference, "
                                 "else the desktop's)")
    appearance.add_argument("--light", action="store_true",
                            help="shorthand for --theme light")
    appearance.add_argument("--highdpi", action="store_true",
                            help="request a multisampled OpenGL 3.3 surface")
    appearance.add_argument("--no_console", action="store_true",
                            help="hide the built-in console dock")
    appearance.add_argument("--forcewindowupdate", action="store_true",
                            help="repaint the whole window on every node update")

    headless = parser.add_argument_group("headless")
    headless.add_argument("--headless", action="store_true",
                          help="run with no visible window (uses Qt's offscreen "
                               "platform plugin); pair with --run")
    headless.add_argument("--run", metavar="GRAPH", dest="run",
                          help="execute this graph, then exit")
    headless.add_argument("--run-loops", type=int, default=1, dest="run_loops",
                          help="how many times to execute --run (default 1)")
    headless.add_argument("--screenshot", metavar="PNG", default=None,
                          help="save a screenshot of the window once it is up")
    headless.add_argument("--quit-after", dest="quit_after", type=float, default=0.0,
                          help="close the app this many seconds after startup "
                               "(used for smoke tests and screenshots)")
    headless.add_argument("--timeout", type=float, default=0.0,
                          help="give up after this many seconds (0 = no limit)")

    qt = parser.add_argument_group("Qt attributes")
    qt.add_argument("--use_opengl_es", action="store_true",
                    help="use OpenGL ES instead of desktop OpenGL")
    qt.add_argument("--enable_high_dpi_scaling", action="store_true",
                    help="enable high-DPI scaling (a no-op on Qt 6, which always does)")
    qt.add_argument("--use_high_dpi_pixmaps", action="store_true",
                    help="use high-DPI pixmaps (a no-op on Qt 6, which always does)")
    qt.add_argument("--disable_window_context_help_button", action="store_true",
                    help="drop the context-help button from window title bars")
    qt.add_argument("--use_stylesheet_propagation_in_widget_styles",
                    action="store_true",
                    help="propagate style sheets into child widgets")
    qt.add_argument("--dont_create_native_widget_siblings", action="store_true",
                    help="do not create native sibling widgets")
    qt.add_argument("--plugin_application", action="store_true",
                    help="declare this process a plugin host")
    qt.add_argument("--use_direct3d_by_default", action="store_true",
                    help="prefer Direct3D on Windows")
    qt.add_argument("--mac_plugin_application", action="store_true",
                    help="declare this process a macOS plugin host")
    qt.add_argument("--disable_shader_disk_cache", action="store_true",
                    help="do not cache compiled shaders on disk")

    # Accepted and ignored: it only ever selected a torch version.
    qt.add_argument("--torch2", action="store_true", help=argparse.SUPPRESS)

    return parser


def get_args(argv=None):
    args = build_parser().parse_args(argv)
    if args.run and not args.headless:
        args.headless = True
    return args
