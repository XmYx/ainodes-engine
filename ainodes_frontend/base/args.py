# Parse command line arguments
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    #Local Huggingface Hub Cache
    parser.add_argument("--local_hf", action="store_true")
    parser.add_argument("--skip_base_nodes", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--skip_update", action="store_true")
    parser.add_argument("--update", action="store_true", default=False)
    parser.add_argument("--torch2", action="store_true")
    parser.add_argument("--no_console", action="store_true")
    parser.add_argument("--highdpi", action="store_true")
    parser.add_argument("--forcewindowupdate", action="store_true")
    parser.add_argument('--use_opengl_es', action='store_true',
                        help='Enables the use of OpenGL ES instead of desktop OpenGL')
    parser.add_argument('--enable_high_dpi_scaling', action='store_true',
                        help='Enables high-DPI scaling for the application')
    parser.add_argument('--use_high_dpi_pixmaps', action='store_true',
                        help='Uses high-DPI pixmaps to render images on high-resolution displays')
    parser.add_argument('--disable_window_context_help_button', action='store_true',
                        help='Disables the context-sensitive help button in the window\'s title bar')
    parser.add_argument('--use_stylesheet_propagation_in_widget_styles', action='store_true',
                        help='Enables the propagation of style sheets to child widgets')
    parser.add_argument('--dont_create_native_widget_siblings', action='store_true',
                        help='Prevents the creation of native sibling widgets for custom widgets')
    parser.add_argument('--plugin_application', action='store_true',
                        help='Specifies that the application is a plugin rather than a standalone executable')
    parser.add_argument('--use_direct3d_by_default', action='store_true',
                        help='Specifies that Direct3D should be used as the default rendering system on Windows')
    parser.add_argument('--mac_plugin_application', action='store_true',
                        help='Specifies that the application is a macOS plugin rather than a standalone executable')
    parser.add_argument('--disable_shader_disk_cache', action='store_true',
                        help='Disables the caching of compiled shader programs to disk')

    args = parser.parse_args()
    return args
