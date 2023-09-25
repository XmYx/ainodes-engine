import os
import platform
import subprocess
import time
from types import SimpleNamespace

import yaml
import traceback

from qtpy.QtGui import QColor

from ainodes_frontend import singleton as gs
from ainodes_frontend.base.help import get_help
from ainodes_frontend.base.yaml_editor import DEFAULT_KEYBINDINGS


def handle_ainodes_exception():
    traceback_str = traceback.format_exc()
    gs.error_stack.append(traceback_str)
    print(traceback_str)
    save_error_log()
    return True

def save_error_log():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    today = time.strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"error_log_{today}.txt")

    machine_info = get_machine_info()

    with open(log_file, 'a') as file:  # Open in append mode
        if os.path.getsize(log_file) == 0:
            # If the log file is empty, write the machine information
            file.write(f"BEGINNING\n")
            file.write(f"Machine Information:\n{machine_info}\n\n")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Timestamp: {timestamp}\n")

        for error in gs.error_stack:
            file.write(error)
            file.write('\n---End of Error---\n')

    print(f"Error log saved at: {log_file}")


def get_machine_info():
    info = ""
    info += f"Operating System: {platform.system()} {platform.release()}\n"
    info += "PIP List:\n"

    try:
        pip_list = subprocess.check_output(['pip', 'list']).decode('utf-8')
        info += pip_list
    except Exception as e:
        info += f"Failed to retrieve PIP list: {str(e)}\n"

    try:
        env_info = subprocess.check_output(['printenv']).decode('utf-8')
        info += f"\nenvinfo:\n{env_info}"
    except Exception as e:
        info += f"Failed to retrieve envinfo: {str(e)}\n"

    return info

def color_to_hex(color):
    return color.name()

def hex_to_color(hex_string):
    return QColor(hex_string)

def save_settings(settings):
    settings_dict = settings.to_dict()
    with open('config/settings.yaml', 'w') as file:
        yaml.dump(settings_dict, file, indent=4)

class Settings:
    def __init__(self):
        # Default values
        self.socket_names = []
        self.SOCKET_COLORS = []
        self.checkpoints = "models/checkpoints"
        self.checkpoints_xl = "models/checkpoints_xl"
        self.hypernetworks = "models/hypernetworks"
        self.vae = "models/vae"
        self.controlnet = "models/controlnet"
        self.embeddings = "models/embeddings"
        self.upscalers = "models/other"
        self.loras = "models/loras"
        self.t2i_adapter = "models/t2i"
        self.output = "output"
        self.opengl = ""
        self.keybindings = DEFAULT_KEYBINDINGS

    def load_from_dict(self, settings_dict):
        for key, value in settings_dict.items():
            setattr(self, key, value)
        # Additional processing for specific settings
        if 'COLORS' in settings_dict:
            self.SOCKET_COLORS = [hex_to_color(hex_string) for hex_string in settings_dict['COLORS']]
    def to_dict(self):
        settings_dict = {
            'socket_names': self.socket_names,
            'COLORS': [color_to_hex(color) for color in self.SOCKET_COLORS],
            'checkpoints': self.checkpoints,
            'hypernetworks': self.hypernetworks,
            'vae': self.vae,
            'controlnet': self.controlnet,
            'embeddings': self.embeddings,
            'upscalers': self.upscalers,
            'loras': self.loras,
            't2i_adapter': self.t2i_adapter,
            'output': self.output,
            'keybindings': self.keybindings if hasattr(self, 'keybindings') else {}
            # Add any new settings here...
        }
        return settings_dict
    #return settings

def load_settings():
    settings = Settings()

    if os.path.exists('config/settings.yaml'):
        path = 'config/settings.yaml'
    else:
        path = 'config/default_settings.yaml'

    with open(path, 'r') as file:
        settings_dict = yaml.safe_load(file)
        # try:
        settings.load_from_dict(settings_dict)

        save_settings(settings)  # Modify your save_settings function to take a Settings object
        # except:
        #     settings_dict = setup_defaults()
        #     settings.load_from_dict(settings_dict)
        #     save_settings(settings)  # Modify your save_settings function to take a Settings object
    gs.prefs = settings

# def load_settings():
#     global SOCKET_COLORS
#     global socket_names
#     if os.path.exists('config/settings.yaml'):
#         path = 'config/settings.yaml'
#     else:
#         path = 'config/default_settings.yaml'
#
#     with open(path, 'r') as file:
#         settings = yaml.safe_load(file)
#         try:
#             socket_names = settings['socket_names']
#             SOCKET_COLORS = [hex_to_color(hex_string) for hex_string in settings['COLORS']]
#             gs.prefs.checkpoints = settings['checkpoints']
#             gs.checkpoints_xl = settings.get("checkpoints_xl", "models/checkpoints_xl")
#             gs.hypernetworks = settings['hypernetworks']
#             gs.vae = settings['vae']
#             gs.controlnet = settings['controlnet']
#             gs.prefs.embeddings = settings['embeddings']
#             gs.upscalers = settings['upscalers']
#             gs.prefs.loras = settings['loras']
#             gs.prefs.t2i_adapter = settings['t2i_adapter']
#             gs.output = settings['output']
#             gs.prefs.opengl = settings['opengl']
#             save_settings()
#         except:
#             setup_defaults()
#             save_settings()

        
def setup_defaults():
    # global SOCKET_COLORS
    # global socket_names
    return {
    "SOCKET_COLORS" : [
        QColor("#FFFF7700"),
        QColor("#FF52e220"),
        QColor("#FF0056a6"),
        QColor("#FFa86db1"),
        QColor("#FFb54747"),
        QColor("#FFdbe220"),
        QColor("#FF888888"),
        QColor("#FFFF7700"),
        QColor("#FF52e220"),
        QColor("#FF0056a6"),
        QColor("#FFa86db1"),
        QColor("#FFb54747"),
        QColor("#FFdbe220"),
        QColor("#FF888888"),
    ],
    "socket_names" : {0: "UNUSED",
                      1: "EXEC",
                      2: "LATENT",
                      3: "COND",
                      4: "PIPE/COND",
                      5: "IMAGE",
                      6: "DATA"},
    "checkpoints" : "models/checkpoints",
    "controlnet" : "models/controlnet",
    "embeddings" : "models/embeddings",
    "upscalers" : 'models/upscalers',
    "vae" : 'models/vae',
    "loras" : "models/loras",
    "t2i_adapter" : "models/t2i_adapter",
    "output" : "output",
    }


def init_globals():
    # Initialize global variables
    gs.obj = {}
    gs.values = {}
    gs.current = {}
    gs.nodes = {}
    gs.system = SimpleNamespace()
    gs.busy = False
    gs.models = {}
    gs.token = ""
    gs.use_deforum_loss = None
    gs.highlight_sockets = True
    gs.loaded_sd = ""
    gs.current = {}
    gs.loaded_vae = ""
    gs.logging = None
    gs.debug = None
    gs.hovered = None
    gs.loaded_loras = []
    gs.metas = "output/metas"
    gs.system.textual_inversion_dir = "models/embeddings"
    gs.error_stack = []
    gs.should_run = True
    gs.loaded_kandinsky = ""
    gs.loaded_hypernetworks = []
    gs.threads = {}
    #gs.help_items = get_help()
    try:
        import xformers
        gs.system.xformer = True
    except:
        gs.system.xformer = False

    gs.current["sd_model"] = None
    gs.current["inpaint_model"] = None
    gs.loaded_vae = ""
