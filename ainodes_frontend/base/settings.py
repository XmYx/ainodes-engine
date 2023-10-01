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
    return color.name(QColor.NameFormat.HexRgb)

def hex_to_color(hex_string):
    return QColor(hex_string)

def save_settings(settings):
    settings_dict = settings.to_dict()
    with open('config/settings.yaml', 'w') as file:
        print(f"Saving settings to YAML: {settings_dict}")  # Debug line
        yaml.dump(settings_dict, file, indent=4)
class Settings:
    def __init__(self):
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
        self.use_exec = False
        self.opengl = False

    def load_from_dict(self, settings_dict):
        for key, value in settings_dict.items():
            setattr(self, key, value)

    def to_dict(self):
        if hasattr(gs, "prefs"):
            if hasattr(gs.prefs, "keybindings"):
                self.keybindings = gs.prefs.keybindings

        settings_dict = {
            'checkpoints': self.checkpoints,
            'hypernetworks': self.hypernetworks,
            'vae': self.vae,
            'controlnet': self.controlnet,
            'embeddings': self.embeddings,
            'upscalers': self.upscalers,
            'loras': self.loras,
            't2i_adapter': self.t2i_adapter,
            'output': self.output,
            'keybindings': self.keybindings if hasattr(self, 'keybindings') else {},
            'use_exec':self.use_exec,
            'opengl':self.opengl
        }

        return settings_dict
    #return settings

def load_settings(file_path=None):
    settings = Settings()
    if file_path == None:
        if os.path.exists('config/settings.yaml'):
            file_path = 'config/settings.yaml'
        else:
            file_path = 'config/default_settings.yaml'

    with open(file_path, 'r') as file:
        settings_dict = yaml.safe_load(file)
        settings.load_from_dict(settings_dict)
        save_settings(settings)
    gs.prefs = settings


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
    try:
        import xformers
        gs.system.xformer = True
    except:
        gs.system.xformer = False

    gs.current["sd_model"] = None
    gs.current["inpaint_model"] = None
    gs.loaded_vae = ""
