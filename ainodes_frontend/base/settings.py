import os
import platform
import subprocess
import time

import yaml
import traceback

from qtpy.QtGui import QColor

from ainodes_frontend import singleton as gs

def handle_ainodes_exception():
    traceback_str = traceback.format_exc()
    gs.error_stack.append(traceback_str)
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

def save_settings():
    settings = {
        'socket_names': socket_names,
        'COLORS': [color_to_hex(color) for color in SOCKET_COLORS],
        'checkpoints': gs.checkpoints,
        'vae': gs.vae,
        'controlnet': gs.controlnet,
        'embeddings': gs.embeddings,
        'upscalers': gs.upscalers,
        'loras': gs.loras,
        't2i_adapter': gs.t2i_adapter,
        'output': gs.output,

    }
    with open('config/default_settings.yaml', 'w') as file:
        yaml.dump(settings, file, indent=4)





def load_settings():
    global SOCKET_COLORS
    global socket_names
    if os.path.exists('config/settings.yaml'):
        path = 'config/settings.yaml'
    else:
        path = 'config/default_settings.yaml'

    with open(path, 'r') as file:
        settings = yaml.safe_load(file)
        try:
            socket_names = settings['socket_names']
            SOCKET_COLORS = [hex_to_color(hex_string) for hex_string in settings['COLORS']]
            gs.checkpoints = settings['checkpoints']
            gs.vae = settings['vae']
            gs.controlnet = settings['controlnet']
            gs.embeddings = settings['embeddings']
            gs.upscalers = settings['upscalers']
            gs.loras = settings['loras']
            gs.t2i_adapter = settings['t2i_adapter']
            gs.output = settings['output']
            save_settings()
        except:
            setup_defaults()
            save_settings()

        
def setup_defaults():
    global SOCKET_COLORS
    global socket_names
    SOCKET_COLORS = [
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
    ]

    socket_names = {0: "UNUSED",
                      1: "EXEC",
                      2: "LATENT",
                      3: "COND",
                      4: "EMPTY",
                      5: "IMAGE",
                      6: "DATA"}
    gs.checkpoints = "models/checkpoints"
    gs.controlnet = "models/controlnet"
    gs.embeddings = "models/embeddings"
    gs.upscalers = 'models/upscalers'
    gs.vae = 'models/vae'
    gs.loras = "models/loras"
    gs.t2i_adapter = "models/t2i_adapter"
    gs.output = "output"
