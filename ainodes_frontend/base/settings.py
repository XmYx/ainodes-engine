import json
import os


from ainodes_frontend import singleton as gs
from qtpy.QtGui import QColor
import yaml
def color_to_hex(color):
    return color.name()

def hex_to_color(hex_string):
    return QColor(hex_string)

def save_settings():
    print("saving")
    settings = {
        'socket_names': gs.socket_names,
        'COLORS': [color_to_hex(color) for color in gs.SOCKET_COLORS],
        'checkpoints': gs.checkpoints,
        'controlnet': gs.controlnet,
        'embeddings': gs.embeddings,
        'loras': gs.loras,
        't2i_adapter': gs.t2i_adapter,
        'output': gs.output,

    }
    print(settings)
    with open('config/settings.yaml', 'w') as file:
        yaml.dump(settings, file, indent=4)


def load_settings():
    if os.path.exists('config/settings.yaml'):
        with open('config/settings.yaml', 'r') as file:
            settings = yaml.safe_load(file)
            try:
                gs.socket_names = settings['socket_names']
                gs.SOCKET_COLORS = [hex_to_color(hex_string) for hex_string in settings['COLORS']]
                gs.checkpoints = settings['checkpoints']
                gs.controlnet = settings['controlnet']
                gs.embeddings = settings['embeddings']
                gs.loras = settings['loras']
                gs.t2i_adapter = settings['t2i_adapter']
                gs.output = settings['output']
            except:
                setup_defaults()
                save_settings()
    else:
        setup_defaults()
        save_settings()
        
def setup_defaults():
    gs.SOCKET_COLORS = [
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

    gs.socket_names = {0: "UNUSED",
                      1: "EXEC",
                      2: "LATENT",
                      3: "COND",
                      4: "EMPTY",
                      5: "IMAGE",
                      6: "DATA"}
    gs.checkpoints = "models/checkpoints"
    gs.controlnet = "models/controlnet"
    gs.embeddings = "models/embeddings"
    gs.loras = "models/loras"
    gs.t2i_adapter = "models/t2i_adapter"
    gs.output = "output"
