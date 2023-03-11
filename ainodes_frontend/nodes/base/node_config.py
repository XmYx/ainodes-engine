import glob
import os
import sys

LISTBOX_MIMETYPE = "application/x-item"

"""OP_NODE_IMG_INPUT = 1
OP_NODE_IMG_PREVIEW = 2
OP_NODE_DIFFUSERS_LOADER = 3
OP_NODE_DIFFUSERS_SAMPLER = 4
OP_NODE_DEBUG_OUTPUT = 5
OP_NODE_DEBUG_MULTI_INPUT = 6
OP_NODE_IMAGE_OPS = 7
OP_NODE_LOOP_NODE = 8
OP_NODE_TORCH_LOADER = 9
OP_NODE_CONDITIONING = 10
OP_NODE_K_SAMPLER = 11
OP_NODE_VIDEO_INPUT = 12
OP_NODE_CONTROLNET_LOADER = 13
OP_NODE_CN_APPLY = 14
OP_NODE_CONDITIONING_COMBINE = 15
OP_NODE_CONDITIONING_SET_AREA = 16
OP_NODE_LATENT = 17
OP_NODE_LATENT_COMPOSITE = 18
OP_NODE_EXEC = 19
OP_NODE_VIDEO_SAVE = 20
OP_NODE_IMAGE_BLEND = 21
OP_NODE_EXEC_SPLITTER = 22
OP_NODE_MATTE = 23
OP_NODE_DATA = 24
OP_NODE_INPAINT = 25
OP_NODE_WHISPER = 26
OP_NODE_LORA_LOADER = 27"""

CALC_NODES = {
}

node_categories = ['default', 'model', 'conditioning', 'latent', 'sampling', 'image', 'controlnet', 'debug']

class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class OpCodeNotRegistered(ConfException): pass

# This should be the maximum opcode used in your project so far
MAX_OPCODE = 9999999

OP_NODE_INIT = 1

op_node_vars = [name for name in globals() if name.startswith("OP_NODE_")]
def get_next_opcode():
    """
    Finds the next available opcode in the global namespace.

    The function returns the next available opcode (an integer).
    """
    # Get a sorted list of all global variables that start with "OP_NODE_"


    op_node_vars.sort()

    # Find the first missing opcode (i.e. the smallest integer greater than
    # MAX_OPCODE that is not already used as an opcode).
    for i in range(99):
        if f"OP_NODE_{i}" not in op_node_vars:
            globals()[f"OP_NODE_{i}"] = i
            op_node_vars.append(f"OP_NODE_{i}")
            return i

    raise RuntimeError("Could not find a free opcode.")
def register_node_now(op_code, class_reference):
    if op_code in CALC_NODES:
        raise InvalidNodeRegistration("Duplicate node registration of '%s'. There is already %s" %(
            op_code, CALC_NODES[op_code]
        ))
    CALC_NODES[op_code] = class_reference


def register_node(op_code):
    def decorator(original_class):
        register_node_now(op_code, original_class)
        return original_class
    return decorator

def get_class_from_opcode(op_code):
    if op_code not in CALC_NODES: raise OpCodeNotRegistered("OpCode '%d' is not registered" % op_code)
    return CALC_NODES[op_code]
def import_nodes_from_directory(directory):
    node_files = glob.glob(os.path.join(directory, "*.py"))
    for node_file in node_files:
        if os.path.basename(node_file) != "__init__.py":
            module_name = os.path.basename(node_file)[:-3].replace('/', '.')
            dir = directory.replace('/', '.')
            exec(f"from {dir} import {module_name}")

def import_nodes_from_subdirectories(directory):
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and subdir != "base":
            import_nodes_from_directory(subdir_path)

