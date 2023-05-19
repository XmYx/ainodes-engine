import glob
import os

from ainodes_frontend import singleton as gs

LISTBOX_MIMETYPE = "application/x-item"

CALC_NODES = {
}

node_categories = []

class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class OpCodeNotRegistered(ConfException): pass

# This should be the maximum opcode used in your project so far
MAX_OPCODE = 500

OP_NODE_INIT = 1


def get_next_opcode():
    """
    Finds the next available opcode in the global namespace.

    The function returns the next available opcode (an integer).
    """
    # Get a sorted list of all global variables that start with "OP_NODE_"
    op_node_vars = [name for name in globals() if name.startswith("OP_NODE_")]
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

    gs.nodes[class_reference.content_label_objname] = {}
    gs.nodes[class_reference.content_label_objname]['op_code'] = op_code
    gs.nodes[class_reference.content_label_objname]['class'] = class_reference


    #print(class_reference.content_label_objname)


def register_node(op_code):
    def decorator(original_class):
        register_node_now(op_code, original_class)
        return original_class
    return decorator

def get_class_from_opcode(op_code):
    if op_code not in CALC_NODES: raise OpCodeNotRegistered("OpCode '%d' is not registered" % op_code)

    return CALC_NODES[op_code]
def get_class_from_content_label_objname(content_label_objname):
    return gs.nodes[content_label_objname]['class']
def import_nodes_from_directory(directory):
    if "ainodes_backend" not in directory and "backend" not in directory:
        node_files = glob.glob(os.path.join(directory, "*.py"))
        for node_file in node_files:
            f = os.path.basename(node_file)
            if f != "__init__.py" and "_node" in f:
                module_name = os.path.basename(node_file)[:-3].replace('/', '.')
                dir = directory.replace('/', '.')
                dir = dir.replace('\\', '.').lstrip('.')
                exec(f"from {dir} import {module_name}")

def import_nodes_from_subdirectories(directory):
    print("importing from", directory)
    if "ainodes_backend" not in directory and "backend" not in directory:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path) and subdir != "base":
                import_nodes_from_directory(subdir_path)

def import_nodes_from_file(file_path):
    if not os.path.isfile(file_path) or not file_path.endswith('.py'):
        return

    file_dir, file_name = os.path.split(file_path)
    module_name = os.path.splitext(file_name)[0]

    root_dir = os.getcwd()
    rel_dir = os.path.relpath(file_dir, root_dir).replace(os.path.sep, '.')

    exec(f"from {rel_dir} import {module_name}")