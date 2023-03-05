LISTBOX_MIMETYPE = "application/x-item"

"""OP_NODE_INPUT = 1
OP_NODE_OUTPUT = 2
OP_NODE_ADD = 3
OP_NODE_SUB = 4
OP_NODE_MUL = 5
OP_NODE_DIV = 6"""


OP_NODE_IMG_INPUT = 1
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



CALC_NODES = {
}


class ConfException(Exception): pass
class InvalidNodeRegistration(ConfException): pass
class OpCodeNotRegistered(ConfException): pass


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



# import all nodes and register them
from nodes.image_nodes import image_op_node, input, output, video_input
from nodes.diffusers_nodes import diffusers_loader, diffusers_sampler
from nodes.exec_op_nodes import loop
from nodes.torch_nodes import torch_loader, ksampler_node, conditioning_node, controlnet_loader
from nodes.debug_nodes import debug_text_output, debug_multi_input