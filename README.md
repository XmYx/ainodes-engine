# ainodes-engine

Simple Qt Node engine written in Python based on Qt Node Graph Engine

Parts of the backend, and some nodes are heavily inspired by ComfyUI,
a great web based node engine repo, please find it at:

https://github.com/comfyanonymous/ComfyUI

Installation / Running the app:

In any case, it is advised to use "git clone https://github.com/XmYx/ainodes-engine" to allow the app to self update at launch.

  Windows:
    Run run.bat
    
  Linux
    python launcher.py
    
Models go to models/checkpoints, controlnets to models/controlnet

Node List:

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
OP_NODE_CN_APPLY = 14
OP_NODE_CONDITIONING_COMBINE = 15
OP_NODE_CONDITIONING_SET_AREA = 16
OP_NODE_LATENT = 17
OP_NODE_LATENT_COMPOSITE = 18
