import math
from types import SimpleNamespace

import numexpr
import numpy as np
import pandas as pd

from deforum import ImageRNGNoise
from deforum.pipeline_utils import next_seed
from deforum.pipelines.deforum_animation.animation_params import RootArgs, DeforumArgs, DeforumAnimArgs, \
    DeforumOutputArgs, ParseqArgs, LoopArgs
from deforum.utils.string_utils import split_weighted_subprompts
from src.deforum.src.deforum.pipelines.deforum_animation.animation_helpers import DeformAnimKeys
#
# DeformAnimKeys
# RootArgs
# ParseqAnimKeys
# process_args, RootArgs, DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, ParseqArgs, LoopArgs
import os
import secrets
import torch
from PIL import Image
# from deforum.datafunctions.prompt import split_weighted_subprompts
# from deforum.datafunctions.seed import next_seed
from qtpy import QtCore, QtWidgets
# split_weighted_subprompts
# next_seed
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend import pil2tensor
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.pipelines.pipeline_t2v_base_pixel import tensor2vid
# from ai_nodes.ainodes_engine_base_nodes.ainodes_backend.showone.showone_model import VideoGenerator
from ainodes_frontend.base import register_node, get_next_opcode
from ainodes_frontend.base import AiNode
from ainodes_frontend.node_engine.node_content_widget import QDMNodeContentWidget
from ainodes_frontend import singleton as gs

OP_NODE_DEFORUM_ITERATE = get_next_opcode()


class DeforumIterateWidget(QDMNodeContentWidget):

    set_frame_signal = QtCore.Signal(int)

    def initUI(self):
        self.reset_iteration = QtWidgets.QPushButton("Reset Frame Counter")
        self.frame_counter = self.create_label("Current Frame: 0")
        self.create_slider("Timeline", min_val=0, max_val=500, default_val=0, step=1, spawn="frame_slider")
        self.create_button_layout([self.reset_iteration])
        self.create_main_layout(grid=1)

    def set_frame_counter(self, number):
        self.frame_counter.setText(f"Current Frame: {number}")
        self.node.set_frame(number)

@register_node(OP_NODE_DEFORUM_ITERATE)
class DeforumIterateNode(AiNode):
    icon = "ainodes_frontend/icons/base_nodes/v2/experimental.png"
    help_text = "Deforum Iterator"
    op_code = OP_NODE_DEFORUM_ITERATE
    op_title = "Deforum Iterator"
    content_label_objname = "deforum_iterate_node"
    category = "aiNodes Deforum/DeForum"
    NodeContent_class = DeforumIterateWidget
    dim = (460, 160)

    make_dirty = True

    def __init__(self, scene):
        super().__init__(scene, inputs=[6,2,1], outputs=[6,2,1])
        self.frame_index = 0
        self.content.set_frame_signal.connect(self.content.set_frame_counter)
        self.content.reset_iteration.clicked.connect(self.reset_iteration)
        self.seed = ""
        self.seeds = []
        self.args = None
        self.root = None
        self.content.frame_slider.valueChanged.connect(self.set_frame)

    def reset_iteration(self):
        self.frame_index = 0

        self.seed = ""
        self.seeds = []
        self.args = None
        self.root = None

        self.content.set_frame_signal.emit(0)

        for node in self.scene.nodes:
            if hasattr(node, "clearOutputs"):
                node.clearOutputs()
            if hasattr(node, "markDirty"):
                node.markDirty(True)

    def set_frame(self, number):
        self.frame_index = number
        self.content.frame_slider.setValue(number)
        if self.args:
            for i in range(self.frame_index):
                if i >= len(self.seeds):
                    self.seed = next_seed(self.args, self.root)
                    self.seeds.append(self.seed)


    def evalImplementation_thread(self, index=0):



        data = self.getInputData(0)

        root_dict = RootArgs()
        args_dict = {key: value["value"] for key, value in DeforumArgs().items()}
        anim_args_dict = {key: value["value"] for key, value in DeforumAnimArgs().items()}
        output_args_dict = {key: value["value"] for key, value in DeforumOutputArgs().items()}
        loop_args_dict = {key: value["value"] for key, value in LoopArgs().items()}
        root = SimpleNamespace(**root_dict)
        args = SimpleNamespace(**args_dict)
        anim_args = SimpleNamespace(**anim_args_dict)
        video_args = SimpleNamespace(**output_args_dict)
        parseq_args = None
        loop_args = SimpleNamespace(**loop_args_dict)
        controlnet_args = SimpleNamespace(**{"controlnet_args": "None"})

        for key, value in args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(args, key, val)

        for key, value in anim_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in video_args.__dict__.items():
            if key in data:
                if data[key] == "" and "schedule" not in key:
                    val = None
                else:
                    val = data[key]
                setattr(anim_args, key, val)

        for key, value in root.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(root, key, val)

        for key, value in loop_args.__dict__.items():
            if key in data:
                if data[key] == "":
                    val = None
                else:
                    val = data[key]
                setattr(loop_args, key, val)
        #print(anim_args.max_frames)
        keys, prompt_series = get_current_keys(anim_args, args.seed, root)
        # print(f"WOULD RETURN\n{keys}\n\n{prompt_series}")

        if self.frame_index > anim_args.max_frames:
            self.reset_iteration()
            self.frame_index = 0
            gs.should_run = False
            return [None]
        else:
            args.scale = keys.cfg_scale_schedule_series[self.frame_index]
            args.prompt = prompt_series[self.frame_index]

            args.seed = int(args.seed)
            root.seed_internal = int(root.seed_internal)
            args.seed_iter_N = int(args.seed_iter_N)

            if self.seed == "":
                self.seed = args.seed
                self.seed_internal = root.seed_internal
                self.seed_iter_N = args.seed_iter_N

            self.seed = next_seed(args, root)
            args.seed = self.seed
            self.seeds.append(self.seed)

            blend_value = 0.0

            # print(frame, anim_args.diffusion_cadence, node.deforum.prompt_series)

            next_frame = self.frame_index + anim_args.diffusion_cadence
            next_prompt = None

            def generate_blend_values(distance_to_next_prompt, blend_type="linear"):
                if blend_type == "linear":
                    return [i / distance_to_next_prompt for i in range(distance_to_next_prompt + 1)]
                elif blend_type == "exponential":
                    base = 2
                    return [1 / (1 + math.exp(-8 * (i / distance_to_next_prompt - 0.5))) for i in
                            range(distance_to_next_prompt + 1)]
                else:
                    raise ValueError(f"Unknown blend type: {blend_type}")

            def find_last_prompt_change(current_index, prompt_series):
                # Step backward from the current position
                for i in range(current_index - 1, -1, -1):
                    if prompt_series[i] != prompt_series[current_index]:
                        return i
                return 0  # default to the start if no change found

            def find_next_prompt_change(current_index, prompt_series):
                # Step forward from the current position
                for i in range(current_index + 1, len(prompt_series)):
                    if prompt_series[i] != prompt_series[current_index]:
                        return i
                return len(prompt_series) - 1  # default to the end if no change found

            # Inside your main loop:

            last_prompt_change = find_last_prompt_change(self.frame_index, prompt_series)
            next_prompt_change = find_next_prompt_change(self.frame_index, prompt_series)

            distance_between_changes = next_prompt_change - last_prompt_change
            current_distance_from_last = self.frame_index - last_prompt_change

            # Generate blend values for the distance between prompt changes
            blend_values = generate_blend_values(distance_between_changes, blend_type="exponential")

            # Fetch the blend value based on the current frame's distance from the last prompt change
            blend_value = blend_values[current_distance_from_last]
            next_prompt = prompt_series[next_prompt_change]

            gen_args = self.get_current_frame(args, anim_args, root, keys, self.frame_index)

            self.content.frame_slider.setMaximum(anim_args.max_frames - 1)

            self.args = args
            self.root = root
            gen_args["next_prompt"] = next_prompt
            gen_args["prompt_blend"] = blend_value
            if self.frame_index == 0:
                self.rng = ImageRNGNoise((4, args.H // 8, args.W // 8), [self.seed], [self.seed - 1],
                                    0.6, 1024, 1024)
                latent = {"samples":self.rng.first().half()}
            else:

                latent = self.getInputData(1)
                #latent = self.rng.next().half()
            print(f"[ Deforum Iterator: {self.frame_index} / {anim_args.max_frames} {self.seed}]")
            self.frame_index += 1
            self.content.set_frame_signal.emit(self.frame_index)
            #print(latent)
            #print(f"[ Current Seed List: ]\n[ {self.seeds} ]")
            return [gen_args, latent]

    def get_current_frame(self, args, anim_args, root, keys, frame_idx):
        prompt, negative_prompt = split_weighted_subprompts(args.prompt, frame_idx, anim_args.max_frames)
        strength = keys.strength_schedule_series[frame_idx] if not frame_idx == 0 or args.use_init else 1.0

        return {"prompt":prompt,
                "negative_prompt":negative_prompt,
                "strength":strength,
                "args":args,
                "anim_args":anim_args,
                "root":root,
                "keys":keys,
                "frame_idx":frame_idx}



def get_current_keys(anim_args, seed, root, parseq_args=None, video_args=None):

    use_parseq = False if parseq_args == None else True
    anim_args.max_frames += 2
    keys = DeformAnimKeys(anim_args, seed) # if not use_parseq else ParseqAnimKeys(parseq_args, video_args)

    # Always enable pseudo-3d with parseq. No need for an extra toggle:
    # Whether it's used or not in practice is defined by the schedules
    if use_parseq:
        anim_args.flip_2d_perspective = True
        # expand prompts out to per-frame
    if use_parseq and keys.manages_prompts():
        prompt_series = keys.prompts
    else:
        prompt_series = pd.Series([np.nan for a in range(anim_args.max_frames + 1)])
        for i, prompt in root.animation_prompts.items():
            if str(i).isdigit():
                prompt_series[int(i)] = prompt
            else:
                prompt_series[int(numexpr.evaluate(i))] = prompt
        prompt_series = prompt_series.ffill().bfill()
    prompt_series = prompt_series
    anim_args.max_frames -= 2
    return keys, prompt_series
