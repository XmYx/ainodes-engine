import torch as th
import torch.nn as nn
import copy

from comfy.ldm.modules.diffusionmodules.openaimodel import UNetModel,forward_timestep_embed,apply_control
from comfy.ldm.modules.diffusionmodules.util import timestep_embedding

origin_forward_timestep_embed = forward_timestep_embed

class ForwardTimestepEmbedModule(th.nn.Module):
    def __init__(self, ts, transformer_options={}, num_video_frames=None):
        super().__init__()
        self.module = ts
        self.transformer_options = transformer_options
        self.num_video_frames = num_video_frames

    def forward(
        self,
        x,
        emb,
        context=None,
        output_shape_tensor=None,
        time_context=None,
        image_only_indicator=None,
    ):
        return origin_forward_timestep_embed(
            self.module,
            x,
            emb,
            context=context,
            transformer_options=self.transformer_options,
            output_shape=output_shape_tensor
            if output_shape_tensor == None
            else output_shape_tensor.shape,
            time_context=time_context,
            num_video_frames=self.num_video_frames,
            image_only_indicator=image_only_indicator,
        )


class PatchUNetModel(UNetModel):

    @staticmethod
    def cast_from(other):
        tcls = UNetModel
        if isinstance(other, tcls):
            other.__class__ = PatchUNetModel
            other.patch_init()
            return other
        raise ValueError(f"instance must be {tcls.__qualname__}")

    def cast_to_base_model(self):
        self.patch_deinit()
        self.__class__ = UNetModel
        return self

    def patch_init(self):
        self.input_block_patch = nn.ModuleList([nn.ModuleList() for _ in self.input_blocks])
        self.input_block_patch_after_skip = nn.ModuleList([nn.ModuleList() for _ in self.input_blocks])
        self.output_block_patch = nn.ModuleList([nn.ModuleList() for _ in self.output_blocks])

    def patch_deinit(self):
        del self.input_block_patch 
        del self.input_block_patch_after_skip
        del self.output_block_patch

    def set_patch_module(self, patch_module):
        if "input_block_patch" in patch_module:
            self.input_block_patch = nn.ModuleList([nn.ModuleList(copy.deepcopy(patch_module["input_block_patch"])) for _ in self.input_blocks])
        if "input_block_patch_after_skip" in patch_module:
            self.input_block_patch_after_skip = nn.ModuleList([nn.ModuleList(copy.deepcopy(patch_module["input_block_patch_after_skip"])) for _ in self.input_blocks])
        if "output_block_patch" in patch_module:
            self.output_block_patch = nn.ModuleList([nn.ModuleList(copy.deepcopy(patch_module["output_block_patch"])) for _ in self.output_blocks])

    def forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        transformer_options["original_shape"] = list(x.shape)
        transformer_options["current_index"] = 0
        transformer_patches = transformer_options.get("patches", {})

        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator = kwargs.get("image_only_indicator", self.default_image_only_indicator)
        time_context = kwargs.get("time_context", None)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(self.dtype)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed(module, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control(h, control, 'input')

            for patch_id ,input_block_patch_module in enumerate(self.input_block_patch[id]):
                h = input_block_patch_module(h, transformer_patches.get("input_block_patch")[patch_id], transformer_options)

            hs.append(h)

            for patch_id ,input_block_patch_after_skip_module in enumerate(self.input_block_patch_after_skip[id]):
                h = input_block_patch_after_skip_module(h, transformer_patches.get("input_block_patch_after_skip")[patch_id], transformer_options)

        transformer_options["block"] = ("middle", 0)
        h = forward_timestep_embed(self.middle_block, h, emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control(h, control, 'middle')


        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = hs.pop()
            hsp = apply_control(hsp, control, 'output')

            for patch_id ,output_block_patch_module in enumerate(self.output_block_patch[id]):
                h, hsp = output_block_patch_module(h, hsp, transformer_patches.get("output_block_patch")[patch_id], transformer_options)

            h = th.cat([h, hsp], dim=1)
            del hsp
            if len(hs) > 0:
                output_shape = hs[-1].shape
            else:
                output_shape = None
            h = forward_timestep_embed(module, h, emb, context, transformer_options, output_shape, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)