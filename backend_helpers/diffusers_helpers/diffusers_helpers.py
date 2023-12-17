from enum import Enum
from typing import Union, Optional, Dict, Any, Tuple, List

import torch
from diffusers import DDIMScheduler, HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, \
    LMSDiscreteScheduler, PNDMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, \
    DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DEISMultistepScheduler

from diffusers.models.controlnet import ControlNetOutput


diffusers_models = [
    {"name": "XL BASE", "repo": "stabilityai/stable-diffusion-xl-base-1.0"},
    {"name": "XL TINY", "repo": "segmind/SDXL-Mini"},
    {"name": "XL REFINER", "repo": "stabilityai/stable-diffusion-xl-refiner-1.0"},
    {"name": "CyberRealistic", "repo": "emilianJR/CyberRealistic_V3"},
    {"name": "segmind_tiny", "repo": "segmind/tiny-sd"},
    {"name": "segmind_tiny-mxfinetune", "repo": "segmind/tiny-sd-mxfinetune"},
    {"name": "segmind_base", "repo": "segmind/small-sd"},
    {"name": "segmind_tiny_portrait", "repo": "segmind/portrait-finetuned"},
    {"name": "stable-diffusion-v1-5", "repo": "runwayml/stable-diffusion-v1-5"},
    {"name": "revAnimated", "repo": "danbrown/RevAnimated-v1-2-2"},
    {"name": "Realistic_Vision_V1.4", "repo": "SG161222/Realistic_Vision_V1.4"},
    {"name": "stable-diffusion-v1-4", "repo": "CompVis/stable-diffusion-v1-4"},
    {"name": "openjourney", "repo": "prompthero/openjourney"},
    {"name": "stable-diffusion-2-1-base", "repo": "stabilityai/stable-diffusion-2-1-base"},
    {"name": "stable-diffusion-inpainting", "repo": "runwayml/stable-diffusion-inpainting"},
    {"name": "waifu-diffusion", "repo": "hakurei/waifu-diffusion"},
    {"name": "stable-diffusion-2-1", "repo": "stabilityai/stable-diffusion-2-1"},
    {"name": "dreamlike-photoreal-2.0", "repo": "dreamlike-art/dreamlike-photoreal-2.0"},
    {"name": "anything-v3.0", "repo": "Linaqruf/anything-v3.0"},
    {"name": "DreamShaper", "repo": "Lykon/DreamShaper"},
    {"name": "dreamlike-diffusion-1.0", "repo": "dreamlike-art/dreamlike-diffusion-1.0"},
    {"name": "stable-diffusion-2", "repo": "stabilityai/stable-diffusion-2"},
    {"name": "vox2", "repo": "plasmo/vox2"},
    {"name": "openjourney-v4", "repo": "prompthero/openjourney-v4"},
    {"name": "sd-pokemon-diffusers", "repo": "lambdalabs/sd-pokemon-diffusers"},
    {"name": "Protogen_x3.4_Official_Release", "repo": "darkstorm2150/Protogen_x3.4_Official_Release"},
    {"name": "Taiyi-Stable-Diffusion-1B-Chinese-v0.1", "repo": "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"},
    {"name": "dreamlike-anime-1.0", "repo": "dreamlike-art/dreamlike-anime-1.0"},
    {"name": "Analog-Diffusion", "repo": "wavymulder/Analog-Diffusion"},
    {"name": "stable-diffusion-2-base", "repo": "stabilityai/stable-diffusion-2-base"},
    {"name": "trinart_stable_diffusion_v2", "repo": "naclbit/trinart_stable_diffusion_v2"},
    {"name": "vintedois-diffusion-v0-1", "repo": "22h/vintedois-diffusion-v0-1"},
    {"name": "stable-diffusion-v1-2", "repo": "CompVis/stable-diffusion-v1-2"},
    {"name": "Arcane-Diffusion", "repo": "nitrosocke/Arcane-Diffusion"},
    {"name": "SomethingV2_2", "repo": "NoCrypt/SomethingV2_2"},
    {"name": "EimisAnimeDiffusion_1.0v", "repo": "eimiss/EimisAnimeDiffusion_1.0v"},
    {"name": "Protogen_x5.8_Official_Release", "repo": "darkstorm2150/Protogen_x5.8_Official_Release"},
    {"name": "Nitro-Diffusion", "repo": "nitrosocke/Nitro-Diffusion"},
    {"name": "anything-midjourney-v-4-1", "repo": "Joeythemonster/anything-midjourney-v-4-1"},
    {"name": "anything-v5", "repo": "stablediffusionapi/anything-v5"},
    {"name": "portraitplus", "repo": "wavymulder/portraitplus"},
    {"name": "epic-diffusion", "repo": "johnslegers/epic-diffusion"},
    {"name": "noggles-v21-6400-best", "repo": "alxdfy/noggles-v21-6400-best"},
    {"name": "Future-Diffusion", "repo": "nitrosocke/Future-Diffusion"},
    {"name": "photorealistic-fuen-v1", "repo": "claudfuen/photorealistic-fuen-v1"},
    {"name": "Comic-Diffusion", "repo": "ogkalu/Comic-Diffusion"},
    {"name": "Ghibli-Diffusion", "repo": "nitrosocke/Ghibli-Diffusion"},
    {"name": "OrangeMixs", "repo": "WarriorMama777/OrangeMixs"},
    {"name": "children_stories_inpainting", "repo": "ducnapa/children_stories_inpainting"},
    {"name": "DucHaitenAIart", "repo": "DucHaiten/DucHaitenAIart"},
    {"name": "Dungeons-and-Diffusion", "repo": "0xJustin/Dungeons-and-Diffusion"},
    {"name": "redshift-diffusion-768", "repo": "nitrosocke/redshift-diffusion-768"},
    {"name": "Anything-V3-X", "repo": "iZELX1/Anything-V3-X"},
    {"name": "anime-kawai-diffusion", "repo": "Ojimi/anime-kawai-diffusion"},
    {"name": "midjourney-v4-diffusion", "repo": "flax/midjourney-v4-diffusion"},
    {"name": "seek.art_MEGA", "repo": "coreco/seek.art_MEGA"},
    {"name": "karlo-v1-alpha", "repo": "kakaobrain/karlo-v1-alpha"},
    {"name": "edge-of-realism", "repo": "stablediffusionapi/edge-of-realism"},
    {"name": "anything-v3-1", "repo": "cag/anything-v3-1"},
    {"name": "classic-anim-diffusion", "repo": "nitrosocke/classic-anim-diffusion"},
]

diffusers_indexed = [value["repo"] for value in diffusers_models]

def multiForward(
    self,
    sample: torch.FloatTensor,
    timestep: Union[torch.Tensor, float, int],
    encoder_hidden_states: torch.Tensor,
    controlnet_cond: List[torch.tensor],
    conditioning_scale: List[float],
    class_labels: Optional[torch.Tensor] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    guess_mode: bool = False,
    return_dict: bool = True,
) -> Union[ControlNetOutput, Tuple]:

    mid_block_res_sample = None
    down_block_res_samples = None

    for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):

        percentage = 100 - (int(timestep) / 10)
        if hasattr(controlnet, "start_control"):
            start = controlnet.start_control
        else:
            start = 0
        if hasattr(controlnet, "stop_control"):
            stop = controlnet.stop_control
        else:
            stop = 100

        if start <= percentage <= stop:
            print("DOING CNET", percentage)
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                guess_mode,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                if down_block_res_samples is not None:
                    down_block_res_samples = [
                        samples_prev + samples_curr
                        for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                    ]
                else:
                    down_block_res_samples = down_samples
                if mid_block_res_sample is not None:
                    mid_block_res_sample += mid_sample
                else:
                    mid_block_res_sample = mid_sample

    return down_block_res_samples, mid_block_res_sample


class SchedulerType(Enum):
    DDIM = "ddim"
    HEUN = "heun"
    DPM_DISCRETE = "dpm_discrete"
    DPM_ANCESTRAL = "dpm_ancestral"
    LMS = "lms"
    PNDM = "pndm"
    EULER = "euler"
    EULER_A = "euler_a"
    DPMPP_2M = "dpmpp_2m"
    DPMPP_2M_KARRAS = "dpmpp_2m_karras"
    DPMPP_2M_SDE = "dpmpp_2m_sde"
    DPMPP_2M_SDE_KARRAS = "dpmpp_2m_sde_karras"
    DPMPP_2S_A = "dpmpp_2s_a"
    DPMPP_SDE = "dpmpp_sde"
    DPMPP_SDE_KARRAS = "dpmpp_sde_karras"
    DPM2_KARRAS = "dpm2_karras"
    DPM2_A_KARRAS = "dpm2_a_karras"
    LMS_KARRAS = "lms_karras"
    DEIS_MULTISTEP = "deis_multistep"
    UNIPC_MULTISTEP = "unipc_multistep"

scheduler_type_values = [item.value for item in SchedulerType]

def get_scheduler(pipe, scheduler: SchedulerType):
    scheduler_mapping = {
        SchedulerType.DDIM: DDIMScheduler.from_config,
        SchedulerType.HEUN: HeunDiscreteScheduler.from_config,
        SchedulerType.DPM_DISCRETE: KDPM2DiscreteScheduler.from_config,
        SchedulerType.DPM_ANCESTRAL: KDPM2AncestralDiscreteScheduler.from_config,
        SchedulerType.LMS: LMSDiscreteScheduler.from_config,
        SchedulerType.PNDM: PNDMScheduler.from_config,
        SchedulerType.EULER: EulerDiscreteScheduler.from_config,
        SchedulerType.EULER_A: EulerAncestralDiscreteScheduler.from_config,
        SchedulerType.DPMPP_2M: DPMSolverMultistepScheduler.from_config,
        SchedulerType.DPMPP_2M_KARRAS: lambda config: DPMSolverMultistepScheduler.from_config(config,
                                                                                              use_karras_sigmas=True),
        SchedulerType.DPMPP_2M_SDE: lambda config: DPMSolverMultistepScheduler.from_config(config,
                                                                                           algorithm_type="sde-dpmsolver++"),
        SchedulerType.DPMPP_2M_SDE_KARRAS: lambda config: DPMSolverMultistepScheduler.from_config(config,
                                                                                                  use_karras_sigmas=True,
                                                                                                  algorithm_type="sde-dpmsolver++"),
        SchedulerType.DPMPP_2S_A: DPMSolverSinglestepScheduler.from_config,
        SchedulerType.DPMPP_SDE: DPMSolverSinglestepScheduler.from_config,
        SchedulerType.DPMPP_SDE_KARRAS: lambda config: DPMSolverSinglestepScheduler.from_config(config,
                                                                                                use_karras_sigmas=True),
        SchedulerType.DPM2_KARRAS: lambda config: KDPM2DiscreteScheduler.from_config(config, use_karras_sigmas=True),
        SchedulerType.DPM2_A_KARRAS: lambda config: KDPM2AncestralDiscreteScheduler.from_config(config,
                                                                                                use_karras_sigmas=True),
        SchedulerType.LMS_KARRAS: lambda config: LMSDiscreteScheduler.from_config(config, use_karras_sigmas=True),
        SchedulerType.UNIPC_MULTISTEP: UniPCMultistepScheduler.from_config,
        SchedulerType.DEIS_MULTISTEP: DEISMultistepScheduler.from_config
    }

    new_scheduler = scheduler_mapping[scheduler](pipe.scheduler.config)
    pipe.scheduler = new_scheduler

    return pipe

def get_scheduler_class(scheduler: SchedulerType):
    scheduler_mapping = {
        SchedulerType.DDIM: DDIMScheduler,
        SchedulerType.HEUN: HeunDiscreteScheduler,
        SchedulerType.DPM_DISCRETE: KDPM2DiscreteScheduler,
        SchedulerType.DPM_ANCESTRAL: KDPM2AncestralDiscreteScheduler,
        SchedulerType.LMS: LMSDiscreteScheduler,
        SchedulerType.PNDM: PNDMScheduler,
        SchedulerType.EULER: EulerDiscreteScheduler,
        SchedulerType.EULER_A: EulerAncestralDiscreteScheduler,
        SchedulerType.DPMPP_SDE_ANCESTRAL: DPMSolverSinglestepScheduler,
        SchedulerType.DPMPP_2M: DPMSolverMultistepScheduler
    }

    new_scheduler = scheduler_mapping[scheduler]
    return new_scheduler


