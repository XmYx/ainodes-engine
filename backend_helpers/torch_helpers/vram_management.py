import torch

from backend_helpers.torch_helpers.torch_gc import torch_gc


def offload_to_device(model, device, dtype=torch.float16):
    if hasattr(model, "to"):
        model.to(device).to(dtype)
    elif hasattr(model, "model"):
        model.model.to(device).to(dtype)
    elif hasattr(model, "inner_model"):
        model.inner_model.to(device).to(dtype)
    elif hasattr(model, "first_stage_model"):
        model.first_stage_model.to(device).to(dtype)
    elif hasattr(model, "cond_stage_model"):
        model.cond_stage_model.to(device).to(dtype)
    torch_gc()