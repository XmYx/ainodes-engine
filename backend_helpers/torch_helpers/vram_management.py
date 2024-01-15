import torch

def offload_to_device(model, device, dtype=torch.float16):
    if hasattr(model, "to"):
        if next(model.parameters()).device != device:
            print("Copying model", model, "to", device)

            model.to(device)
            model.to(dtype)
    elif hasattr(model, "model"):
        if next(model.model.parameters()).device != device:
            print("Copying model", model, "to", device)
            model.model.to(device)
            model.model.to(dtype)

    elif hasattr(model, "inner_model"):
        if next(model.inner_model.parameters()).device != device:
            print("Copying model", model, "to", device)

            model.inner_model.to(device)
            model.inner_model.to(dtype)
    elif hasattr(model, "first_stage_model"):
        if next(model.first_stage_model.parameters()).device != device:
            print("Copying model", model, "to", device)

            model.first_stage_model.to(device)
            model.first_stage_model.to(dtype)
    elif hasattr(model, "cond_stage_model"):
        if next(model.cond_stage_model.parameters()).device != device:
            print("Copying model", model, "to", device)

            model.cond_stage_model.to(device)
            model.cond_stage_model.to(dtype)