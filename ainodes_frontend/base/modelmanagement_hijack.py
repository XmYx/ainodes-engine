import psutil
from enum import Enum
from comfy.cli_args import args
import comfy.utils
import torch
import sys

class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0

lowvram_available = True
xpu_available = False

directml_enabled = False
if args.directml is not None:
    import torch_directml
    directml_enabled = True
    device_index = args.directml
    if device_index < 0:
        directml_device = torch_directml.device()
    else:
        directml_device = torch_directml.device(device_index)
    print("Using directml with device:", torch_directml.device_name(device_index))
    # torch_directml.disable_tiled_resources(True)
    lowvram_available = False #TODO: need to find a way to get free memory in directml before this can be enabled by default.

try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass

try:
    if torch.backends.mps.is_available():
        cpu_state = CPUState.MPS
        import torch.mps
except:
    pass

if args.cpu:
    cpu_state = CPUState.CPU

def is_intel_xpu():
    global cpu_state
    global xpu_available
    if cpu_state == CPUState.GPU:
        if xpu_available:
            return True
    return False

def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu")
        else:
            return torch.device(torch.cuda.current_device())

def get_total_memory(dev=None, torch_total_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_total = psutil.virtual_memory().total
        mem_total_torch = mem_total
    else:
        if directml_enabled:
            mem_total = 1024 * 1024 * 1024 #TODO
            mem_total_torch = mem_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            mem_total = torch.xpu.get_device_properties(dev).total_memory
            mem_total_torch = mem_reserved
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_reserved = stats['reserved_bytes.all.current']
            _, mem_total_cuda = torch.cuda.mem_get_info(dev)
            mem_total_torch = mem_reserved
            mem_total = mem_total_cuda

    if torch_total_too:
        return (mem_total, mem_total_torch)
    else:
        return mem_total

# total_vram = get_total_memory(get_torch_device()) / (1024 * 1024)
# total_ram = psutil.virtual_memory().total / (1024 * 1024)
# print("Total VRAM {:0.0f} MB, total RAM {:0.0f} MB".format(total_vram, total_ram))
# if not args.normalvram and not args.cpu:
#     if lowvram_available and total_vram <= 4096:
#         print("Trying to enable lowvram mode because your GPU seems to have 4GB or less. If you don't want this use: --normalvram")
#         set_vram_to = VRAMState.LOW_VRAM

try:
    OOM_EXCEPTION = torch.cuda.OutOfMemoryError
except:
    OOM_EXCEPTION = Exception

XFORMERS_VERSION = ""
XFORMERS_ENABLED_VAE = True
if args.disable_xformers:
    XFORMERS_IS_AVAILABLE = False
else:
    try:
        import xformers
        import xformers.ops
        XFORMERS_IS_AVAILABLE = True
        # try:
        #     XFORMERS_VERSION = xformers.version.__version__
        #     print("xformers version:", XFORMERS_VERSION)
        #     if XFORMERS_VERSION.startswith("0.0.18"):
        #         print()
        #         print("WARNING: This version of xformers has a major bug where you will get black images when generating high resolution images.")
        #         print("Please downgrade or upgrade xformers to a different version.")
        #         print()
        #         XFORMERS_ENABLED_VAE = False
        # except:
        #     pass
    except:
        XFORMERS_IS_AVAILABLE = False

def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False

ENABLE_PYTORCH_ATTENTION = args.use_pytorch_cross_attention
VAE_DTYPE = torch.float32

try:
    if is_nvidia():
        torch_version = torch.version.__version__
        if int(torch_version[0]) >= 2:
            if ENABLE_PYTORCH_ATTENTION == False and XFORMERS_IS_AVAILABLE == False and args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
                ENABLE_PYTORCH_ATTENTION = True
            if torch.cuda.is_bf16_supported():
                VAE_DTYPE = torch.bfloat16
    if is_intel_xpu():
        if args.use_split_cross_attention == False and args.use_quad_cross_attention == False:
            ENABLE_PYTORCH_ATTENTION = True
except:
    pass

if is_intel_xpu():
    VAE_DTYPE = torch.bfloat16

if args.fp16_vae:
    VAE_DTYPE = torch.float16
elif args.bf16_vae:
    VAE_DTYPE = torch.bfloat16
elif args.fp32_vae:
    VAE_DTYPE = torch.float32


if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    XFORMERS_IS_AVAILABLE = False

if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
FORCE_FP16 = False
if args.force_fp32:
    print("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if args.force_fp16:
    print("Forcing FP16.")
    FORCE_FP16 = True

if lowvram_available:
    try:
        import accelerate
        if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
            vram_state = set_vram_to
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print("ERROR: LOW VRAM MODE NEEDS accelerate.")
        lowvram_available = False


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

#print(f"Set vram state to: {vram_state.name}")

DISABLE_SMART_MEMORY = True

# if DISABLE_SMART_MEMORY:
#     print("Disabling smart memory management")

def get_torch_device_name(device):
    if hasattr(device, 'type'):
        if device.type == "cuda":
            try:
                allocator_backend = torch.cuda.get_allocator_backend()
            except:
                allocator_backend = ""
            return "{} {} : {}".format(device, torch.cuda.get_device_name(device), allocator_backend)
        else:
            return "{}".format(device.type)
    elif is_intel_xpu():
        return "{} {}".format(device, torch.xpu.get_device_name(device))
    else:
        return "CUDA {}: {}".format(device, torch.cuda.get_device_name(device))

# try:
#     print("Device:", get_torch_device_name(get_torch_device()))
# except:
#     print("Could not pick default device.")

#print("VAE dtype:", VAE_DTYPE)

current_loaded_models = []

class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.model_accelerated = False
        self.device = model.load_device

    def model_memory(self):
        return self.model.model_size()

    def model_memory_required(self, device):
        if device == self.model.current_device:
            return 0
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0):
        patch_model_to = None
        if lowvram_model_memory == 0:
            patch_model_to = self.device

        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        try:
            self.real_model = self.model.patch_model(device_to=patch_model_to) #TODO: do something with loras and offloading to CPU
        except Exception as e:
            self.model.unpatch_model(self.model.offload_device)
            self.model_unload()
            raise e

        if lowvram_model_memory > 0:
            print("loading in lowvram mode", lowvram_model_memory/(1024 * 1024))
            device_map = accelerate.infer_auto_device_map(self.real_model, max_memory={0: "{}MiB".format(lowvram_model_memory // (1024 * 1024)), "cpu": "16GiB"})
            accelerate.dispatch_model(self.real_model, device_map=device_map, main_device=self.device)
            self.model_accelerated = True

        if is_intel_xpu() and not args.disable_ipex_optimize:
            self.real_model = torch.xpu.optimize(self.real_model.eval(), inplace=True, auto_kernel_selection=True, graph_mode=True)

        return self.real_model

    def model_unload(self):
        if self.model_accelerated:
            accelerate.hooks.remove_hook_from_submodules(self.real_model)
            self.model_accelerated = False

        self.model.unpatch_model(self.model.offload_device)
        self.model.model_patches_to(self.model.offload_device)

    def __eq__(self, other):
        return self.model is other.model

def minimum_inference_memory():
    return (1024 * 1024 * 1024)

def unload_model_clones(model):
    to_unload = []
    for i in range(len(current_loaded_models)):
        if model.is_clone(current_loaded_models[i].model):
            to_unload = [i] + to_unload

    for i in to_unload:
        print("unload clone", i)
        current_loaded_models.pop(i).model_unload()

def free_memory(memory_required, device, keep_loaded=[]):
    unloaded_model = False
    for i in range(len(current_loaded_models) -1, -1, -1):
        if not DISABLE_SMART_MEMORY:
            if get_free_memory(device) > memory_required:
                break
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                m = current_loaded_models.pop(i)
                m.model_unload()
                del m
                unloaded_model = True

    if unloaded_model:
        soft_empty_cache()


def load_models_gpu(models, memory_required=0):
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required)

    models_to_load = []
    models_already_loaded = []
    for x in models:
        loaded_model = LoadedModel(x)

        if loaded_model in current_loaded_models:
            index = current_loaded_models.index(loaded_model)
            current_loaded_models.insert(0, current_loaded_models.pop(index))
            models_already_loaded.append(loaded_model)
        else:
            models_to_load.append(loaded_model)

    if len(models_to_load) == 0:
        devs = set(map(lambda a: a.device, models_already_loaded))
        for d in devs:
            if d != torch.device("cpu"):
                free_memory(extra_mem, d, models_already_loaded)
        return

    print("loading new")

    total_memory_required = {}
    for loaded_model in models_to_load:
        unload_model_clones(loaded_model.model)
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.3 + extra_mem, device, models_already_loaded)

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM):
            model_size = loaded_model.model_memory_required(torch_dev)
            current_free_mem = get_free_memory(torch_dev)
            lowvram_model_memory = int(max(256 * (1024 * 1024), (current_free_mem - 1024 * (1024 * 1024)) / 1.3 ))
            if model_size > (current_free_mem - inference_memory): #only switch to lowvram if really necessary
                vram_set_state = VRAMState.LOW_VRAM
            else:
                lowvram_model_memory = 0

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 256 * 1024 * 1024

        cur_loaded_model = loaded_model.model_load(lowvram_model_memory)
        current_loaded_models.insert(0, loaded_model)
    return


def load_model_gpu(model):
    return load_models_gpu([model])

def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        print(sys.getrefcount(current_loaded_models[i].model))
        if sys.getrefcount(current_loaded_models[i].model) <= 2:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        x.model_unload()
        del x

def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    return dtype_size

def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev

def text_encoder_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def text_encoder_device():
    if args.gpu_only:
        return get_torch_device()
    elif vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.NORMAL_VRAM:
        if is_intel_xpu():
            return torch.device("cpu")
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
        else:
            return torch.device("cpu")
    else:
        return torch.device("cpu")

def vae_device():
    return get_torch_device()

def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_dtype():
    global VAE_DTYPE
    return VAE_DTYPE

def get_autocast_device(dev):
    if hasattr(dev, 'type'):
        return dev.type
    return "cuda"

def cast_to_device(tensor, device, dtype, copy=False):
    device_supports_cast = False
    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        device_supports_cast = True
    elif tensor.dtype == torch.bfloat16:
        if hasattr(device, 'type') and device.type.startswith("cuda"):
            device_supports_cast = True
        elif is_intel_xpu():
            device_supports_cast = True

    if device_supports_cast:
        if copy:
            if tensor.device == device:
                return tensor.to(dtype, copy=copy)
            return tensor.to(device, copy=copy).to(dtype)
        else:
            return tensor.to(device).to(dtype)
    else:
        return tensor.to(dtype).to(device, copy=copy)

def xformers_enabled():
    global directml_enabled
    global cpu_state
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu():
        return False
    if directml_enabled:
        return False
    return XFORMERS_IS_AVAILABLE


def xformers_enabled_vae():
    enabled = xformers_enabled()
    if not enabled:
        return False

    return XFORMERS_ENABLED_VAE

def pytorch_attention_enabled():
    global ENABLE_PYTORCH_ATTENTION
    return ENABLE_PYTORCH_ATTENTION

def pytorch_attention_flash_attention():
    global ENABLE_PYTORCH_ATTENTION
    if ENABLE_PYTORCH_ATTENTION:
        #TODO: more reliable way of checking for flash attention?
        if is_nvidia(): #pytorch flash attention only works on Nvidia
            return True
    return False

def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_allocated = stats['allocated_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = torch.xpu.get_device_properties(dev).total_memory - mem_allocated
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total

def batch_area_memory(area):
    if xformers_enabled() or pytorch_attention_flash_attention():
        #TODO: these formulas are copied from maximum_batch_area below
        return (area / 20) * (1024 * 1024)
    else:
        return (((area * 0.6) / 0.9) + 1024) * (1024 * 1024)

def maximum_batch_area():
    global vram_state
    if vram_state == VRAMState.NO_VRAM:
        return 0

    memory_free = get_free_memory() / (1024 * 1024)
    if xformers_enabled() or pytorch_attention_flash_attention():
        #TODO: this needs to be tweaked
        area = 20 * memory_free
    else:
        #TODO: this formula is because AMD sucks and has memory management issues which might be fixed in the future
        area = ((memory_free - 1024) * 0.9) / (0.6)
    return int(max(area, 0))

def cpu_mode():
    global cpu_state
    return cpu_state == CPUState.CPU

def mps_mode():
    global cpu_state
    return cpu_state == CPUState.MPS

def is_device_cpu(device):
    if hasattr(device, 'type'):
        if (device.type == 'cpu'):
            return True
    return False

def is_device_mps(device):
    if hasattr(device, 'type'):
        if (device.type == 'mps'):
            return True
    return False

def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    global directml_enabled

    if device is not None:
        if is_device_cpu(device):
            return False

    if FORCE_FP16:
        return True

    if device is not None: #TODO
        if is_device_mps(device):
            return False

    if FORCE_FP32:
        return False

    if directml_enabled:
        return False

    if cpu_mode() or mps_mode():
        return False #TODO ?

    if is_intel_xpu():
        return True

    if torch.cuda.is_bf16_supported():
        return True

    props = torch.cuda.get_device_properties("cuda")
    if props.major < 6:
        return False

    fp16_works = False
    #FP16 is confirmed working on a 1080 (GP104) but it's a bit slower than FP32 so it should only be enabled
    #when the model doesn't actually fit on the card
    #TODO: actually test if GP106 and others have the same type of behavior
    nvidia_10_series = ["1080", "1070", "titan x", "p3000", "p3200", "p4000", "p4200", "p5000", "p5200", "p6000", "1060", "1050"]
    for x in nvidia_10_series:
        if x in props.name.lower():
            fp16_works = True

    if fp16_works:
        free_model_memory = (get_free_memory() * 0.9 - minimum_inference_memory())
        if (not prioritize_performance) or model_params * 4 > free_model_memory:
            return True

    if props.major < 7:
        return False

    #FP16 is just broken on these cards
    nvidia_16_series = ["1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX"]
    for x in nvidia_16_series:
        if x in props.name:
            return False

    return True

def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        if force or is_nvidia(): #This seems to make things worse on ROCm so I only do it for cuda
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

def resolve_lowvram_weight(weight, model, key):
    if weight.device == torch.device("meta"): #lowvram NOTE: this depends on the inner working of the accelerate library so it might break.
        key_split = key.split('.')              # I have no idea why they don't just leave the weight there instead of using the meta device.
        op = comfy.utils.get_attr(model, '.'.join(key_split[:-1]))
        weight = op._hf_hook.weights_map[key_split[-1]]
    return weight

