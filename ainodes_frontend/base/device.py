"""Accelerator detection for aiNodes Engine.

Everything in the app that needs to know "where do tensors live" goes through
here instead of calling ``torch.cuda.*`` directly, so the same code runs on
NVIDIA (CUDA), AMD (ROCm), Intel (XPU), Apple silicon (MPS) and plain CPU.

Note on ROCm: a ROCm build of torch still exposes its GPUs as ``torch.cuda``
devices (``torch.version.hip`` is set instead of ``torch.version.cuda``), so
the CUDA branch covers AMD too - only the reported backend name differs.
"""
import os
import platform
import sys

try:
    import torch

    torch_available = True
except Exception:  # torch is optional for --help / headless validation
    torch = None
    torch_available = False


_forced_device = None
_cached_device = None
_cached_backend = None


def _xpu_available():
    return (
        torch_available
        and hasattr(torch, "xpu")
        and callable(getattr(torch.xpu, "is_available", None))
        and torch.xpu.is_available()
    )


def _mps_available():
    try:
        return (
            torch_available
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False


def _cuda_available():
    try:
        return torch_available and torch.cuda.is_available()
    except Exception:
        return False


def init(forced=None):
    """Pin the device for the rest of the process.

    ``forced`` is a device string (``cuda``, ``cuda:1``, ``rocm``, ``xpu``,
    ``mps``, ``cpu``) coming from ``--device`` or the ``AINODES_DEVICE``
    environment variable. ``None``/``auto`` means detect.
    """
    global _forced_device, _cached_device, _cached_backend
    if forced in (None, "", "auto"):
        forced = os.environ.get("AINODES_DEVICE") or None
    if forced in ("", "auto"):
        forced = None
    _forced_device = forced
    _cached_device = None
    _cached_backend = None
    return get_torch_device()


def backend():
    """One of ``cuda``, ``rocm``, ``xpu``, ``mps``, ``cpu``."""
    if _cached_backend is None:
        get_torch_device()
    return _cached_backend


def get_torch_device():
    """The ``torch.device`` every node should put its tensors on."""
    global _cached_device, _cached_backend
    if _cached_device is not None:
        return _cached_device

    if not torch_available:
        _cached_backend, _cached_device = "cpu", "cpu"
        return _cached_device

    want = (_forced_device or "").lower()

    if want.startswith("cpu"):
        _cached_backend, _cached_device = "cpu", torch.device("cpu")
    elif want.startswith("mps") and _mps_available():
        _cached_backend, _cached_device = "mps", torch.device("mps")
    elif want.startswith("xpu") and _xpu_available():
        _cached_backend, _cached_device = "xpu", torch.device(want)
    elif (want.startswith("cuda") or want.startswith("rocm")) and _cuda_available():
        _cached_backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        _cached_device = torch.device(want if want.startswith("cuda") else "cuda")
    elif want:
        print(f"[device] '{_forced_device}' is not available, auto-detecting instead")
        return _detect()
    else:
        return _detect()

    return _cached_device


def _detect():
    global _cached_device, _cached_backend
    if _cuda_available():
        _cached_backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        _cached_device = torch.device("cuda", torch.cuda.current_device())
    elif _xpu_available():
        _cached_backend = "xpu"
        _cached_device = torch.device("xpu", torch.xpu.current_device())
    elif _mps_available():
        _cached_backend = "mps"
        _cached_device = torch.device("mps")
    else:
        _cached_backend = "cpu"
        _cached_device = torch.device("cpu")
    return _cached_device


def device_type():
    """``'cuda'``/``'xpu'``/``'mps'``/``'cpu'`` - the torch device *type*.

    ROCm reports ``cuda`` here, because that is what torch calls it.
    """
    dev = get_torch_device()
    return getattr(dev, "type", str(dev))


def is_accelerator():
    return device_type() != "cpu"


def to_device(obj, device=None):
    """``obj.to(device)``, tolerating objects without ``.to``."""
    device = device if device is not None else get_torch_device()
    try:
        return obj.to(device)
    except AttributeError:
        return obj


def device_count():
    if not torch_available:
        return 0
    try:
        if device_type() == "cuda":
            return torch.cuda.device_count()
        if device_type() == "xpu":
            return torch.xpu.device_count()
    except Exception:
        return 0
    return 1 if is_accelerator() else 0


def available_devices():
    """Human readable device strings, for settings dropdowns."""
    out = ["cpu"]
    if not torch_available:
        return out
    try:
        if _cuda_available():
            out = [f"cuda:{i}" for i in range(torch.cuda.device_count())] + out
        if _xpu_available():
            out = [f"xpu:{i}" for i in range(torch.xpu.device_count())] + out
        if _mps_available():
            out = ["mps"] + out
    except Exception:
        pass
    return out


def empty_cache():
    """Release cached accelerator memory, whatever the accelerator is."""
    if not torch_available:
        return
    try:
        if device_type() == "cuda":
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        elif device_type() == "xpu":
            torch.xpu.empty_cache()
        elif device_type() == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


def synchronize():
    if not torch_available:
        return
    try:
        if device_type() == "cuda":
            torch.cuda.synchronize()
        elif device_type() == "xpu":
            torch.xpu.synchronize()
        elif device_type() == "mps":
            torch.mps.synchronize()
    except Exception:
        pass


def manual_seed_all(seed):
    if not torch_available:
        return
    torch.manual_seed(seed)
    try:
        if device_type() == "cuda":
            torch.cuda.manual_seed_all(seed)
        elif device_type() == "xpu":
            torch.xpu.manual_seed_all(seed)
    except Exception:
        pass


def autocast_dtype():
    """A dtype that is actually fast on the detected device."""
    if not torch_available:
        return None
    if device_type() == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    if device_type() == "xpu":
        return torch.bfloat16
    if device_type() == "mps":
        return torch.float16
    return torch.float32


def describe():
    """One line for the console / about box."""
    if not torch_available:
        return "torch not installed - CPU only"
    dev = get_torch_device()
    extra = ""
    try:
        if device_type() == "cuda":
            extra = torch.cuda.get_device_name(dev.index or 0)
            if backend() == "rocm":
                extra += f" (ROCm {torch.version.hip})"
            else:
                extra += f" (CUDA {torch.version.cuda})"
        elif device_type() == "xpu":
            extra = torch.xpu.get_device_name(dev.index or 0) + " (Intel XPU)"
        elif device_type() == "mps":
            extra = f"Apple {platform.machine()} (MPS)"
        else:
            extra = platform.processor() or platform.machine()
    except Exception:
        pass
    return f"{dev} - {extra} | torch {torch.__version__} | python {sys.version.split()[0]}"
