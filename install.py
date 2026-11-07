#!/usr/bin/env python3
"""aiNodes Engine installer - Linux, macOS and Windows.

    python install.py                      # detect the GPU, build nodes_env/
    python install.py --comfy-path ~/ComfyUI
    python install.py --with-comfy         # clone ComfyUI into src/ComfyUI
    python install.py --torch-backend cpu  # force a build

The old setup.py/ainodes.sh/setup_ainodes.bat trio downloaded Miniconda,
created a venv inside a venv, cloned repositories listed in config/src.txt and
installed two different requirements files depending on the platform - which
is what issue #32 ran into.  This replaces all of it with one script that does
the same job on every platform.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(ROOT, "nodes_env")
MIN_PYTHON = (3, 11)
MAX_PYTHON = (3, 14)

# PyTorch publishes one wheel index per accelerator.  Override with
# --torch-index if you need a different CUDA/ROCm level than the default.
TORCH_INDEXES = {
    "cu130": "https://download.pytorch.org/whl/cu130",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "rocm": "https://download.pytorch.org/whl/rocm6.4",
    "xpu": "https://download.pytorch.org/whl/xpu",
    "cpu": "https://download.pytorch.org/whl/cpu",
    # macOS wheels on PyPI already carry MPS support
    "mps": None,
}


def info(message):
    print(f"[install] {message}", flush=True)


def fail(message, code=1):
    print(f"[install] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def run(command, **kwargs):
    info(" ".join(str(part) for part in command))
    result = subprocess.run(command, **kwargs)
    if result.returncode != 0:
        fail(f"command failed with exit code {result.returncode}")
    return result


def have(executable):
    return shutil.which(executable) is not None


# ---------------------------------------------------------------- detection

def detect_backend():
    """Guess the accelerator from what is installed on this machine."""
    system = platform.system()

    if system == "Darwin":
        return "mps" if platform.machine() in ("arm64", "aarch64") else "cpu"

    if have("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return "cu130"
        except Exception:
            pass

    if have("rocminfo") or have("rocm-smi") or os.path.isdir("/opt/rocm"):
        return "rocm"

    if have("xpu-smi") or have("sycl-ls"):
        return "xpu"

    # Intel Arc shows up in lspci even without the oneAPI tools installed
    if system == "Linux" and have("lspci"):
        try:
            out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=20)
            text = out.stdout.lower()
            if "intel" in text and ("arc " in text or "dg2" in text or "battlemage" in text):
                return "xpu"
        except Exception:
            pass

    return "cpu"


def describe_backend(backend):
    return {
        "cu130": "NVIDIA GPU (CUDA 13.0 wheels)",
        "cu128": "NVIDIA GPU (CUDA 12.8 wheels)",
        "rocm": "AMD GPU (ROCm wheels)",
        "xpu": "Intel GPU (XPU wheels)",
        "mps": "Apple silicon (MPS, default PyPI wheels)",
        "cpu": "CPU only",
    }.get(backend, backend)


# ------------------------------------------------------------------- python

def check_python():
    version = sys.version_info[:2]
    if version < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, this is "
             f"{version[0]}.{version[1]}")
    if version > MAX_PYTHON:
        info(f"warning: Python {version[0]}.{version[1]} is newer than the "
             f"{MAX_PYTHON[0]}.{MAX_PYTHON[1]} this was tested against; "
             "wheels may not exist yet")
    info(f"python {sys.version.split()[0]} at {sys.executable}")


def venv_python(venv_dir):
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_venv(venv_dir, use_uv):
    python = venv_python(venv_dir)
    if os.path.isfile(python):
        info(f"reusing the existing environment at {venv_dir}")
        return python
    info(f"creating a virtual environment at {venv_dir}")
    if use_uv:
        run(["uv", "venv", "--python", sys.executable, venv_dir])
    else:
        run([sys.executable, "-m", "venv", venv_dir])
    if not os.path.isfile(python):
        fail(f"the environment was created but {python} is missing")
    return python


def pip_install(python, arguments, use_uv, index_url=None):
    if use_uv:
        command = ["uv", "pip", "install", "--python", python]
        if index_url:
            # uv only looks at the first index that carries a package, which
            # would pin everything to PyTorch's mirror of PyPI; this makes it
            # pick the best version across both indexes, like pip does.
            command += ["--extra-index-url", index_url,
                        "--index-strategy", "unsafe-best-match"]
    else:
        command = [python, "-m", "pip", "install", "--upgrade"]
        if index_url:
            command += ["--extra-index-url", index_url]
    run(command + list(arguments))


# -------------------------------------------------------------------- comfy

def write_comfy_config(path):
    os.makedirs(os.path.join(ROOT, "config"), exist_ok=True)
    config = os.path.join(ROOT, "config", "comfy.yaml")
    with open(config, "w", encoding="utf-8") as fh:
        fh.write("# Where aiNodes looks for ComfyUI.\n")
        fh.write("# Override at runtime with --comfy-path or $AINODES_COMFYUI_PATH.\n")
        fh.write(f"comfyui_path: {path}\n")
    info(f"wrote {config}")


def clone_comfy(target):
    if os.path.isfile(os.path.join(target, "nodes.py")):
        info(f"ComfyUI is already at {target}")
        return target
    if not have("git"):
        fail("git is needed to clone ComfyUI; install it or pass --comfy-path")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    run(["git", "clone", "--depth", "1",
         "https://github.com/comfyanonymous/ComfyUI", target])
    return target


def install_comfy_requirements(python, comfy_path, use_uv, index_url, custom_nodes):
    requirements = os.path.join(comfy_path, "requirements.txt")
    if os.path.isfile(requirements):
        info("installing ComfyUI's own requirements")
        pip_install(python, ["-r", requirements], use_uv, index_url)
    if not custom_nodes:
        return
    custom_dir = os.path.join(comfy_path, "custom_nodes")
    if not os.path.isdir(custom_dir):
        return
    for pack in sorted(os.listdir(custom_dir)):
        pack_requirements = os.path.join(custom_dir, pack, "requirements.txt")
        if not os.path.isfile(pack_requirements):
            continue
        info(f"installing requirements for custom node pack '{pack}'")
        # A failing pack must not abort the install, so this one is not run().
        if use_uv:
            command = ["uv", "pip", "install", "--python", python,
                       "--index-strategy", "unsafe-best-match"]
            if index_url:
                command += ["--extra-index-url", index_url]
        else:
            command = [python, "-m", "pip", "install"]
            if index_url:
                command += ["--extra-index-url", index_url]
        subprocess.run(command + ["-r", pack_requirements])


# --------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--venv", default=VENV_DIR,
                        help="where to build the environment (default: nodes_env)")
    parser.add_argument("--torch-backend", choices=sorted(TORCH_INDEXES) + ["auto"],
                        default="auto",
                        help="accelerator wheels to install (default: detect)")
    parser.add_argument("--torch-index", default=None,
                        help="override the PyTorch wheel index URL entirely")
    parser.add_argument("--comfy-path", default=None,
                        help="use an existing ComfyUI checkout")
    parser.add_argument("--with-comfy", action="store_true",
                        help="clone ComfyUI into src/ComfyUI")
    parser.add_argument("--comfy-requirements", action="store_true",
                        help="also install ComfyUI's requirements into this environment")
    parser.add_argument("--comfy-node-requirements", action="store_true",
                        help="also install every ComfyUI custom node pack's requirements")
    parser.add_argument("--optional", action="store_true",
                        help="install requirements-optional.txt as well")
    parser.add_argument("--no-uv", action="store_true",
                        help="use pip even when uv is available")
    parser.add_argument("--system", action="store_true",
                        help="install into the current interpreter, no venv")
    options = parser.parse_args()

    check_python()

    backend = options.torch_backend
    if backend == "auto":
        backend = detect_backend()
    info(f"target: {describe_backend(backend)}")

    index_url = options.torch_index or TORCH_INDEXES.get(backend)
    if index_url:
        info(f"torch wheel index: {index_url}")

    use_uv = have("uv") and not options.no_uv
    info("package manager: " + ("uv" if use_uv else "pip"))

    if options.system:
        python = sys.executable
        info("installing into the current interpreter (--system)")
    else:
        python = create_venv(options.venv, use_uv)
        if not use_uv:
            run([python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

    info("installing aiNodes' requirements")
    pip_install(python, ["-r", os.path.join(ROOT, "requirements.txt")], use_uv, index_url)

    if options.optional:
        info("installing the optional extras")
        pip_install(python, ["-r", os.path.join(ROOT, "requirements-optional.txt")],
                    use_uv, index_url)

    comfy_path = options.comfy_path
    if options.with_comfy and not comfy_path:
        comfy_path = clone_comfy(os.path.join(ROOT, "src", "ComfyUI"))
    if comfy_path:
        comfy_path = os.path.abspath(os.path.expanduser(comfy_path))
        if not os.path.isfile(os.path.join(comfy_path, "nodes.py")):
            fail(f"{comfy_path} does not look like a ComfyUI checkout")
        write_comfy_config(comfy_path)
        if options.comfy_requirements or options.comfy_node_requirements:
            install_comfy_requirements(python, comfy_path, use_uv, index_url,
                                       options.comfy_node_requirements)
        else:
            info("ComfyUI's own requirements were NOT installed; add "
                 "--comfy-requirements if its nodes fail to import")

    print()
    info("done. Start aiNodes with:")
    if options.system:
        print(f"    {python} main.py")
    elif os.path.abspath(options.venv) == VENV_DIR:
        print("    ainodes.bat" if platform.system() == "Windows" else "    ./ainodes.sh")
    else:
        print(f"    {python} main.py")
    if not comfy_path:
        print()
        info("no ComfyUI configured - pass --comfy-path /path/to/ComfyUI, set "
             "$AINODES_COMFYUI_PATH, or rerun with --with-comfy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
