"""Find, configure and import a ComfyUI checkout.

aiNodes used to get this for free from ``deforum.generators.comfy_utils``,
which pinned a ComfyUI revision from 2024 and hard-failed when deforum was
not installed (issue #32).  This module does the job on its own and works
against an existing ComfyUI install, so you can point aiNodes at the ComfyUI
you already run.

Resolution order for the ComfyUI directory:

1. ``--comfy-path`` on the command line
2. ``$AINODES_COMFYUI_PATH`` or ``$COMFYUI_PATH``
3. ``comfyui_path:`` in ``config/comfy.yaml``
4. ``./src/ComfyUI``  (where ``install.py --with-comfy`` puts a clone)
5. ``../ComfyUI``, ``~/ComfyUI``, ``~/ai/ComfyUI``

Import order matters.  ``comfy.cli_args`` parses ``sys.argv`` at import time
unless ``comfy.options.args_parsing`` is left False, which would make ComfyUI
choke on aiNodes' own flags - so ``comfy.options`` is imported first and left
alone, and ComfyUI's args are then set programmatically.
"""
import asyncio
import logging
import os
import sys

logger = logging.getLogger(__name__)

#: Set once :func:`ensure_comfy` succeeds.
COMFY_PATH = None
PROMPT_SERVER = None
_initialised = False
_loop = None

CANDIDATE_SUFFIXES = (
    os.path.join("src", "ComfyUI"),
    os.path.join("..", "ComfyUI"),
)


def _looks_like_comfy(path):
    if not path:
        return False
    path = os.path.abspath(os.path.expanduser(path))
    return os.path.isfile(os.path.join(path, "nodes.py")) and os.path.isdir(
        os.path.join(path, "comfy")
    )


def _from_config():
    for name in ("comfy.yaml", "comfy.yml"):
        cfg = os.path.join("config", name)
        if not os.path.isfile(cfg):
            continue
        try:
            import yaml

            with open(cfg, "r") as fh:
                data = yaml.safe_load(fh) or {}
            value = data.get("comfyui_path")
            if value:
                return os.path.expanduser(str(value))
        except Exception as exc:
            logger.warning("could not read %s: %s", cfg, exc)
    return None


def find_comfy(explicit=None):
    """Return the first plausible ComfyUI directory, or ``None``."""
    candidates = [
        explicit,
        os.environ.get("AINODES_COMFYUI_PATH"),
        os.environ.get("COMFYUI_PATH"),
        _from_config(),
    ]
    here = os.path.abspath(os.getcwd())
    candidates += [os.path.join(here, suffix) for suffix in CANDIDATE_SUFFIXES]
    candidates += [
        os.path.expanduser("~/ComfyUI"),
        os.path.expanduser("~/ai/ComfyUI"),
        os.path.expanduser("~/comfy/ComfyUI"),
    ]
    for candidate in candidates:
        if _looks_like_comfy(candidate):
            return os.path.abspath(os.path.expanduser(candidate))
    return None


def clone_comfy(target=None, ref=None):
    """Clone ComfyUI into ``target`` (default ``src/ComfyUI``)."""
    import subprocess

    target = target or os.path.join("src", "ComfyUI")
    if _looks_like_comfy(target):
        return os.path.abspath(target)
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"]
    if ref:
        cmd += ["--branch", ref]
    cmd += ["https://github.com/comfyanonymous/ComfyUI", target]
    print(f"[comfy] cloning ComfyUI into {target}")
    subprocess.run(cmd, check=True)
    return os.path.abspath(target)


def comfy_version(path=None):
    path = path or COMFY_PATH
    if not path:
        return None
    version_file = os.path.join(path, "comfyui_version.py")
    if not os.path.isfile(version_file):
        return None
    scope = {}
    try:
        with open(version_file, "r") as fh:
            exec(fh.read(), scope)  # the file is a single __version__ assignment
        return scope.get("__version__")
    except Exception:
        return None


def _apply_comfy_args(cli_args, device_backend=None, vram_state=None):
    """Translate aiNodes' device/VRAM choice into ComfyUI's arg namespace."""
    args = cli_args.args
    if device_backend == "cpu":
        args.cpu = True
    elif device_backend == "xpu":
        # ComfyUI picks XPU up automatically from torch; nothing to force.
        pass

    # aiNodes runs graphs itself and has no asset browser; the asset scanner
    # would otherwise spin up a background thread and a database.
    if hasattr(args, "disable_assets"):
        args.disable_assets = True

    mapping = {
        "low": "lowvram",
        "medium": "normalvram",
        "high": "highvram",
    }
    flag = mapping.get((vram_state or "").lower())
    if flag and hasattr(args, flag):
        for candidate in mapping.values():
            if hasattr(args, candidate):
                setattr(args, candidate, False)
        setattr(args, flag, True)
    return args


def configure_folder_paths(models_dir=None, base_path=None, keep_comfy_paths=True):
    """Point ComfyUI's ``folder_paths`` at aiNodes' own ``models/`` tree.

    ``keep_comfy_paths`` leaves the ComfyUI install's own model directories in
    the search list as well, so models you already downloaded for ComfyUI stay
    visible - that is the behaviour people expect when aiNodes is pointed at a
    live ComfyUI (issue #28, "it does not load from there").
    """
    import folder_paths

    base_path = os.path.abspath(base_path or os.getcwd())
    models_dir = os.path.abspath(models_dir or os.path.join(base_path, "models"))

    extensions = folder_paths.supported_pt_extensions

    simple = [
        "checkpoints",
        "loras",
        "vae",
        "clip",
        "text_encoders",
        "unet",
        "diffusion_models",
        "clip_vision",
        "style_models",
        "embeddings",
        "vae_approx",
        "gligen",
        "upscale_models",
        "hypernetworks",
        "photomaker",
        "model_patches",
    ]
    for key in simple:
        _prepend_path(folder_paths, key, os.path.join(models_dir, key), extensions,
                      keep=keep_comfy_paths)

    _prepend_path(folder_paths, "configs", os.path.join(models_dir, "configs"),
                  [".yaml"], keep=keep_comfy_paths)
    _prepend_path(folder_paths, "diffusers", os.path.join(models_dir, "diffusers"),
                  ["folder"], keep=keep_comfy_paths)
    _prepend_path(folder_paths, "controlnet", os.path.join(models_dir, "controlnet"),
                  extensions, keep=keep_comfy_paths)
    _prepend_path(folder_paths, "controlnet", os.path.join(models_dir, "t2i_adapter"),
                  extensions, keep=keep_comfy_paths)
    local_custom_nodes = os.path.join(base_path, "custom_nodes")
    # ComfyUI's loader does os.listdir() on every registered custom_nodes path
    # and raises if one is missing, so make sure ours exists.
    os.makedirs(local_custom_nodes, exist_ok=True)
    _prepend_path(folder_paths, "custom_nodes", local_custom_nodes, [],
                  keep=keep_comfy_paths)

    folder_paths.output_directory = os.path.join(base_path, "output")
    folder_paths.temp_directory = os.path.join(base_path, "temp")
    os.makedirs(folder_paths.output_directory, exist_ok=True)
    folder_paths.filename_list_cache = {}
    return folder_paths


def _prepend_path(folder_paths, key, path, extensions, keep=True):
    existing_paths, existing_ext = folder_paths.folder_names_and_paths.get(
        key, ([], set(extensions) if not isinstance(extensions, list) else extensions)
    )
    paths = [path]
    if keep:
        paths += [p for p in existing_paths if os.path.abspath(p) != os.path.abspath(path)]
    folder_paths.folder_names_and_paths[key] = (paths, existing_ext or extensions)


def load_extra_model_paths(config_path=None):
    """Apply an ``extra_model_paths.yaml``-style file if one is present."""
    config_path = config_path or os.path.join("config", "extra_model_paths.yaml")
    if not os.path.isfile(config_path):
        return False
    try:
        import utils.extra_config as extra_config

        extra_config.load_extra_path_config(config_path)
        print(f"[comfy] applied extra model paths from {config_path}")
        return True
    except Exception as exc:
        logger.warning("could not apply %s: %s", config_path, exc)
        return False


def ensure_comfy(path=None, device_backend=None, vram_state=None,
                 load_custom_nodes=True, load_api_nodes=False, install=False):
    """Make ``import nodes`` / ``import comfy.*`` work, and load the node packs.

    Returns the absolute ComfyUI path.  Raises ``RuntimeError`` when no
    ComfyUI can be found and ``install`` is False.
    """
    global COMFY_PATH, _initialised
    if _initialised:
        return COMFY_PATH

    found = find_comfy(path)
    if not found and install:
        found = clone_comfy()
    if not found:
        raise RuntimeError(
            "ComfyUI not found. Pass --comfy-path /path/to/ComfyUI, set "
            "AINODES_COMFYUI_PATH, put comfyui_path: in config/comfy.yaml, or "
            "run `python install.py --with-comfy` to clone one into src/ComfyUI."
        )

    COMFY_PATH = found
    if found not in sys.path:
        # Appended, not prepended: ComfyUI ships a top-level ``main.py`` too,
        # and prepending made ``from main import ...`` inside aiNodes resolve
        # to ComfyUI's entry point.
        sys.path.append(found)

    # ComfyUI's own argparse must not see aiNodes' flags.  Leaving
    # comfy.options.args_parsing at its default False makes cli_args use
    # parser.parse_args([]) instead of sys.argv.
    import comfy.options  # noqa: F401  (imported for the side effect above)
    import comfy.cli_args as cli_args

    _apply_comfy_args(cli_args, device_backend=device_backend, vram_state=vram_state)

    configure_folder_paths()
    load_extra_model_paths()

    import nodes

    if load_custom_nodes or load_api_nodes:
        loop = _ensure_loop()
        _start_prompt_server(loop)
        loop.run_until_complete(
            nodes.init_extra_nodes(init_custom_nodes=load_custom_nodes,
                                   init_api_nodes=load_api_nodes)
        )

    _initialised = True
    version = comfy_version(found) or "unknown"
    print(f"[comfy] using ComfyUI {version} at {found} "
          f"({len(nodes.NODE_CLASS_MAPPINGS)} node classes)")
    return found


def _ensure_loop():
    """A long-lived event loop for ComfyUI's async bits.

    It is deliberately never closed: ``PromptServer.instance`` keeps a
    reference to it, and some custom nodes schedule work on it at import.
    """
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


def _start_prompt_server(loop):
    """Create a ComfyUI ``PromptServer`` without listening on a socket.

    Quite a few custom node packs touch ``PromptServer.instance`` at import
    time (to register routes, send progress, or read ``/userdata``).  Building
    the instance costs nothing - ``PromptServer.__init__`` only sets up the
    aiohttp application object; ``start()`` is what would bind a port, and we
    never call it.
    """
    global PROMPT_SERVER
    if PROMPT_SERVER is not None:
        return PROMPT_SERVER
    try:
        import inspect

        import server

        kwargs = {}
        params = inspect.signature(server.PromptServer.__init__).parameters
        if "asset_manager" in params:
            # ComfyUI >= 0.44 takes an asset manager; aiNodes has no asset
            # browser, so the no-op one is what we want.
            from app.assets.manager import default_asset_manager

            kwargs["asset_manager"] = default_asset_manager()
        PROMPT_SERVER = server.PromptServer(loop, **kwargs)
        # aiNodes executes graphs itself, so nothing should try to queue
        # prompts through the HTTP API.
        PROMPT_SERVER.number = 0
        return PROMPT_SERVER
    except Exception as exc:
        logger.warning("ComfyUI PromptServer could not be created (%s); "
                       "custom nodes that need it may fail to load", exc)
        return None
