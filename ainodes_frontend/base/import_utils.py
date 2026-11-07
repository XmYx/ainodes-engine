"""Importing node packs, and the Qt attribute flags.

A node pack that cannot import - usually because an optional dependency is
missing - used to take the whole application down with it (issue #32).  Now
each module is imported on its own and failures are collected and printed as a
summary, so aiNodes starts with the packs that do work.
"""
import glob
import importlib
import os
import subprocess
import sys
import traceback

from qtpy.QtCore import Qt

#: ``{module path: short reason}`` for every node module that failed to import.
IMPORT_FAILURES = {}
#: Modules that imported cleanly.
IMPORTED_MODULES = []


def _short_reason(exc):
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        return f"missing dependency '{exc.name}'"
    return f"{type(exc).__name__}: {exc}"


def import_nodes_from_directory(directory, verbose=True):
    """Import every ``*_node*.py`` in ``directory``, skipping broken ones."""
    if any(part in directory for part in ("ainodes_backend", "backend")):
        return
    if "_nodes" not in directory:
        return

    package = directory.replace("\\", "/").replace("/", ".").strip(".")
    for node_file in sorted(glob.glob(os.path.join(directory, "*.py"))):
        filename = os.path.basename(node_file)
        if filename == "__init__.py" or "_node" not in filename:
            continue
        module_name = f"{package}.{filename[:-3]}"
        try:
            importlib.import_module(module_name)
            IMPORTED_MODULES.append(module_name)
        except Exception as exc:
            IMPORT_FAILURES[module_name] = _short_reason(exc)
            if verbose:
                print(f"[nodes] skipped {module_name}: {_short_reason(exc)}")
            if os.environ.get("AINODES_TRACE_IMPORTS"):
                traceback.print_exc()


def import_nodes_from_subdirectories(directory, verbose=True):
    import_nodes_from_directory(directory, verbose=verbose)
    if any(part in directory for part in ("ainodes_backend", "backend")):
        return
    if not directory.endswith("_nodes"):
        return
    print("Importing from", directory)
    for subdir in sorted(os.listdir(directory)):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path) and subdir not in ("base", "__pycache__"):
            import_nodes_from_directory(subdir_path, verbose=verbose)


def import_failure_report():
    """A short multi-line summary, or ``''`` when everything imported."""
    if not IMPORT_FAILURES:
        return ""
    lines = [f"{len(IMPORT_FAILURES)} node module(s) were skipped:"]
    missing = set()
    for module, reason in sorted(IMPORT_FAILURES.items()):
        lines.append(f"  {module}: {reason}")
        if reason.startswith("missing dependency"):
            missing.add(reason.split("'")[1])
    if missing:
        lines.append("")
        lines.append("Install the optional dependencies you want with:")
        lines.append(f"  {sys.executable} -m pip install " + " ".join(sorted(missing)))
    return "\n".join(lines)


def update_all_nodes_req(top_folder="./ai_nodes"):
    """``git pull`` + ``pip install -r`` for each external node pack.

    ``ai_nodes/`` is where separately cloned node packs live; it simply does
    not exist in a fresh checkout, which used to raise on ``--update``.
    """
    if not os.path.isdir(top_folder):
        print(f"[nodes] no {top_folder} directory - nothing to update")
        return []

    updated = []
    for folder in sorted(os.listdir(top_folder)):
        repository = os.path.join(top_folder, folder)
        if not os.path.isdir(os.path.join(repository, ".git")):
            continue
        print(f"[nodes] updating {folder}")
        subprocess.run(["git", "-C", repository, "pull"], check=False)
        requirements = os.path.join(repository, "requirements.txt")
        if os.path.isfile(requirements):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements],
                           check=False)
        updated.append(folder)
    return updated


def check_repo_update(folder_path):
    """``True`` when ``origin/main`` is ahead of the local HEAD."""
    try:
        subprocess.check_output(["git", "-C", folder_path, "fetch"])
        remote = subprocess.check_output(
            ["git", "-C", folder_path, "ls-remote", "--quiet", "--refs", "origin",
             "refs/heads/main"]).decode().strip().split()[0]
        local = subprocess.check_output(
            ["git", "-C", folder_path, "rev-parse", "HEAD"]).decode().strip()
        return local != remote
    except (subprocess.CalledProcessError, IndexError) as exc:
        print(f"[nodes] could not check {folder_path} for updates: {exc}")
        return None


def set_application_attributes(qapp, args):
    """Apply the ``--use_opengl_es``-style flags.

    Qt 6 removed several of the Qt 5 attributes (high-DPI scaling is always on
    now), so each one is applied only if this Qt build still has it.
    """
    flags = {
        "use_opengl_es": "AA_UseOpenGLES",
        "enable_high_dpi_scaling": "AA_EnableHighDpiScaling",
        "use_high_dpi_pixmaps": "AA_UseHighDpiPixmaps",
        "disable_window_context_help_button": "AA_DisableWindowContextHelpButton",
        "use_stylesheet_propagation_in_widget_styles":
            "AA_UseStyleSheetPropagationInWidgetStyles",
        "dont_create_native_widget_siblings": "AA_DontCreateNativeWidgetSiblings",
        "plugin_application": "AA_PluginApplication",
        "use_direct3d_by_default": "AA_MSWindowsUseDirect3DByDefault",
        "mac_plugin_application": "AA_MacPluginApplication",
        "disable_shader_disk_cache": "AA_DisableShaderDiskCache",
    }
    for arg_name, attribute_name in flags.items():
        if not getattr(args, arg_name, False):
            continue
        attribute = getattr(Qt.ApplicationAttribute, attribute_name, None)
        if attribute is None:
            print(f"[qt] this Qt build has no {attribute_name}; ignoring "
                  f"--{arg_name}")
            continue
        qapp.setAttribute(attribute)
