import glob
import importlib
import os
import subprocess

from PyQt6.QtCore import Qt
from tqdm import tqdm


def update_all_nodes_req():
    top_folder = "./custom_nodes"
    folders = [folder for folder in os.listdir(top_folder) if os.path.isdir(os.path.join(top_folder, folder))]

    for folder in tqdm(folders, desc="Folders"):
        repository = f"{top_folder}/{folder}"
        # command = f"git -C {repository} stash && git -C {repository} pull && pip install -r {repository}/requirements.txt"
        command = f"git -C {repository} pull && pip install -r {repository}/requirements.txt"

        #print("RUNNING COMMAND", command)

        with tqdm(total=100, desc=f"Updating {folder}") as pbar:
            result = subprocess.run(command, shell=True, stdout=None, stderr=None,
                                    universal_newlines=True)
            pbar.update(50)  # Indicate that git pull is 50% complete
            pbar.set_description(f"Installing {folder}'s requirements")
            pbar.update(50)  # Indicate that requirements installation is 50% complete

def import_nodes_from_directory(directory):
    if "ainodes_backend" not in directory and "backend" not in directory and "_nodes" in directory:
        node_files = glob.glob(os.path.join(directory, "*.py"))
        for node_file in node_files:
            f = os.path.basename(node_file)
            if f != "__init__.py" and "_node" in f:
                module_name = os.path.basename(node_file)[:-3].replace('/', '.')
                dir = directory.replace('/', '.')
                dir = dir.replace('\\', '.').lstrip('.')
                module = importlib.import_module(f"{dir}.{module_name}")

def import_nodes_from_subdirectories(directory):

    if "ainodes_backend" not in directory and "backend" not in directory and directory.endswith("_nodes"):
        print("Importing from", directory)
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path) and subdir != "base":
                import_nodes_from_directory(subdir_path)


def check_repo_update(folder_path):
    repo_path = folder_path
    try:
        # Run 'git fetch' to update the remote-tracking branches
        subprocess.check_output(['git', '-C', repo_path, 'fetch'])

        # Get the commit hash of the remote 'origin/master' branch
        remote_commit_hash = subprocess.check_output(
            ['git', '-C', repo_path, 'ls-remote', '--quiet', '--refs', 'origin',
             'refs/heads/main']).decode().strip().split()[0]

        # Get the commit hash of the local branch
        local_commit_hash = subprocess.check_output(['git', '-C', repo_path, 'rev-parse', 'HEAD']).decode().strip().split()[0]

        if local_commit_hash != remote_commit_hash:
            return True
        else:
            return None
    except subprocess.CalledProcessError as e:
        print(e)
        return None


def set_application_attributes(qapp, args):
    if args.use_opengl_es:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES)
    if args.enable_high_dpi_scaling:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    if args.use_high_dpi_pixmaps:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    if args.disable_window_context_help_button:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_DisableWindowContextHelpButton)
    if args.use_stylesheet_propagation_in_widget_styles:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_UseStyleSheetPropagationInWidgetStyles)
    if args.dont_create_native_widget_siblings:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_DontCreateNativeWidgetSiblings)
    if args.plugin_application:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_PluginApplication)
    if args.use_direct3d_by_default:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_MSWindowsUseDirect3DByDefault)
    if args.mac_plugin_application:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_MacPluginApplication)
    if args.disable_shader_disk_cache:
        qapp.setAttribute(Qt.ApplicationAttribute.AA_DisableShaderDiskCache)