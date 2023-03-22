"""ainodes-engine launcher"""
#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
from platform import platform


def main():
    """launch ainodes-engine"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_hf", action="store_true")
    parser.add_argument("--skip_base_nodes", action="store_true")
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--skip_update", action="store_true")
    parser.add_argument("--torch2", action="store_true")
    parser.add_argument("--no_console", action="store_true")
    args = parser.parse_args()

    if not args.local_hf:
        print("Using HF Cache in app dir")

    venv_path = "nodes_env"
    ensure_virtualenv_installed()
    create_venv(venv_path)
    update_project()

    activate_this = get_activate_path()
    activate_env(activate_this)
    install_requirements()
    check_python_version()
    run_main_script(args, venv_path)
def ensure_virtualenv_installed():
    try:
        # Check if virtualenv is installed
        subprocess.check_call(["virtualenv", "--version"])
    except (OSError, subprocess.CalledProcessError):
        # If virtualenv is not installed, install it using pip
        subprocess.check_call(["pip", "install", "virtualenv"])


def check_python_version():
    # List of supported Python versions
    python_versions = [(3, 10), (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (2, 7)]

    # Get current Python version
    current_version = sys.version_info[:2]

    # Iterate through the list of supported Python versions in descending order
    for version in python_versions:
        # Check if the version is less than 3.11 and greater than or equal to the current version
        if version < (3, 11) and version >= current_version:
            # If the version is found, return it as a string
            return f"{version[0]}.{version[1]}"

    # If no version is found, return None
    return None
def create_venv(venv_path):
    """create virtualenv environment"""
    if os.path.exists(venv_path):
        return
    try:
        version = check_python_version()
        if "Windows" in platform():
            subprocess.check_call(["python", "-m", "virtualenv", venv_path"])
        else:
            subprocess.check_call(["python3", "-m", "virtualenv", venv_path"])
    except subprocess.CalledProcessError as cpe:
        print(f"Command '{cpe.cmd}' failed with return code {cpe.returncode}")
        print("Error, Python 3.10 not found.")
        print("Trying to make env with any python available.")
        print("If you run into any issue, please delete test_venv folder")
        print("Then, install Python 3.10 before running the installer again.")
        subprocess.check_call(["python3", "-m", "virtualenv", venv_path])


def update_project():
    """update ainodes-engine"""
    try:
        subprocess.check_call(["git", "pull"])
    except subprocess.CalledProcessError as cpe:
        print(f"Command '{cpe.cmd}' failed with return code {cpe.returncode}")


def get_activate_path():
    """get activate_this.py path"""
    if "Windows" in platform():
        activate_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nodes_env",
            "Scripts",
            "activate_this.py",
        )
    else:
        activate_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "nodes_env",
            "bin",
            "activate_this.py",
        )
    return activate_path


def activate_env(activate_this):
    """activate env"""
    exec(open(activate_this, encoding="utf-8").read(), {"__file__": activate_this})


def install_requirements():
    """install requirements"""
    subprocess.check_call(["pip", "install", "-r", "requirements.txt"])


def run_main_script(args, venv_path):
    """run main.py"""
    python = (
        os.path.join(venv_path, "Scripts", "python.exe")
        if "Windows" in platform()
        else os.path.join(venv_path, "bin", "python")
    )
    cmd_args = [python, "main.py"]
    if args.local_hf:
        cmd_args.append("--local_hf")
    if args.light:
        cmd_args.append("--light")
    if args.skip_base_nodes:
        cmd_args.append("--skip_base_nodes")
    if args.skip_update:
        cmd_args.append("--skip_update")
    if args.torch2:
        cmd_args.append("--torch2")
    if args.no_console:
        cmd_args.append("--no_console")
    subprocess.Popen(cmd_args)

main()

"""if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")"""
