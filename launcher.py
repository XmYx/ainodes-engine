"""ainodes-engine launcher"""
#!/usr/bin/env python3

import os
import argparse
import subprocess
from platform import platform


def main():
    """launch ainodes-engine"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_hf", action="store_true")
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--skip_base_nodes", action="store_true")
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
    run_main_script(args, venv_path)
def ensure_virtualenv_installed():
    try:
        # Check if virtualenv is installed
        subprocess.check_call(["virtualenv", "--version"])
    except (OSError, subprocess.CalledProcessError):
        # If virtualenv is not installed, install it using pip
        subprocess.check_call(["pip", "install", "virtualenv"])

def create_venv(venv_path):
    """create virtualenv environment"""
    if os.path.exists(venv_path):
        return
    try:
        if "Windows" in platform():
            subprocess.check_call(["python", "-m", "virtualenv", venv_path])
        else:
            subprocess.check_call(["python3", "-m", "virtualenv", venv_path])
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
    subprocess.check_call(cmd_args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
