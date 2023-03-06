import os, sys
import subprocess

from platform import platform
# Determine the location of the executable

def create_venv(venv_path):
    try:
        if 'Windows' in platform():
            subprocess.run(["python3", "-m", "virtualenv", venv_path])
        else:
            subprocess.run(["python3", "-m", "virtualenv", venv_path])
    except:
        print("Error, Python 3.10 not found. Trying to make env with any python available. If you run into any issue, please delete test_venv folder, and install Python 3.10 before running the installer again.")
        subprocess.run(["python3", "-m", "virtualenv", venv_path])

def activate_venv(venv_path):
    activate_this = os.path.join(venv_path, "Scripts", "activate.bat")
    subprocess.run([activate_this])

if __name__ == "__main__":
    subprocess.run(["pip", "install", "-q", "virtualenv"])

    if os.path.exists("nodes_env") == False:
        create_venv("nodes_env")
    try:
        subprocess.run(["git", "pull"])
    except:
        pass

    if 'Windows' in platform():
        python = "nodes_env/Scripts/python.exe"
        activate_this = os.path.join(os.path.dirname(os.path.abspath(__file__)) + "/nodes_env/Scripts/activate_this.py")
    else:
        python = "nodes_env/bin/python"
        activate_this = "nodes_env/bin/activate_this.py"

    exec(open(activate_this).read(), {'__file__': activate_this})
    subprocess.run(["pip", "install", "-r", "requirements.txt"])
    subprocess.run([python, "main.py"])