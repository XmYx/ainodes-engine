import os, sys
import subprocess
from platform import platform
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("--local_hf", action="store_true")

args = parser.parse_args()
if not args.local_hf:
    print("Using HF Cache in app dir")
    #os.makedirs('hf_cache', exist_ok=True)
    #os.environ['HF_HOME'] = 'hf_cache'

def create_venv(venv_path):
    try:
        if 'Windows' in platform():
            subprocess.run(["python", "-m", "virtualenv", venv_path])
        else:
            subprocess.run(["python3", "-m", "virtualenv", venv_path])
    except:
        print("Error, Python 3.10 not found. Trying to make env with any python available. If you run into any issue, please delete test_venv folder, and install Python 3.10 before running the installer again.")
        subprocess.run(["python3", "-m", "virtualenv", venv_path])
    finally:
        if os.path.exists("nodes_env") == False:
            subprocess.run(["python", "-m", "virtualenv", venv_path])
        else:
            return

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
    subprocess.run(["git", "pull"])
    if not args.local_hf:
        subprocess.run([python, "main.py"])
    else:
        subprocess.run([python, "main.py", "--local_hf"])