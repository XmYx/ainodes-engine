import os
import urllib.request
import subprocess
import tempfile
import shutil

def install_and_launch():
    # Check if Python 3.8 is already installed
    python_path = os.path.join(os.getcwd(), 'python38', 'python.exe')
    if os.path.isfile(python_path):
        # Python 3.8 is already installed, launch launcher.py using Python 3.8
        subprocess.check_call([python_path, 'launcher.py'])
        return

    # Download Python 3.8 x64 installer
    url = 'https://www.python.org/ftp/python/3.8.12/python-3.8.12-amd64.exe'
    with urllib.request.urlopen(url) as response:
        installer_data = response.read()

    # Save installer to temporary file
    installer_file = os.path.join(tempfile.gettempdir(), 'python-3.8.12-amd64.exe')
    with open(installer_file, 'wb') as f:
        f.write(installer_data)

    # Run installer
    subprocess.check_call([installer_file, '/quiet', f'TargetDir={os.getcwd()}\\python38'])

    # Launch launcher.py using Python 3.8
    subprocess.check_call([python_path, 'launcher.py'])

    # Remove temporary installer file
    os.remove(installer_file)


install_and_launch()