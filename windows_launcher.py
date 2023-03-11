import os
import sys
import urllib.request
import subprocess
import tempfile
import shutil

def install_and_launch():
    # Check if Python 3.8 is already installed
    try:
        subprocess.check_call(['python', '-c', 'import sys; assert sys.version_info >= (3, 8)'])
        # Python 3.8 is already installed, launch launcher.py using Python 3.8
        subprocess.check_call(['python', 'launcher.py'])
        return
    except subprocess.CalledProcessError:
        pass

    # Download Python 3.8 x64 installer
    url = 'https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe'
    with urllib.request.urlopen(url) as response:
        installer_data = response.read()

    # Save installer to temporary file
    installer_file = os.path.join(tempfile.gettempdir(), 'python-3.8.10-amd64.exe')
    with open(installer_file, 'wb') as f:
        f.write(installer_data)

    # Run installer
    subprocess.check_call([installer_file, '/quiet', f'TargetDir={os.path.expanduser("~")}\\AppData\\Local\\Programs\\Python\\Python38'])

    # Launch launcher.py using Python 3.8
    python_path = os.path.join(os.path.expanduser("~"), 'AppData', 'Local', 'Programs', 'Python', 'Python38', 'python.exe')
    subprocess.check_call([python_path, 'launcher.py'])

    # Remove temporary installer file
    os.remove(installer_file)

install_and_launch()