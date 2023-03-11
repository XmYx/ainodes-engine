from setuptools import setup, find_packages
import platform
import os
import shutil

from win32com.client import Dispatch

with open('requirements_venv.txt') as f:
    requirements = f.read().splitlines()

entry_point = 'launcher:main'

# Get the path to the user's application data folder
appdata = os.environ.get('APPDATA')
if appdata is None:
    raise RuntimeError('APPDATA environment variable not set')
user_folder = os.path.join(appdata, 'ainodes-engine')

setup(
    name='ainodes-engine',
    version='0.1',
    author='deforum',
    author_email='miklos.mnagy@gmail.com',
    packages=find_packages(),
    install_requires=[
        'virtualenv'
    ],
    entry_points={
        'console_scripts': [
            f'ainodes={entry_point}'
        ]
    },
    # Install the package to the user's application data folder
    data_files=[(user_folder, ['launcher.py'])]
)

# Create a desktop shortcut
desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
shortcut_path = os.path.join(desktop, 'ainodes.lnk')

shell = Dispatch('WScript.Shell')
shortcut = shell.CreateShortCut(shortcut_path)
shortcut.TargetPath = os.path.join(user_folder, 'launcher.py')
shortcut.WorkingDirectory = user_folder
shortcut.save()