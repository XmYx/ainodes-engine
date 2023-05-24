import platform
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

entry_point = 'launcher:main'

# Define platform-specific settings
platform_specific = {
    'windows': {
        'install_requires': ['virtualenv'],
        'entry_point': 'launcher:main --skip_update',
        'shortcut_script': 'create_shortcut.bat',
        'shortcut_icon': 'icon.ico',
    },
    'linux': {
        'install_requires': ['virtualenv'],
        'entry_point': f'python3 {entry_point} --skip_update',
        'shortcut_script': 'create_shortcut.sh',
        'shortcut_icon': 'icon.png',
    },
    'darwin': {
        'install_requires': ['virtualenv'],
        'entry_point': f'python3 {entry_point} --skip_update',
        'shortcut_script': 'create_shortcut.sh',
        'shortcut_icon': 'icon.png',
    },
}

# Get the current platform
current_platform = platform.system().lower()

# Set platform-specific settings
if current_platform in platform_specific:
    settings = platform_specific[current_platform]
else:
    raise ValueError(f"Unsupported platform: {current_platform}")

# Set the version requirements
python_requires = '>=3.8, <3.11' if current_platform == 'windows' else '>=3.8'

# Set the console_scripts entry point
console_scripts_entry_point = f'launcher:main'

# Set the entry point for Windows cmd
windows_entry_point = f'{settings["entry_point"]}'

# Set the install_requires based on the platform
install_requires = settings['install_requires']

# Create a shortcut for Windows
if current_platform == 'windows':
    import os
    from setuptools import Command

    class CreateShortcut(Command):
        description = 'create shortcut'
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            os.system(settings['shortcut_script'])

    cmdclass = {
        'create_shortcut': CreateShortcut,
    }
else:
    cmdclass = {}


setup(
    name='ainodes-engine',
    version='0.1',
    author='deforum',
    author_email='miklos.mnagy@gmail.com',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=python_requires,
    entry_points={
        'console_scripts': [
            f'ainodes={console_scripts_entry_point}'
        ]
    },
    cmdclass=cmdclass,
)