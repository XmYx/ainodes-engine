from setuptools import setup, find_packages
import platform

with open('requirements_venv.txt') as f:
    requirements = f.read().splitlines()

entry_point = 'launcher:main'
"""if platform.system() == 'Windows':
    entry_point = 'run.bat'
else:
    entry_point = 'launcher:main'"""

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
    }
)