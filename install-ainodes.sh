
# create and activate python environment
python -m venv .env
source .env/bin/activate

# update ainodes
git pull origin

# install dependencies
python -m pip install -r requirements-linux.txt

mkdir src
cd src
git clone https://www.github.com/XmYx/deforum-studio deforum
cd deforum
pip install -e .
cd ../
git clone https://github.com/comfyanonymous/ComfyUI
