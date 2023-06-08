git clone https://github.com/XmYx/ainodes-engine
cd ainodes-engine
python -m venv nodes_env
source nodes_env/bin/activate
pip install -r requirements.txt
cd custom_nodes
git clone https://github.com/XmYx/ainodes_engine_base_nodes
cd ainodes_engine_base_nodes
pip install -r requirements.txt
cd ..
git clone https://github.com/XmYx/ainodes_engine_deforum_nodes
cd ainodes_engine_deforum_nodes
pip install -r requirements.txt