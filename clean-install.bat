@echo off
if exist nodes_env rmdir /s /q nodes_env
if exist custom_nodes\ainodes_engine_base_nodes rmdir /s /q custom_nodes\ainodes_engine_base_nodes
python launcher.py