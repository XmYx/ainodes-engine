import json
import requests
import os
import shutil
import zipfile


def parse_custom_nodes_json():
    url = "https://github.com/ltdrdata/ComfyUI-Manager/raw/main/custom-node-list.json"

    # Fetch the JSON file from the URL
    response = requests.get(url)
    json_data = response.json()

    # Create the destination folders if they don't exist
    custom_nodes_folder = "ai_nodes/ainodes_engine_comfy_nodes/src/ai_nodes"
    extras_folder = "ai_nodes/ainodes_engine_comfy_nodes/src/extras"
    os.makedirs(custom_nodes_folder, exist_ok=True)
    os.makedirs(extras_folder, exist_ok=True)

    # Iterate over the custom nodes
    for node in json_data["custom_nodes"]:
        install_type = node["install_type"]
        files = node["files"]

        if install_type == "git-clone":
            # Git clone the repository to the ai_nodes folder
            for file_url in files:
                repo_name = file_url.split("/")[-1]
                git_clone_folder = os.path.join(custom_nodes_folder, repo_name.replace("-", "_"))
                if not os.path.exists(git_clone_folder):
                    os.system(f"git clone {file_url} {git_clone_folder}")

        elif install_type == "copy":
            # Download and copy the files to the extras folder
            for file_url in files:
                file_name = file_url.split("/")[-1]
                file_path = os.path.join(extras_folder, file_name)
                download_file(file_url, file_path)

        elif install_type == "unzip":
            # Download and unzip the files to the extras folder
            for file_url in files:
                file_name = file_url.split("/")[-1]
                file_path = os.path.join(extras_folder, file_name)
                download_file(file_url, file_path)
                unzip_file(file_path, extras_folder, encoding="latin-1")

        else:
            print(f"Unknown install_type: {install_type}")

    print("Custom nodes installation completed successfully.")


def download_file(url, file_path):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

def unzip_file(zip_path, destination_folder, encoding="utf-8"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

# Call the function to start the installation process
parse_custom_nodes_json()
