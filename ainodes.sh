#!/usr/bin/env bash

# Define script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Configuration
SRC_DIR="${SCRIPT_DIR}/src"
INSTALL_DIR="${SCRIPT_DIR}/installer_files"
INSTALL_ENV_DIR="${SCRIPT_DIR}/nodes_env"
INSTALL_ENV_NAME="nodes_env"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-linux.txt"
MINICONDA_DOWNLOAD_URL="https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh"
delimiter="################################################################"
PYTHON_CMD="python3"
# Function to check and install Miniconda
install_miniconda() {
    if ! command -v conda &> /dev/null; then
        echo "Downloading and installing Miniconda..."
        curl -L "${MINICONDA_DOWNLOAD_URL}" -o "${INSTALL_DIR}/miniconda_installer.sh"
        bash "${INSTALL_DIR}/miniconda_installer.sh" -b -p "${INSTALL_DIR}/conda"
        export PATH="${INSTALL_DIR}/conda/bin:$PATH"
    fi
}

# Function to create and activate the environment
create_and_activate_env() {
    printf "\n%s\n" "${delimiter}"
    printf "Create and activate python venv\n"
    printf "\n%s\n" "${delimiter}"

    if [[ ! -d "${INSTALL_ENV_NAME}" ]]; then
        echo "Creating virtual environment..."
        "${PYTHON_CMD}" -m venv "${INSTALL_ENV_NAME}" || { echo "Failed to create venv"; exit 1; }
    else
        echo "Virtual environment already exists."
    fi

    if [[ -f "${INSTALL_ENV_NAME}/bin/activate" ]]; then
        echo "Activating virtual environment..."
        source "${INSTALL_ENV_NAME}/bin/activate"
    else
        printf "\n%s\n" "${delimiter}"
        printf "\e[1m\e[31mERROR: Cannot activate python venv, aborting...\e[0m\n"
        printf "\n%s\n" "${delimiter}"
        exit 1
    fi
}

# Install Miniconda and create environment
#mkdir -p "${INSTALL_DIR}"
#install_miniconda
create_and_activate_env

# Install requirements
pip install -r "${REQUIREMENTS_FILE}"

# Clone repositories as per 'src.txt'
src_file="${SCRIPT_DIR}/config/src.txt"
while IFS= read -r line || [[ -n "$line" ]]; do
    repository=$(echo $line | cut -d ' ' -f1)
    branch=$(echo $line | cut -d ' ' -f2-3)
    directory=$(echo $line | cut -d ' ' -f4)
    echo "Cloning repository: $repository into $directory"
    mkdir -p "${SRC_DIR}/$directory"
    cd "${SRC_DIR}/$directory"
    git clone "https://www.github.com/$repository" . "$branch"
    pip install -e .
    cd - > /dev/null
done < "$src_file"

# Main execution logic
echo "Installation complete. Starting the application..."
# Add the application's main execution commands here


# Run the main script
PYTHON_DIR="$SCRIPT_DIR/src/python"
PYTHON_LIB_DIR="$PYTHON_DIR/Lib"
PYTHON_SCRIPTS_DIR="$PYTHON_DIR/Scripts"
export PATH="$PYTHON_SCRIPTS_DIR:$PYTHON_LIB_DIR:$PYTHON_DIR:$PATH"

python main.py "$@"