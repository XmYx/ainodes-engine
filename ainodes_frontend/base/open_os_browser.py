import subprocess
import platform

def open_folder_in_file_browser(folder_path):

    #print("Opening", folder_path.replace("\\", "/"))

    system = platform.system()
    if system == 'Darwin':  # macOS
        subprocess.run(['open', folder_path])
    elif system == 'Windows':  # Windows
        subprocess.run(['explorer', folder_path.replace("/", "\\")])
    elif system == 'Linux':  # Linux
        subprocess.run(['xdg-open', folder_path])
    else:
        print("Unsupported operating system.")