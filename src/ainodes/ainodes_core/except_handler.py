import os
import platform
import subprocess
import time
from types import SimpleNamespace

import yaml
import traceback

from qtpy.QtGui import QColor

def handle_ainodes_exception():
    traceback_str = traceback.format_exc()
    # gs.error_stack.append(traceback_str)
    print(traceback_str)
    save_error_log()
    return True

def get_machine_info():
    info = ""
    info += f"Operating System: {platform.system()} {platform.release()}\n"
    info += "PIP List:\n"

    try:
        pip_list = subprocess.check_output(['pip', 'list']).decode('utf-8')
        info += pip_list
    except Exception as e:
        info += f"Failed to retrieve PIP list: {str(e)}\n"

    try:
        env_info = subprocess.check_output(['printenv']).decode('utf-8')
        info += f"\nenvinfo:\n{env_info}"
    except Exception as e:
        info += f"Failed to retrieve envinfo: {str(e)}\n"

    return info

def save_error_log():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    today = time.strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"error_log_{today}.txt")

    machine_info = get_machine_info()

    with open(log_file, 'a') as file:  # Open in append mode
        if os.path.getsize(log_file) == 0:
            # If the log file is empty, write the machine information
            file.write(f"BEGINNING\n")
            file.write(f"Machine Information:\n{machine_info}\n\n")

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Timestamp: {timestamp}\n")

        for error in gs.error_stack:
            file.write(error)
            file.write('\n---End of Error---\n')

    print(f"Error log saved at: {log_file}")