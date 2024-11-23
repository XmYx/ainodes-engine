import os
import yaml
from pathlib import Path


def ensure_directories():
    home = Path.home()
    ainodes_path = home / "ainodes"
    config_path = ainodes_path / "configs"
    temp_path = ainodes_path / "temp"
    output_path = ainodes_path / "output"

    for path in [ainodes_path, config_path, temp_path, output_path]:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")


def load_config():
    home = Path.home()
    config_dir = home / "ainodes" / "configs"

    # Load default config
    default_config_path = Path("configs/default.yaml")
    if default_config_path.exists():
        with open(default_config_path, 'r') as file:
            config = yaml.safe_load(file)
            print("Loaded default config.")
    else:
        config = {}
        print("Default config not found, using empty config.")

    # Load last config if it exists
    last_config_path = config_dir / ".lastconfig.yaml"
    if last_config_path.exists():
        with open(last_config_path, 'r') as file:
            last_config = yaml.safe_load(file)
            config.update(last_config)
            print("Loaded and updated with last config.")

    return config


def save_last_config(config):
    home = Path.home()
    last_config_path = home / "ainodes" / "configs" / ".lastconfig.yaml"

    with open(last_config_path, 'w') as file:
        yaml.safe_dump(config, file)
        print(f"Saved current config to {last_config_path}")