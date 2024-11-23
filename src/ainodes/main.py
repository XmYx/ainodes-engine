#Base Imports


#Internal Library Imports


#Parse Args
from ainodes_core.argsparser import args
from ainodes_core.config import ensure_directories, load_config, save_last_config

#Load Config

# Ensure required directories exist
ensure_directories()

# Load Config
config = load_config()
print(config)

# Your application code here (e.g., start GUI, run CLI, etc.)

# Save the config on exit
save_last_config(config)
print(args)

#Run Task

## GUI ##



## CLI ##



## WEBUI ##