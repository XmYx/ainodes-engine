import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="A versatile application with GUI, CLI, and WebUI modes."
    )

    # Mode options
    parser.add_argument(
        '--mode',
        choices=['gui', 'cli', 'webui'],
        default='gui',
        help="Select the mode to run the application in. Defaults to 'gui'."
    )

    # Config file option
    parser.add_argument(
        '--config',
        type=str,
        help="Path to the configuration file."
    )

    # Additional CLI-specific arguments
    parser.add_argument(
        '--task',
        type=str,
        help="Specify the task to run in CLI mode."
    )

    # Additional WebUI-specific arguments
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help="Host IP for WebUI. Defaults to '127.0.0.1'."
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help="Port number for WebUI. Defaults to 8080."
    )

    return parser.parse_args()

# Parse Args
args = parse_args()