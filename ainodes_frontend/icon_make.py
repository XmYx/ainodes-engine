from pathlib import Path
hex_content = Path("qss/icon.png").read_bytes()
Path('icon.py').write_text(f'icon = {hex_content}')