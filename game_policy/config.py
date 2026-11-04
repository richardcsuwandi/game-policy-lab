"""Read literal project .env values without exporting secrets to subprocesses."""
import os
from pathlib import Path


def settings(root):
    values = {}
    path = Path(root) / '.env'
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, sep, value = line.partition('=')
            if not sep or not name.isidentifier():
                raise ValueError(f'Invalid assignment in .env on line {number}')
            values[name] = value.strip()
    return {**values, **os.environ}
