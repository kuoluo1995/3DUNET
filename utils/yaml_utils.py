import yaml
from pathlib import Path


def read(path):
    with Path(path).open('r') as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    return params


def write(path, data):
    with Path(path).open('w') as file:
        yaml.dump(data, file)
