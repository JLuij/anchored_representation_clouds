"""
This utilities script determines and relays what
platform code is running on. This is useful in
maintaining relative imports
"""

import socket
import os
from pathlib import Path
import pathlib
from contextlib import contextmanager
from dotenv import load_dotenv

from enum import Enum, auto

# Possible platforms
class Platform(Enum):
    CLUSTER_INTERACTIVE = auto()
    CLUSTER_SUBMITTED = auto()
    CLUSTER = auto()
    LOCAL = auto()

## Get current platform
current_platform : Platform = None

# Load environment variables
load_dotenv()
environment_vars = dict(os.environ)
if socket.gethostname() == 'backspace':
    current_platform = Platform.LOCAL
else:
    current_platform = Platform.CLUSTER
    
    if 'SLURM_JOB_NAME' in environment_vars.keys():
        if environment_vars['SLURM_JOB_NAME'] == 'sinteractive':
            current_platform = Platform.CLUSTER_INTERACTIVE
    elif 'SLURM_JOBID' in environment_vars.keys():
        current_platform = Platform.CLUSTER_SUBMITTED

# Backwards compatibility
on_cluster = current_platform != Platform.LOCAL

print(f"Running on {current_platform.name}, {on_cluster=}")

## Set relative folders
code_root = Path(environment_vars['HOME']) / 'anchored_representation_clouds' if on_cluster else Path(environment_vars['ARC_DIR']) / 'anchored_representation_clouds'
dataset_root = Path(environment_vars['HOME']) / 'umbrella/datasets' if on_cluster else Path(environment_vars['ARC_DIR'])  / 'datasets'

assert code_root.exists()
assert dataset_root.exists()


## Sometimes a path type workaround is needed
@contextmanager
def posix_path_workaround():
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = temp

@contextmanager
def windows_path_workaround():
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
    try:
        yield
    finally:
        pathlib.WindowsPath = temp


def reload_module(modulename):
    import importlib
    importlib.reload(modulename)


def setup_logging(log_level):
    import logging
    logger = logging.getLogger().setLevel(log_level)

    # PIL will pollute the log each time an image is loaded
    logging.getLogger("PIL").setLevel(logging.WARNING)


def str2bool(v):
    import argparse

    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def cli_tuple_type(strings):
    strings = [int(x) for x in strings.split(',')]
    assert len(strings) == 2, f'!! Error. cli argument fixed-size must be ' \
        f'of length 2, not {len(strings)} ({strings})'
    return strings


def cli_partwhole(string):
    try:
        a, b = string.split('/')
        a = int(a)
        b = int(b)
        assert a <= b, f'{a} in {string} must be in [1, {b}]'
        assert a != 0, f'{a} in {string} must be in [1, {b}]'
        assert b >= a
    except Exception as e:
        print(e)
        print(f'!! Error. subset must be selected as part/whole, not {string}')
    return string
