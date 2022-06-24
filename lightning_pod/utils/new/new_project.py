import os
import shutil
from pathlib import Path
from lightning_pod.utils.paths import create_target_path
from lightning_pod.conf import PROJECT_NAME

FILEPATH = Path(__file__)
PROJECTPATH = create_target_path(FILEPATH, PROJECT_NAME)


def _preserve_dir(main_source_dir: str, sub_source_dir: str, destination: str):
    destinationpath = os.path.join(PROJECTPATH, destination)
    if not os.path.isdir(destinationpath):
        os.mkdir(destinationpath)
    src = os.path.join(PROJECTPATH, main_source_dir, sub_source_dir)
    dest = os.path.join(PROJECTPATH, destinationpath)
    shutil.copy(src, dest)


def preserve_examples():
    _preserve_dir("lightning_pod", "agents", "examples")
    _preserve_dir("lightning_pod", "pipeline", "examples")


def _clean_and_build_lightning_pod(module_to_copy):
    src = os.path.join(FILEPATH.parent, module_to_copy)
    dest = os.path.join(PROJECTPATH, "lightning_pod", module_to_copy)
    shutil.copy(src, dest)


def make_new_lightning_pod():
    _clean_and_build_lightning_pod("agents")
    _clean_and_build_lightning_pod("pipeline")
