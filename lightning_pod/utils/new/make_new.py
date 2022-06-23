import os
import shutil
from pathlib import Path
from lightning_pod.utils.paths import create_target_path
from lightning_pod.conf import PROJECT_NAME

filepath = Path(__file__)
PROJECTPATH = create_target_path(filepath, PROJECT_NAME)


def preserve_example_module():
    exampledirpath = os.path.join(PROJECTPATH, "examples")
    os.mkdir(exampledirpath)
    src = os.path.join(PROJECTPATH, "lightning_pod", "agents")
    dest = os.path.join(PROJECTPATH, "examples")
    shutil.copy(src, dest)


def preserve_example_pipeline():
    exampledirpath = os.path.join(PROJECTPATH, "examples")
    os.mkdir(exampledirpath)
    src = os.path.join(PROJECTPATH, "lightning_pod", "pipeline")
    dest = os.path.join(PROJECTPATH, "examples")
    shutil.copy(src, dest)


def make_new_module():
    src = os.path.join(filepath.parent, "new_module.py")
    dest = os.path.join(PROJECTPATH, "lightning_pod", "agents", "_module.py")
    shutil.copy(src, dest)


def main():
    preserve_example_module()
    preserve_example_pipeline()
    make_new_module()
