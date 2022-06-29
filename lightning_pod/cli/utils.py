import os
import shutil
from pathlib import Path


PROJECTPATH = os.getcwd()
FILEPATH = Path(__file__)


def _preserve_dir(main_source_dir: str, sub_source_dir: str, destination: str):
    destinationpath = os.path.join(PROJECTPATH, destination)
    if not os.path.isdir(destinationpath):
        os.mkdir(destinationpath)
    src = os.path.join(PROJECTPATH, main_source_dir, sub_source_dir)
    dest = os.path.join(PROJECTPATH, destinationpath, main_source_dir, sub_source_dir)
    shutil.copytree(src, dest)


def preserve_examples():
    _preserve_dir("lightning_pod", "agents", "examples")
    _preserve_dir("lightning_pod", "pipeline", "examples")


def _clean_and_build_lightning_pod(module_to_copy):
    src = os.path.join(FILEPATH.parent, module_to_copy)
    dest = os.path.join(PROJECTPATH, "lightning_pod", module_to_copy)
    shutil.rmtree(dest)
    shutil.copytree(src, dest)


def make_new_lightning_pod():
    _clean_and_build_lightning_pod("agents")
    _clean_and_build_lightning_pod("pipeline")


def build():
    preserve_examples()
    make_new_lightning_pod()


def teardown():
    filepath = Path(__file__)
    project_root_path = os.getcwd()

    do_not_delete = "01-README.md"

    target_dirs = [
        os.path.join(project_root_path, "models", "checkpoints"),
        os.path.join(project_root_path, "models", "onnx"),
        os.path.join(project_root_path, "logs", "logger"),
        os.path.join(project_root_path, "logs", "profiler"),
        os.path.join(project_root_path, "data", "cache"),
        os.path.join(project_root_path, "data", "predictions"),
        os.path.join(project_root_path, "data", "training_split"),
        os.path.join(project_root_path, "docs"),
    ]

    for dir in target_dirs:
        for target in os.listdir(dir):
            targetpath = os.path.join(project_root_path, dir, target)
            if not os.path.isdir(targetpath):
                if target != do_not_delete:
                    os.remove(targetpath)
            else:  ## for checkpoint version directories
                dirpath = os.path.join(project_root_path, dir, target)
                shutil.rmtree(dirpath)
