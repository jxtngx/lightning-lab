import os
import shutil
from lightning_pod.utils.paths import create_target_path
from pathlib import Path
from lightning_pod.conf import PROJECT_NAME


def main():
    filepath = Path(__file__)
    project_root_path = create_target_path(filepath, PROJECT_NAME)

    do_not_delete = "01-README.md"

    target_dirs = [
        os.path.join(project_root_path, "models", "checkpoints"),
        os.path.join(project_root_path, "models", "production"),
        os.path.join(project_root_path, "logs", "lightning_logs"),
        os.path.join(project_root_path, "logs", "profiler"),
        os.path.join(project_root_path, "data", "predictions"),
        os.path.join(project_root_path, "data", "training_split"),
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


if __name__ == "__main__":
    main()
