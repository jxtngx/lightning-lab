import os
import shutil
from pathlib import Path

import click

from lightning_pod.backend.bugreport import bugreport

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]


@click.group()
def main() -> None:
    pass


@main.command("bug-report")
def bug_report() -> None:
    def make_bug_trainer():
        source = os.path.join(PROJECTPATH, "lightning_pod", "backend", "bugreport", "trainer.py")
        destination = os.path.join(PROJECTPATH, "lightning_pod", "core", "bug_trainer.py")
        shutil.copyfile(source, destination)

    bugreport.main()
    print("\n")
    make_bug_trainer()
    trainer = os.path.join(PKGPATH, "core", "bug_trainer.py")
    run_command = " ".join(["python", trainer, " 2> boring_trainer_error.md"])
    os.system(run_command)
    os.remove(trainer)
