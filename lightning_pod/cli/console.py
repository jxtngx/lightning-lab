import os
from pathlib import Path

import click

from lightning_pod.cli.utils import build, common_destructive_flow, teardown

FILEPATH = Path(__file__)
PKGPATH = FILEPATH.parents[1]


@click.group()
def main() -> None:
    pass


@main.command("teardown")
def _teardown() -> None:
    common_destructive_flow([teardown], command_name="tear down")


# TODO add help description
@main.command("seed")
def seed() -> None:
    common_destructive_flow([teardown, build], command_name="seed")


@main.group("trainer")
def trainer() -> None:
    pass


# TODO add help description
@trainer.command("help")
def help() -> None:
    trainer = os.path.join(PKGPATH, "core", "trainer.py")
    os.system(f"python {trainer} --help")


# TODO add help description
@trainer.command("run")
@click.argument("hydra-args", nargs=-1)
def run_trainer(hydra_args: tuple) -> None:
    trainer = os.path.join(PKGPATH, "core", "trainer.py")
    hydra_args = list(hydra_args)
    hydra_args = [f"'trainer.{i}'" for i in hydra_args]
    hydra_args = " ".join(hydra_args)
    run_command = " ".join(["python", trainer, hydra_args])
    os.system(run_command)
