import os
import click
from lightning_pod.utils import teardown
from lightning_pod.utils import build


@click.group()
def main():
    pass


@main.group("project")
def project():
    pass


# TODO add help description
@project.command("teardown")
def _teardown():
    teardown.main()


# TODO add help description
@project.command("seed")
def seed():
    teardown.main()
    build.main()


@main.group("trainer")
def trainer():
    pass


# TODO add help description
@trainer.command("config-help")
def config_help():
    trainer = os.path.join("lightning_pod", "agents", "trainer.py")
    os.system(f"python {trainer} --help")


# TODO add help description
@trainer.command("run")
@click.argument("hydra-args", nargs=-1)
def run_trainer(hydra_args):
    trainer = os.path.join("lightning_pod", "agents", "trainer.py")
    hydra_args = list(hydra_args)
    hydra_args = [f"'trainer.{i}'" for i in hydra_args]
    hydra_args = " ".join(hydra_args)
    run_command = " ".join(["python", trainer, hydra_args])
    os.system(run_command)
