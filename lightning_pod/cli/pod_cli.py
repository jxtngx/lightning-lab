import os
import click
from lightning_pod.utils import teardown
from lightning_pod.utils import build
from lightning_pod.agents import trainer
from lightning_pod.conf import PROJECT_NAME


@click.group()
def cli():
    pass


@cli.command("teardown")
def _teardown():
    teardown.main()


@cli.command("run_trainer")
def _static_trainer():
    trainer = os.path.join("lightning_pod", "agents", "trainer.py")
    os.system(f"python3 {trainer}")


@cli.command("seed_new_pod")
def _new():
    teardown.main()
    build.main()
