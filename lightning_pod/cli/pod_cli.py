import click
from lightning_pod.utils import teardown
from lightning_pod.utils import make_new
from lightning_pod.agents import run_trainer
from lightning_pod.conf import PROJECT_NAME


@click.group()
def cli():
    pass


@cli.command("teardown")
def _teardown():
    teardown()


@cli.command("run-trainer")
def _static_trainer():
    run_trainer()


@cli.command("make-new")
def _new():
    teardown()
    make_new()
