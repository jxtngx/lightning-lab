import click
from lightning_pod.utils import teardown
from lightning_pod.agents import static_trainer


@click.group()
def main():
    pass


@main.command("teardown")
def _teardown():
    teardown.main()


@main.command("static-trainer")
def _static_trainer():
    static_trainer.main()
