import click
from lightning_pod.utils import teardown
from lightning_pod.utils.new import make_new
from lightning_pod.agents import static_trainer


@click.group()
def cli():
    pass


@cli.command("teardown")
def _teardown():
    teardown.main()


@cli.command("static-trainer")
def _static_trainer():
    static_trainer.main()


@cli.command("make-new")
def _new():
    teardown.main()
    make_new.main()
