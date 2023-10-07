from pathlib import Path

import typer

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]

app = typer.Typer()


@app.callback()
def callback() -> None:
    pass


@app.command()
def run() -> None:
    typer.echo("boop bop beep")
