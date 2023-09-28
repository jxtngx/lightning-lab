from pathlib import Path

import click

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]


@click.group()
def main() -> None:
    pass
