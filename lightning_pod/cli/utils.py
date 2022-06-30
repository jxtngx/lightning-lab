import os
import click
import shutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich import print as rprint


PROJECTPATH = os.getcwd()
FILEPATH = Path(__file__)


def _preserve_dir(main_source_dir: str, sub_source_dir: str, destination: str):
    destinationpath = os.path.join(PROJECTPATH, destination)
    if not os.path.isdir(destinationpath):
        os.mkdir(destinationpath)
    src = os.path.join(PROJECTPATH, main_source_dir, sub_source_dir)
    dest = os.path.join(PROJECTPATH, destinationpath, main_source_dir, sub_source_dir)
    shutil.copytree(src, dest)


def preserve_examples():
    _preserve_dir("lightning_pod", "core", "examples")
    _preserve_dir("lightning_pod", "pipeline", "examples")


def _clean_and_build_lightning_pod(module_to_copy):
    src = os.path.join(FILEPATH.parent, "seed", module_to_copy)
    dest = os.path.join(PROJECTPATH, "lightning_pod", module_to_copy)
    shutil.rmtree(dest)
    shutil.copytree(src, dest)


def make_new_lightning_pod():
    _clean_and_build_lightning_pod("core")
    _clean_and_build_lightning_pod("pipeline")


def build():
    preserve_examples()
    make_new_lightning_pod()


def teardown():

    cwd = os.getcwd()

    do_not_delete = "README.md"

    target_dirs = [
        os.path.join(cwd, "models", "checkpoints"),
        os.path.join(cwd, "models", "onnx"),
        os.path.join(cwd, "logs", "logger"),
        os.path.join(cwd, "logs", "profiler"),
        os.path.join(cwd, "data", "cache"),
        os.path.join(cwd, "data", "predictions"),
        os.path.join(cwd, "data", "training_split"),
        os.path.join(cwd, "docs"),
    ]

    for dir in target_dirs:
        for target in os.listdir(dir):
            targetpath = os.path.join(cwd, dir, target)
            if not os.path.isdir(targetpath):
                if target != do_not_delete:
                    os.remove(targetpath)
            else:
                dirpath = os.path.join(cwd, dir, target)
                shutil.rmtree(dirpath)


def show_purge_table():
    # TITLE
    table = Table(title="Directories To Be Purged")
    # COLUMNS
    table.add_column("Directory", justify="right", style="cyan", no_wrap=True)
    table.add_column("Contents", style="magenta")
    # ROWS
    for dirname in ["data", "logs", "models", os.path.join("lightning_pod", "core")]:
        dirpath = os.path.join(os.getcwd(), dirname)
        contents = ", ".join([f for f in os.listdir(dirpath) if f != "README.md"])
        table.add_row(dirname, contents)
    # SHOW
    console = Console()
    console.print(table)
    return


def show_destructive_behavior_warning():
    """
    uses rich console markup

    notes: https://rich.readthedocs.io/en/stable/markup.html
    """
    print()
    rprint(
        ":warning: [bold red]Alert![/bold red] This action has destructive behavior! :warning: "
    )
    print()
    rprint("The following directories will be [bold red]purged[/bold red]")
    print()
    show_purge_table()
    print()
    return


def common_destructive_flow(commands: list, command_name: str):
    show_destructive_behavior_warning()
    if click.confirm("Do you want to continue"):
        for command in commands:
            command()
        print()
        rprint(f"[bold green]{command_name.title()} complete[bold green]")
        print()
    else:
        print()
        rprint("[bold green]No Action Taken[/bold green]")
        print()
