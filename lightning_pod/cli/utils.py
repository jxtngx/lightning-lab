# Copyright Justin R. Goheen.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from pathlib import Path
from typing import Union

import click
from rich import print as rprint
from rich.console import Console
from rich.table import Table

FILEPATH = Path(__file__)
PROJECTPATH = FILEPATH.parents[2]
PKGPATH = FILEPATH.parents[1]


def _preserve_dir(main_source_dir: str, sub_source_dir: str, destination: str) -> None:
    destinationpath = os.path.join(PROJECTPATH, destination)
    if not os.path.isdir(destinationpath):
        os.mkdir(destinationpath)
    src = os.path.join(PROJECTPATH, main_source_dir, sub_source_dir)
    dest = os.path.join(PROJECTPATH, destinationpath, main_source_dir, sub_source_dir)
    shutil.copytree(src, dest)


def preserve_examples() -> None:
    _preserve_dir(PKGPATH.name, "core", "examples")
    _preserve_dir(PKGPATH.name, "pipeline", "examples")


def _clean_and_build_package(module_to_copy: Union[str, Path]) -> None:
    src = os.path.join(FILEPATH.parent, "init", module_to_copy)
    dest = os.path.join(PROJECTPATH, PKGPATH, module_to_copy)
    shutil.rmtree(dest)
    shutil.copytree(src, dest)


def make_new_package() -> None:
    _clean_and_build_package("core")
    _clean_and_build_package("pipeline")


def build() -> None:
    preserve_examples()
    make_new_package()


def teardown() -> None:

    do_not_delete = "README.md"

    target_dirs = [
        os.path.join(PROJECTPATH, "models", "checkpoints"),
        os.path.join(PROJECTPATH, "models", "onnx"),
        os.path.join(PROJECTPATH, "logs", "optuna"),
        os.path.join(PROJECTPATH, "logs", "tensorboard"),
        os.path.join(PROJECTPATH, "logs", "torch_profiler"),
        os.path.join(PROJECTPATH, "logs", "wandb_logs"),
        os.path.join(PROJECTPATH, "data", "cache"),
        os.path.join(PROJECTPATH, "data", "predictions"),
        os.path.join(PROJECTPATH, "data", "training_split"),
        os.path.join(PROJECTPATH, "docs"),
    ]

    for dir in target_dirs:
        for target in os.listdir(dir):
            targetpath = os.path.join(PROJECTPATH, dir, target)
            if not os.path.isdir(targetpath):
                if target != do_not_delete:
                    os.remove(targetpath)
            else:
                dirpath = os.path.join(PROJECTPATH, dir, target)
                shutil.rmtree(dirpath)


def make_bug_trainer():
    source = os.path.join(PROJECTPATH, "lightning_pod", "cli", "bugreport", "trainer.py")
    destination = os.path.join(PROJECTPATH, "lightning_pod", "core", "bug_trainer.py")
    shutil.copyfile(source, destination)


def show_purge_table(command_name) -> None:
    # TITLE
    table = Table(title="Directories To Be Purged")
    # COLUMNS
    table.add_column("Directory", justify="right", style="cyan", no_wrap=True)
    table.add_column("Contents", style="magenta")
    # ROWS
    trash = ["data", "logs", "models"]
    if command_name == "init":
        trash.append(os.path.join(PKGPATH, "core"))
    for dirname in trash:
        dirpath = os.path.join(os.getcwd(), dirname)
        contents = ", ".join([f for f in os.listdir(dirpath) if f != "README.md"])
        table.add_row(dirname, contents)
    # SHOW
    console = Console()
    console.print(table)


def show_destructive_behavior_warning(command_name) -> None:
    """
    uses rich console markup

    notes: https://rich.readthedocs.io/en/stable/markup.html
    """
    print()
    rprint(":warning: [bold red]Alert![/bold red] This action has destructive behavior! :warning: ")
    print()
    rprint("The following directories will be [bold red]purged[/bold red]")
    print()
    show_purge_table(command_name)
    print()


def common_destructive_flow(commands: list, command_name: str) -> None:
    show_destructive_behavior_warning(command_name)
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
