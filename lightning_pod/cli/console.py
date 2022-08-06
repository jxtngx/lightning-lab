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
