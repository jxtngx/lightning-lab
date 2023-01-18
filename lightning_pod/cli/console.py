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

from lightning_pod.cli.bugreport import bugreport
from lightning_pod.cli.utils import build, common_destructive_flow, make_bug_trainer, teardown
from lightning_pod.flows.optuna_flow import TrialFlow
from lightning_pod.flows.wandb_flow import SweepFlow

FILEPATH = Path(__file__)
PKGPATH = FILEPATH.parents[1]
PROJECTPATH = FILEPATH.parents[2]


@click.group()
def main() -> None:
    pass


@main.command("teardown")
def _teardown() -> None:
    common_destructive_flow([teardown], command_name="teardown")


# TODO add help description
@main.command("seed")
def seed() -> None:
    common_destructive_flow([teardown, build], command_name="seed")


@main.command("bug-report")
def bug_report() -> None:
    bugreport.main()
    print("\n")
    make_bug_trainer()
    trainer = os.path.join(PKGPATH, "core", "bug_trainer.py")
    run_command = " ".join(["python", trainer, " 2> boring_trainer_error.md"])
    os.system(run_command)
    os.remove(trainer)


@main.group("trainer")
def trainer() -> None:
    pass


# TODO add help description
@trainer.command("help")
def help() -> None:
    trainer = os.path.join(PKGPATH, "core", "trainer.py")
    os.system(f"python {trainer} --help")


# TODO add help description
@trainer.command("run-hydra")
@click.argument("hydra-args", nargs=-1)
def run_hydra(hydra_args: tuple) -> None:
    trainer = os.path.join(PROJECTPATH, "lightning_pod", "flows", "hydra", "hydra_flow.py")
    hydra_args = list(hydra_args)
    hydra_args = [f"'trainer.{i}'" for i in hydra_args]
    hydra_args = " ".join(hydra_args)
    run_command = " ".join(["python3", trainer, hydra_args])
    os.system(run_command)


@trainer.command("run-wandb")
@click.option("--project-name")
def run_wandb(project_name) -> None:
    sweep = SweepFlow(project_name=project_name)
    sweep.run()


@trainer.command("run-optuna")
@click.option("--project-name", default="lightning-examples-optuna")
def run_optuna(project_name) -> None:
    trial = TrialFlow(project_name=project_name)
    trial.run()
