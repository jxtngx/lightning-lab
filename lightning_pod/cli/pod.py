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
from lightning_pod.components import sweeps
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule

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
@main.command("init")
def init() -> None:
    common_destructive_flow([teardown, build], command_name="init")


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


@trainer.command("run-example")
def run_example() -> None:

    model = PodModule()
    dm = PodDataModule()
    trainer = PodTrainer(fast_dev_run=True)
    trainer.fit(model=model, datamodule=dm)


@trainer.command("run")
@click.option("--em", default="wandb", type=click.Choice(["wandb", "aim"]))
@click.option("--project-name", default="lightningpod-train")
@click.option("--trial-count", default=10)
@click.option("--persist_model", is_flag=True)
@click.option("--persist_predictions", is_flag=True)
@click.option("--persist_splits", is_flag=True)
def run_trainer(em, project_name, trial_count, persist_model, persist_predictions, persist_splits) -> None:
    project_name = "-".join([project_name, em])
    if em == "wandb":
        trainer = sweeps.WandbTrainFlow(project_name=project_name, trial_count=trial_count)
        sweeps
        trainer.run(persist_model=persist_model, persist_predictions=persist_predictions, persist_splits=persist_splits)
    if em == "aim":
        raise NotImplementedError("Aim experiment manager is not implemented")


@main.group("sweep")
def sweep() -> None:
    pass


@sweep.command("wandb")
@click.option("--project-name", default="lightningpod-sweep-wandb")
@click.option("--trial-count", default=10)
def run_wandb_sweep(project_name, trial_count) -> None:
    sweep = sweeps.WandbSweepFlow(project_name=project_name, trial_count=trial_count)
    sweep.run()


@sweep.command("wandb-optuna")
@click.option("--project-name", default="lightningpod-sweep-optuna")
def run_wandb_optuna_sweep(project_name) -> None:
    sweep = sweeps.WandbOptunaSweepFlow(project_name=project_name)
    sweep.run()


@sweep.command("aim")
@click.option("--project-name", default="lightningpod-sweep-aim")
def run_aim_sweep(project_name) -> None:
    pass
