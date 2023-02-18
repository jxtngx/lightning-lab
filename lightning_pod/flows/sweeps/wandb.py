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
import sys
from typing import Any, Dict, Optional

import wandb
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from rich.console import Console
from rich.table import Table
from torch import optim

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class ObjectiveWork:
    def __init__(self, sweep_config: Dict[str, Any], project_name: str, wandb_save_dir: str, log_preprocessing: bool):
        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.log_preprocessing = log_preprocessing
        self.sweep_config = sweep_config
        self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=project_name)
        self.sweep_name = "-".join(["Sweep", self.sweep_id])
        self.sweep_config.update({"name": self.sweep_name})
        self.datamodule = PodDataModule()
        self.trial_number = 1

    @property
    def wandb_settings(self) -> Dict[str, Any]:
        return self.trainer.logger.experiment.settings

    @property
    def sweep_path(self) -> str:
        return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def entity(self) -> str:
        return self.trainer.logger.experiment.entity

    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def persist_splits(self) -> None:
        self.trainer.datamodule.persist_splits()

    def _objective(self) -> float:

        logger = WandbLogger(
            project=self.project_name,
            name="-".join(["sweep", self.sweep_id, "trial", str(self.trial_number)]),
            group=self.sweep_config["name"],
            save_dir=self.wandb_save_dir,
        )

        lr = wandb.config.lr
        optimizer_name = wandb.config.optimizer
        optimizer = getattr(optim, optimizer_name)
        dropout = wandb.config.dropout

        model = PodModule(dropout=dropout, optimizer=optimizer, lr=lr)

        trainer_init_kwargs = {
            "max_epochs": 10,
            "callbacks": [
                EarlyStopping(monitor="training_loss", mode="min"),
            ],
        }

        self.trainer = PodTrainer(
            logger=logger,
            **trainer_init_kwargs,
        )

        # logs hyperparameters to logs/wandb_logs/wandb/{run_name}/files/config.yaml
        hyperparameters = dict(optimizer=optimizer_name, lr=lr, dropout=dropout)
        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        self.trial_number += 1

        return self.trainer.callback_metrics["val_acc"].item()

    def run(self, count=5) -> float:
        wandb.agent(self.sweep_id, function=self._objective, count=count)

    def stop(self) -> None:
        os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")


class SweepFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
        trial_count: int = 10,
        wandb_dir: Optional[str] = conf.WANDBPATH,
        log_preprocessing: bool = False,
    ) -> None:
        """
        Notes:
            see: https://community.wandb.ai/t/run-best-model-off-sweep/2423
        """
        # settings
        self.project_name = project_name
        self.wandb_dir = wandb_dir
        self.log_preprocessing = log_preprocessing
        self.trial_count = trial_count
        # _ helps to avoid LightningFlow from checking JSON serialization if converting to Lightning App
        self._sweep_config = dict(
            method="random",
            metric={"goal": "maximize", "name": "val_acc"},
            parameters={
                "lr": {"min": 0.0001, "max": 0.1},
                "optimizer": {"distribution": "categorical", "values": ["Adam", "RMSprop", "SGD"]},
                "dropout": {"min": 0.2, "max": 0.5},
            },
        )
        self._objective_work = ObjectiveWork(
            self._sweep_config,
            self.project_name,
            self.wandb_dir,
            self.log_preprocessing,
        )
        self._wandb_api = wandb.Api()

    @property
    def best_run(self):
        return self._wandb_api.sweep(self._objective_work.sweep_path).best_run()

    @staticmethod
    def _display_report(best_config_dict: Dict[str, Any]) -> None:

        table = Table(title="Best Run Config")

        for col in best_config_dict.keys():
            table.add_column(col, header_style="cyan")

        table.add_row(*[str(v) for v in best_config_dict.values()])

        console = Console()
        console.print(table, new_line_start=True)
        print()

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
        display_report: bool = False,
    ) -> None:

        # this is blocking
        self._objective_work.run(count=self.trial_count)
        # will only run after objective is complete
        self._objective_work.stop()

        if display_report:
            self._display_report(self.best_run.config)
        if persist_model:
            self._objective_work.persist_model()
        if persist_predictions:
            self._objective_work.persist_predictions()
        if persist_splits:
            self._objective_work.persist_splits()
        if issubclass(SweepFlow, LightningFlow):
            sys.exit()


class TrainWork:
    def persist_model(self) -> None:
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self) -> None:
        self.trainer.persist_predictions()

    def persist_splits(self) -> None:
        self.trainer.datamodule.persist_splits()

    def run(
        self,
        lr: float,
        dropout: float,
        optimizer: str,
        project_name: str,
        training_run_name: str,
        wandb_dir: Optional[str] = conf.WANDBPATH,
    ) -> None:
        self.model = PodModule(lr=lr, dropout=dropout, optimizer=getattr(optim, optimizer))
        self.datamodule = PodDataModule()
        logger = WandbLogger(
            project=project_name,
            name=training_run_name,
            group="Training Runs",
            save_dir=wandb_dir,
        )
        trainer_init_kwargs = {
            "max_epochs": 100,
            "callbacks": [
                EarlyStopping(monitor="training_loss", mode="min"),
            ],
        }

        self.trainer = PodTrainer(logger=logger, **trainer_init_kwargs)
        self.trainer.fit(model=self.model, datamodule=self.datamodule)


class TrainFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
        sweep_trial_count: int = 10,
    ) -> None:
        self.project_name = project_name
        self._sweep_flow = SweepFlow(project_name=project_name, trial_count=sweep_trial_count)
        self._train_work = TrainWork()

    @property
    def lr(self) -> float:
        return self._sweep_flow.best_run.config["lr"]

    @property
    def dropout(self) -> float:
        return self._sweep_flow.best_run.config["dropout"]

    @property
    def optimizer(self) -> str:
        return self._sweep_flow.best_run.config["optimizer"]

    @property
    def sweep_group(self) -> str:
        return self._sweep_flow._sweep_config["name"]

    @property
    def run_name(self) -> str:
        return self.sweep_group.replace("Sweep", "train")

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
    ) -> None:
        self._sweep_flow.run(display_report=False)
        self._train_work.run(
            lr=self.lr,
            dropout=self.dropout,
            optimizer=self.optimizer,
            project_name=self.project_name,
            training_run_name=self.run_name,
        )

        if persist_model:
            self._train_work.persist_model()
        if persist_predictions:
            self._train_work.persist_predictions()
        if persist_splits:
            self._train_work.persist_splits()
        if issubclass(TrainFlow, LightningFlow):
            sys.exit()
