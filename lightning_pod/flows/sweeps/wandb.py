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

from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch import optim

import wandb
from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class ObjectiveWork:
    def __init__(self, project_name: str, wandb_save_dir: str, log_preprocessing: bool):
        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.log_preprocessing = log_preprocessing
        self.sweep_name = "-".join(["Sweep", wandb.util.generate_id()])
        self.sweep_id = wandb.sweep(sweep=self.sweep_config, project=project_name)  # set once
        self.datamodule = PodDataModule()
        self.trial_number = 1

    @property
    def wandb_settings(self) -> Dict[str, Any]:
        return self.trainer.logger.experiment.settings

    @property
    def sweep_path(self):
        return "/".join([self.entity, self.project_name, "sweeps", self.sweep_id])

    @property
    def sweep_config(self):
        cfg = dict(
            method="random",
            name=self.sweep_name,
            metric={"goal": "maximize", "name": "val_acc"},
            parameters={
                "lr": {"min": 0.0001, "max": 0.1},
                "optimizer": {"distribution": "categorical", "values": ["Adam", "RMSprop", "SGD"]},
                "dropout": {"min": 0.2, "max": 0.5},
            },
        )
        return cfg

    @property
    def entity(self):
        return self.trainer.logger.experiment.entity

    def persist_model(self):
        """should be called after persist predictions"""
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self):
        self.trainer.persist_predictions()

    def persist_splits(self):
        """should be called after persist predictions"""
        self.trainer.datamodule.persist_splits()

    def _objective(self) -> float:

        logger = WandbLogger(
            project=self.project_name,
            name="-".join(["trial", str(self.trial_number)]),
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

    def stop(self):
        os.system(f"wandb sweep --stop {self.entity}/{self.project_name}/{self.sweep_id}")


class WandbSweepFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
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
        # _ helps to avoid LightningFlow from checking JSON serialization if converting to Lightning App
        self._objective_work = ObjectiveWork(self.project_name, self.wandb_dir, self.log_preprocessing)
        # self._wandb_api = wandb.Api()

    def run(
        self,
        persist_model: bool = False,
        persist_predictions: bool = False,
        persist_splits: bool = False,
    ) -> None:

        # this will block
        self._objective_work.run(count=2)
        # will only run after obejective is complete
        self._objective_work.stop()
        # sweep_results = self._wandb_api.sweep(self._objective_work.sweep_path)
        # print(sweep_results)

        if persist_model:
            self._objective_work.persist_model()
        if persist_predictions:
            self._objective_work.persist_predictions()
        if persist_splits:
            self._objective_work.persist_splits()

        if issubclass(WandbSweepFlow, LightningFlow):
            sys.exit()
