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

import logging
import os
import sys
from typing import Any, Dict, Optional

import optuna
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import Logger, WandbLogger
from optuna.integration import PyTorchLightningPruningCallback
from optuna.trial import TrialState
from rich.console import Console
from rich.table import Table
from torch import optim

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class PipelineWork:
    def run(
        self,
        datamodule,
        logger: Optional[Logger] = None,
        log_preprocessing: bool = False,
    ):
        """initiates preprocessing with .prepare_data()

        Note:
            a wandb run can be passed in for users who want to log intermediate preprocessing results.
            see https://docs.wandb.ai/guides/track/log
        """
        datamodule.prepare_data(logger=logger, log_preprocessing=log_preprocessing)


class TrainerWork:
    def __init__(
        self,
        logger: Optional[Logger] = None,
        trainer_init_kwargs: Optional[Dict[str, Any]] = {},
    ):
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._trainer = PodTrainer(logger=logger, **trainer_init_kwargs)

    def run(
        self,
        model,
        datamodule,
        fit: bool = True,
        validate: bool = False,
        test: bool = False,
        fit_kwargs: Optional[Dict[str, Any]] = {},
        val_kwargs: Optional[Dict[str, Any]] = {},
        test_kwargs: Optional[Dict[str, Any]] = {},
    ):
        """fit, validate, test

        Note:
            Validation can be set via Trainer flags, or called as Trainer method.
            See:
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#validation
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#check-val-every-n-epoch
        """
        if fit:
            self._trainer.fit(model=model, datamodule=datamodule, **fit_kwargs)
        if validate:
            self._trainer.validate(model=model, datamodule=datamodule, **val_kwargs)
        if test:
            self._trainer.test(ckpt_path="best", datamodule=datamodule, **test_kwargs)


class ObjectiveWork:
    def run(self, trial, trainer_work, model, datamodule):

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(optim, optimizer_name)(self._trainer.model.parameters(), lr=lr)

        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)]

        config = dict(trial.params)
        config["trial.number"] = trial.number

        trainer_work(
            model,
            datamodule,
            logger=trainer_work._trainer.logger(
                project=self.project_name, save_dir=self.wandb_dir, config=config, reinit=True
            ),
            model_params={"dropout": dropout, "optimizer": optimizer, "lr": lr},
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [
                    EarlyStopping(monitor="loss", mode="min"),
                    PyTorchLightningPruningCallback(trial, monitor="val_acc"),
                ],
            },
        )

        hyperparameters = dict(
            optimizer=optimizer_name,
            lr=lr,
            dropout=dropout,
            # n_layers=n_layers,
            # output_dims=output_dims,
        )

        trainer_work.trainer.logger.log_hyperparams(hyperparameters)

        trainer_work._trainer.run()

        return trainer_work._trainer.callback_metrics["val_acc"].item()


class TrialFlow:
    """
    Note:
        see:
         - https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_lightning_simple.py
         - https://github.com/nzw0301/optuna-wandb/blob/main/part-1/wandb_optuna.py
         - https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893
         - PyTorch with Optuna (by PyTorch) https://youtu.be/P6NwZVl8ttc
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        wandb_dir: Optional[str] = conf.WANDBPATH,
        study_name: Optional[str] = "lightning-pod",
        log_preprocessing: bool = False,
    ):
        # settings
        self.project_name = project_name
        self.wandb_dir = wandb_dir
        self.log_preprocesing = log_preprocessing
        self._model = PodModule
        self._datamodule = PodDataModule
        self._logger = WandbLogger
        # works
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._pipeline_work = PipelineWork()
        self._trainer_work = TrainerWork()
        self._objective_work = ObjectiveWork()
        # optuna study
        self._study = optuna.create_study(direction="maximize", study_name=study_name)

    def _set_artifact_path(self) -> None:
        """sets optuna log file
        Note:
            borrowed from Optuna
            see https://github.com/optuna/optuna/blob/fd841edc732124961113d1915ee8b7f750a0f04c/optuna/cli.py#L1026
        """

        root_logger = logging.getLogger("optuna")
        root_logger.setLevel(logging.DEBUG)

        full_artifact_path = os.path.join(conf.OPTUNAPATH, self.artifact_path)

        os.mkdir(full_artifact_path)

        file_handler = logging.FileHandler(filename=os.path.join(conf.OPTUNAPATH, full_artifact_path, "optuna.log"))
        file_handler.setFormatter(optuna.logging.create_default_formatter())
        root_logger.addHandler(file_handler)

    @property
    def artifact_path(self):
        """helps to sync wandb and optuna directory names for logs"""
        log_dir = self.wandb_settings.log_user or self.wandb_settings.log_internal

        if log_dir:
            log_dir = os.path.dirname(log_dir.replace(os.getcwd(), "."))

        return str(log_dir).split(os.sep)[-2]

    @property
    def trials(self):
        return self._study.trials

    @property
    def pruned_trial(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    @property
    def complete_trials(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    @property
    def best_trial(self):
        return self._study.best_trial

    @property
    def wandb_settings(self):
        return self._trainer_work._trainer.logger.experiment.settings

    def _display_report(self):
        """a rich table"""
        # TITLE
        table = Table(title="Study Statistics")
        # COLUMNS
        for col in ["Finished Trials", "Pruned Trials", "Completed Trials", "Best Trial"]:
            table.add_column(col)
        # ROW
        table.add_row(len(self.trials), len(self.pruned_trial), len(self.complete_trials), self.best_trial.value)
        # SHOW
        console = Console()
        console.print(table)

    def run(self, display_report: bool = False):
        self._pipeline_work.run(
            self._trainer_work._trainer.datamodule,
            logger=self._trainer_work._trainer.logger,
            log_preprocessing=self.log_preprocessing,
        )
        # set stage to fit since pipeline_work calls prepare_data
        self._trainer_work._trainer.datamodule.setup(stage="fit")
        self._set_artifact_path()
        self._study.optimize(self._objective_work.run, n_trials=10, timeout=600)
        if display_report:
            self._display_report()
        if issubclass(TrialFlow, LightningFlow):
            sys.exit()
