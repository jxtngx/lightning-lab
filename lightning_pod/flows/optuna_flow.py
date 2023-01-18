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
from lightning.pytorch.loggers import WandbLogger
from optuna.trial import TrialState
from rich.console import Console
from rich.table import Table
from torch import optim

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class ObjectiveWork:
    def __init__(self, project_name, wandb_save_dir, log_preprocessing):
        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.log_preprocessing = log_preprocessing
        self.datamodule = PodDataModule()
        self.prep_and_setup_data = False

    def _set_artifact_path(self) -> None:
        """sets optuna log file
        Note:
            borrowed from Optuna
            see https://github.com/optuna/optuna/blob/fd841edc732124961113d1915ee8b7f750a0f04c/optuna/cli.py#L1026
        """

        root_logger = logging.getLogger("optuna")
        root_logger.setLevel(logging.DEBUG)

        full_artifact_path = os.path.join(conf.OPTUNAPATH, self.artifact_path)

        if not os.path.isdir(full_artifact_path):
            os.mkdir(full_artifact_path)

        file_handler = logging.FileHandler(filename=os.path.join(conf.OPTUNAPATH, full_artifact_path, "optuna.log"))
        file_handler.setFormatter(optuna.logging.create_default_formatter())
        root_logger.addHandler(file_handler)

    @property
    def artifact_path(self) -> str:
        """helps to sync wandb and optuna directory names for logs"""
        log_dir = self.wandb_settings.log_user or self.wandb_settings.log_internal

        if log_dir:
            log_dir = os.path.dirname(log_dir.replace(os.getcwd(), "."))

        return str(log_dir).split(os.sep)[-2]

    @property
    def wandb_settings(self) -> Dict[str, Any]:
        return self.trainer.logger.experiment.settings

    def _objective(self, trial):

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(optim, optimizer_name)

        dropout = trial.suggest_float("dropout", 0.2, 0.5)
        # n_layers = trial.suggest_int("n_layers", 1, 3)
        # output_dims = [trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)]

        config = dict(trial.params)
        config["trial.number"] = trial.number

        model = PodModule(dropout=dropout, optimizer=optimizer, lr=lr)

        trainer_init_kwargs = {
            "max_epochs": 10,
            "callbacks": [
                EarlyStopping(monitor="training_loss", mode="min"),
            ],
        }

        self.trainer = PodTrainer(
            logger=WandbLogger(
                project=self.project_name,
                name="-".join(["Trial", str(trial.number)]),
                save_dir=self.wandb_save_dir,
                config=config,
            ),
            **trainer_init_kwargs,
        )

        self._set_artifact_path()

        hyperparameters = dict(optimizer=optimizer_name, lr=lr, dropout=dropout)

        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        self.trainer.logger.experiment.finish()

        return self.trainer.callback_metrics["val_acc"].item()

    def run(self, trial) -> float:

        if not self.prep_and_setup_data:
            self.datamodule.prepare_data()
            self.datamodule.setup(stage="fit")
            self.prep_and_setup_data = True

        return self._objective(trial)


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
    ) -> None:
        # settings
        self.project_name = project_name
        self.wandb_dir = wandb_dir
        self.log_preprocessing = log_preprocessing
        # work
        self._objective_work = ObjectiveWork(self.project_name, self.wandb_dir, self.log_preprocessing)

        # optuna study
        self._study = optuna.create_study(direction="maximize", study_name=study_name)

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

    def _display_report(self) -> None:
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

    def run(self, display_report: bool = False) -> None:
        self._study.optimize(self._objective_work.run, n_trials=10, timeout=600)
        if display_report:
            self._display_report()
        if issubclass(TrialFlow, LightningFlow):
            sys.exit()
