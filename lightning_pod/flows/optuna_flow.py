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
import sys
from typing import Any, Dict, Optional

import optuna
import wandb  # NOQA
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import Logger, WandbLogger
from optuna.trial import TrialState
from rich.console import Console
from rich.table import Table
from torch import optim

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class PipelineWorker:
    def run(self, datamodule, logger: Optional[Logger] = None, log_preprocessing: bool = False):
        """initiates preprocessing with .prepare_data()

        Note:
            a wandb run can be passed in for users who want to log intermediate preprocessing results.
            see https://docs.wandb.ai/guides/track/log
        """
        datamodule.prepare_data(logger=logger, log_preprocessing=log_preprocessing)


class TrainerWorker:
    def __init__(
        self,
        model,
        datamodule,
        logger,
        trainer_init_kwargs: Optional[Dict[str, Any]] = {},
        trainer_fit_kwargs: Optional[Dict[str, Any]] = {},
        trainer_val_kwargs: Optional[Dict[str, Any]] = {},
        trainer_test_kwargs: Optional[Dict[str, Any]] = {},
    ):
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._trainer = PodTrainer(model, datamodule, logger=logger, **trainer_init_kwargs)
        self.fit_kwargs = trainer_fit_kwargs
        self.val_kwargs = trainer_val_kwargs
        self.test_kwargs = trainer_test_kwargs

    def run(self, fit: bool = True, validate: bool = False, test: bool = False):
        """fit, validate, test

        Note:
            Validation can be set via Trainer flags, or called as Trainer method.
            See:
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#validation
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#check-val-every-n-epoch
        """
        if fit:
            self._trainer.fit(model=self._trainer.model, datamodule=self._trainer.datamodule, **self.fit_kwargs)
        if validate:
            self._trainer.validate(model=self._trainer.model, datamodule=self._trainer.datamodule, **self.val_kwargs)
        if test:
            self._trainer.test(ckpt_path="best", datamodule=self._trainer.datamodule, **self.test_kwargs)


class TrialWorker:
    def __init__(self, trainer: PodTrainer, study: optuna.study.Study):
        self._trainer = trainer
        self._study = study

    @staticmethod
    def _set_log_file() -> None:
        """sets optuna log file
        Note:
            borrowed from Optuna
            see https://github.com/optuna/optuna/blob/fd841edc732124961113d1915ee8b7f750a0f04c/optuna/cli.py#L1026
        """

        root_logger = logging.getLogger("optuna")
        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(filename=conf.OPTUNAPATH)
        file_handler.setFormatter(optuna.logging.create_default_formatter())
        root_logger.addHandler(file_handler)

    @property
    def pruned_trial(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    @property
    def complete_trials(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    @property
    def best_trial(self):
        return self._study.best_trial

    def display_report(self):
        # TITLE
        table = Table(title="Study Statistics")
        # COLUMNS
        for col in ["Finished Trials", "Pruned Trials", "Completed Trials", "Best Trial"]:
            table.add_column(col)
        # ROW
        table.add_row(len(self._study.trials), len(self.pruned_trial), len(self.complete_trials), self.best_trial.value)
        # SHOW
        console = Console()
        console.print(table)

    def _objective(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer = getattr(optim, optimizer_name)(self._trainer.model.parameters(), lr=lr)
        config = dict(trial.params)
        config["trial.number"] = trial.number
        self._trainer.optimizers = optimizer
        self._trainer.logger.experiment.config = config
        self._trainer.run()

    def run(self, display_report: bool = False):
        self._set_log_file()
        self._study.optimize(self._objective, n_trials=10, timeout=600)
        if display_report:
            self.display_report()


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
    ):
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._pipeline_work = PipelineWorker()
        self._trainer_work = TrainerWorker(
            PodModule,
            PodDataModule,
            WandbLogger(project=project_name, save_dir=wandb_dir),
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [EarlyStopping(monitor="loss", mode="min")],
            },
        )

        self._study = optuna.create_study(direction="maximize", study_name=study_name)
        self._trial_work = TrialWorker(self._trainer_work, self._study)

    def run(self, display_report: bool = False):
        self._pipeline_work.run(
            self._trainer_work.datamodule, logger=self._trainer_work.logger, log_preprocessing=False
        )
        # set stage to fit since pipeline_work calls prepare_data
        self._pipeline_work._datamodule.setup(stage="fit")
        # run trial
        self._trial_work.run(display_report)
        # stop Lightning App's loop after training is complete
        if issubclass(TrialFlow, LightningFlow):
            sys.exit()
