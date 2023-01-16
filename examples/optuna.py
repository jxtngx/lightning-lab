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
import wandb as wb
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from optuna.trial import TrialState
from rich.console import Console
from rich.table import Table

from lightning_pod import conf
from lightning_pod.core.trainer import LitTrainer
from lightning_pod.pipeline.datamodule import LitDataModule


class PipelineWorker:
    def __init__(self, wandb_run=None):
        """
        Note:
            a wandb run can be passed in for users who want to log intermediate preprocessing results.
            see https://docs.wandb.ai/guides/track/log
        """
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._datamodule = LitDataModule()
        self.experiment = wandb_run

    def run(self):
        """preprocessing"""
        self._datamodule.prepare_data()


class TrainerWorker:
    def __init__(
        self,
        logger,
        trainer_init_kwargs: Optional[Dict[str, Any]] = {},
        trainer_fit_kwargs: Optional[Dict[str, Any]] = {},
        trainer_val_kwargs: Optional[Dict[str, Any]] = {},
        trainer_test_kwargs: Optional[Dict[str, Any]] = {},
    ):
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._trainer = LitTrainer(logger=logger, **trainer_init_kwargs)
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
    def __init__(self):
        ...

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

    def display_report(self):
        # TITLE
        table = Table(title="Study Statistics")
        # COLUMNS
        table.add_column("Finished Trials")
        table.add_column("Pruned Trials")
        table.add_column("Completed Trials")
        table.add_column("Best Trial")
        # ROW
        table.add_row(
            len(self._study.trials),
            len(self.pruned_trial),
            len(self.complete_trials),
            self.best_trial.value,
        )
        # SHOW
        console = Console()
        console.print(table)

    def _run_trial(self):
        ...

    def run(self, display_report: bool = False):
        self._set_log_file()
        self._run_trial()
        if display_report:
            self.display_report()


class TrialFlow:
    """
    Note:
        see:
         - https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_simple.py
         - https://medium.com/optuna/optuna-meets-weights-and-biases-58fc6bab893
         - PyTorch with Optuna (by PyTorch) https://youtu.be/P6NwZVl8ttc
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        wandb_dir: Optional[str] = conf.WANDBPATH,
    ):
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._wb_run = wb.init(project=project_name, dir=wandb_dir)
        self._study = optuna.create_study(direction="maximize")
        self.pipeline_work = PipelineWorker()
        self.training_work = TrainerWorker(
            WandbLogger(experiment=self._wb_run),
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [EarlyStopping(monitor="loss", mode="min")],
            },
        )
        self.trial_work = TrialWorker()

    @property
    def pruned_trial(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    @property
    def complete_trials(self):
        return self._study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    @property
    def best_trial(self):
        return self._study.best_trial

    def run(self, display_report: bool = False):
        self.pipeline_work.run()
        # set stage to fit since pipeline_work calls prepare_data
        self.pipeline_work._datamodule.setup(stage="fit")
        # run trial
        self.trial_work.run(display_report)
        # stop Lightning App's loop after training is complete
        if issubclass(TrialFlow, LightningFlow):
            sys.exit()


if __name__ == "__main__":
    # to set a project name, use as
    # python3 examples/optuna.py some_project_name
    args = sys.argv
    if len(args) > 1:
        project_name = sys.argv[1]
    else:
        project_name = "lightingpod-examples-optuna"
    trial = TrialFlow(project_name=project_name)
    trial.run(display_report=True)
