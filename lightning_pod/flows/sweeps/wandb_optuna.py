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
from typing import Any, Dict, List, Optional

import optuna
import wandb
from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from optuna.trial import FrozenTrial, Trial, TrialState
from rich.console import Console
from rich.table import Table
from torch import optim

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class ObjectiveWork:
    def __init__(self, project_name: str, wandb_save_dir: str, log_preprocessing: bool):
        self.project_name = project_name
        self.wandb_save_dir = wandb_save_dir
        self.log_preprocessing = log_preprocessing
        self.datamodule = PodDataModule()

    def _set_artifact_dir(self) -> None:
        """sets optuna log file
        Note:
            borrowed from Optuna
            see https://github.com/optuna/optuna/blob/fd841edc732124961113d1915ee8b7f750a0f04c/optuna/cli.py#L1026
        """

        root_logger = logging.getLogger("optuna")
        root_logger.setLevel(logging.DEBUG)

        artifact_dir = os.path.join(conf.OPTUNAPATH, self.artifact_path)

        if not os.path.isdir(artifact_dir):
            os.mkdir(artifact_dir)

        file_handler = logging.FileHandler(filename=os.path.join(conf.OPTUNAPATH, artifact_dir, "optuna.log"))
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

    def persist_model(self):
        input_sample = self.trainer.datamodule.train_data.dataset[0][0]
        self.trainer.model.to_onnx(conf.MODELPATH, input_sample=input_sample, export_params=True)

    def persist_predictions(self):
        self.trainer.persist_predictions()

    def persist_splits(self):
        self.trainer.datamodule.persist_splits()

    def _objective(self, trial: Trial) -> float:

        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        optimizer = getattr(optim, optimizer_name)
        dropout = trial.suggest_float("dropout", 0.2, 0.5)

        model = PodModule(dropout=dropout, optimizer=optimizer, lr=lr)

        config = dict(trial.params)
        config["trial.number"] = trial.number

        trainer_init_kwargs = {
            "max_epochs": 10,
            "callbacks": [
                EarlyStopping(monitor="training_loss", mode="min"),
            ],
        }

        if trial.number == 0:
            self.trial_group = wandb.util.generate_id()

        if hasattr(self, "trainer"):
            # stops previous wandb run so that a new run can be initialized on new trial
            # also helps to avoid hanging process in wandb sdk
            # i.e. if this is called after self.trainer.fit,
            # a key error is encountered in wandb.sdk on the final trial
            # and wandb does not finish, and does not return control to TrialFlow
            self.trainer.logger.experiment.finish()

        self.trainer = PodTrainer(
            logger=WandbLogger(
                project=self.project_name,
                name="-".join(["trial", str(trial.number)]),
                group=self.trial_group,
                save_dir=self.wandb_save_dir,
                config=config,
            ),
            **trainer_init_kwargs,
        )

        # set optuna logs dir
        self._set_artifact_dir()
        # logs hyperparameters to logs/wandb_logs/wandb/{run_name}/files/config.yaml
        hyperparameters = dict(optimizer=optimizer_name, lr=lr, dropout=dropout)
        self.trainer.logger.log_hyperparams(hyperparameters)

        self.trainer.fit(model=model, datamodule=self.datamodule)

        return self.trainer.callback_metrics["val_acc"].item()

    def run(self, trial: Trial) -> float:

        if trial.number == 0:
            self.datamodule.prepare_data()

        self.datamodule.setup(stage="fit")

        return self._objective(trial)


class SweepFlow:
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
        # _ helps to avoid LightningFlow from checking JSON serialization if converting to Lightning App
        self._objective_work = ObjectiveWork(self.project_name, self.wandb_dir, self.log_preprocessing)
        # optuna study
        self._study = optuna.create_study(direction="maximize", study_name=study_name)

    @property
    def trials(self) -> List[FrozenTrial]:
        return self._study.trials

    @property
    def pruned_trial(self) -> List[FrozenTrial]:
        return self._study.get_trials(deepcopy=False, states=[TrialState.PRUNED])

    @property
    def complete_trials(self) -> List[FrozenTrial]:
        return self._study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    @property
    def best_trial(self) -> FrozenTrial:
        return self._study.best_trial

    @staticmethod
    def _display_report(trial_metric_names: List[str], trial_info: List[str]) -> None:

        table = Table(title="Study Statistics")

        for col in trial_metric_names:
            table.add_column(col, header_style="cyan")

        table.add_row(*trial_info)

        console = Console()
        console.print(table)

    def run(
        self,
        display_report: bool = True,
        persist_model: bool = True,
        persist_predictions: bool = True,
        persist_splits: bool = True,
    ) -> None:
        self._study.optimize(self._objective_work.run, n_trials=10, timeout=600)
        if display_report:
            trial_metric_names = ["Finished Trials", "Pruned Trials", "Completed Trials", "Best Trial"]
            trial_info = [len(self.trials), len(self.pruned_trial), len(self.complete_trials), self.best_trial.value]
            trial_info = [str(i) for i in trial_info]
            self._display_report(trial_metric_names, trial_info)
        if persist_model:
            self._objective_work.persist_model()
        if persist_predictions:
            self._objective_work.persist_predictions()
        if persist_splits:
            self._objective_work.persist_splits()
        if issubclass(SweepFlow, LightningFlow):
            sys.exit()
