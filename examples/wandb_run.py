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

import sys
from typing import Any, Dict, Optional

import wandb as wb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from lightning_pod import conf
from lightning_pod.core.trainer import LitTrainer
from lightning_pod.pipeline.datamodule import LitDataModule


class PipelineWorker:
    def __init__(self, datamodule, wandb_run=None):
        """
        Note:
            a wandb run can be passed in for users who want to log intermediate preprocessing results.
            see https://docs.wandb.ai/guides/track/log
        """
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._datamodule = datamodule()
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
    ):
        self._trainer = LitTrainer(logger=logger, **trainer_init_kwargs)
        self._datamodule = self._trainer.datamodule
        self._model = self._trainer.model
        self.fit_kwargs = trainer_fit_kwargs

    def run(self):
        """fit, validate, test"""
        self._trainer.fit(model=self._model, datamodule=self._datamodule, **self.fit_kwargs)


class SweepFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
        wandb_dir: Optional[str] = conf.WANDBPATH,
    ):
        self._wb_run = wb.init(project=project_name, dir=wandb_dir)
        self.pipeline_work = PipelineWorker(LitDataModule)
        self.training_work = TrainerWorker(
            WandbLogger(experiment=self._wb_run),
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [EarlyStopping(monitor="loss", mode="min")],
            },
        )

    def run(self):
        self.pipeline_work.run()
        self.pipeline_work._datamodule.setup(stage="fit")
        self.training_work.run()
        sys.exit()


if __name__ == "__main__":
    # to set a project name, use as
    # python3 examples/wandb_run.py some_project_name
    args = sys.argv
    if len(args) > 1:
        project_name = sys.argv[1]
    else:
        project_name = None
    sweep = SweepFlow(project_name=project_name)
    sweep.run()
