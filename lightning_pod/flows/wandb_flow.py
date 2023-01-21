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

from lightning import LightningFlow
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.wandb_run import Run

from lightning_pod import conf
from lightning_pod.core.module import PodModule
from lightning_pod.core.trainer import PodTrainer
from lightning_pod.pipeline.datamodule import PodDataModule


class PipelineWorker:
    def run(self, datamodule: PodDataModule, experiment: Optional[Run] = None) -> None:
        """preprocessing"""
        datamodule.prepare_data()


class TrainerWorker:
    def __init__(
        self,
        logger,
        trainer_init_kwargs: Optional[Dict[str, Any]] = {},
        trainer_fit_kwargs: Optional[Dict[str, Any]] = {},
        trainer_val_kwargs: Optional[Dict[str, Any]] = {},
        trainer_test_kwargs: Optional[Dict[str, Any]] = {},
    ) -> None:
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._trainer = PodTrainer(logger=logger, **trainer_init_kwargs)
        self.fit_kwargs = trainer_fit_kwargs
        self.val_kwargs = trainer_val_kwargs
        self.test_kwargs = trainer_test_kwargs

    def run(
        self,
        model: PodModule,
        datamodule: PodDataModule,
        fit: bool = True,
        validate: bool = False,
        test: bool = False,
    ) -> None:
        """fit, validate, test

        Note:
            Validation can be set via Trainer flags, or called as Trainer method.
            See:
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#validation
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#val-check-interval
             - https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#check-val-every-n-epoch
        """
        if fit:
            self._trainer.fit(model=model, datamodule=datamodule, **self.fit_kwargs)
        if validate:
            self._trainer.validate(model=model, datamodule=datamodule, **self.val_kwargs)
        if test:
            self._trainer.test(ckpt_path="best", datamodule=datamodule, **self.test_kwargs)


class SweepFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
        wandb_dir: Optional[str] = conf.WANDBPATH,
    ) -> None:
        # _ prevents flow from checking JSON serialization if converting to Lightning App
        self._model = PodModule()
        self._datamodule = PodDataModule()
        self.pipeline_work = PipelineWorker()
        self.training_work = TrainerWorker(
            logger=WandbLogger(project=project_name, save_dir=wandb_dir),
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [EarlyStopping(monitor="loss", mode="min")],
            },
        )

    def run(self) -> None:
        self.pipeline_work.run(self._datamodule)
        #  set stage to fit since pipeline_work calls prepare_data
        self._datamodule.setup(stage="fit")
        self.training_work.run(self._model, self._datamodule)
        #  stop Lightning App's loop after training is complete
        if issubclass(SweepFlow, LightningFlow):
            sys.exit()
