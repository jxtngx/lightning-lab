import os
import sys
from typing import Any, Dict, Optional

import lightning as L
import wandb as wb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from lightning_pod.core.module import LitModel
from lightning_pod.pipeline.datamodule import LitDataModule


class PipelineWorker:
    def __init__(self, datamodule, wandb_run=None):
        """
        Note:
            a wandb run can be passed in for users who want to log intermediate preprocessing results.
            see https://docs.wandb.ai/guides/track/log
        """
        # _ prevents flow from checking JSON serialization if converting to App
        self._datamodule = datamodule()
        self.experiment = wandb_run

    def run(self):
        """preprocessing"""
        self._datamodule.prepare_data()


class TrainerWorker:
    def __init__(
        self,
        model,
        datamodule,
        logger,
        trainer_init_kwargs: Optional[Dict[str, Any]] = {},
        trainer_fit_kwargs: Optional[Dict[str, Any]] = {},
    ):
        self._trainer = L.Trainer(logger=logger, **trainer_init_kwargs)
        self._datamodule = datamodule
        self._model = model
        self.fit_kwargs = trainer_fit_kwargs

    def run(self):
        """fit, validate, test"""
        self._trainer.fit(model=self._model, datamodule=self._datamodule, **self.fit_kwargs)


class SweepFlow:
    def __init__(
        self,
        project_name: Optional[str] = None,
        wandb_dir: Optional[str] = os.path.join(os.getcwd(), "logs", "wandb_logs"),
    ):
        self._wb_run = wb.init(project=project_name, dir=wandb_dir)
        self.pipeline_work = PipelineWorker(LitDataModule())
        self.training_work = TrainerWorker(
            LitModel(),
            self.pipeline_work._datamodule,
            WandbLogger(experiment=self._wb_run),
            trainer_init_kwargs={
                "max_epochs": 10,
                "callbacks": [
                    EarlyStopping(monitor="loss", mode="min"),
                    ModelCheckpoint(dirpath=os.path.join(os.getcwd(), "models", "checkpoints"), filename="model"),
                ],
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
