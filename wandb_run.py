import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Callable, Union

import lightning as L
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import wandb as wb
from lightning_pod.core.module import LitModel
from lightning_pod.pipeline.dataset import LitDataset

filepath = Path(__file__)
PROJECTPATH = os.getcwd()
NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class LitDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any = LitDataset,
        data_dir: str = "data",
        split: bool = True,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        transforms: Callable = transforms.ToTensor(),
        experiment=None,
    ):
        super().__init__()
        self.data_dir = os.path.join(PROJECTPATH, data_dir, "cache")
        self.dataset = dataset
        self.split = split
        self.train_size = train_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.experiment = experiment

    def prepare_data(self) -> None:
        self.dataset(self.data_dir, download=True)
        ds = self.dataset(self.data_dir, train=False, transform=self.transforms)
        self.experiment.log({"image": wb.Image(ds[0][0])})

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = self.dataset(self.data_dir, train=True, transform=self.transforms)
            train_size = int(len(full_dataset) * self.train_size)
            test_size = len(full_dataset) - train_size
            self.train_data, self.val_data = random_split(full_dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, num_workers=self.num_workers)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, num_workers=self.num_workers)


class PipelineWorker:
    def __init__(self, litdatamodule):
        self._wb_run = wb.init()  # _ prevents flow from checking JSON serialization
        self._datamodule = litdatamodule(experiment=self._wb_run)

    def run(self):
        """do preprocessing and logging"""
        self._datamodule.prepare_data()


class TrainingWorker:
    def __init__(self, model, datamodule, logger):
        chkpt_dir = os.path.join(PROJECTPATH, "models", "checkpoints")
        self._trainer = L.Trainer(
            logger=logger,
            max_epochs=10,
            callbacks=[EarlyStopping(monitor="loss", mode="min"), ModelCheckpoint(dirpath=chkpt_dir, filename="model")],
        )
        self._datamodule = datamodule
        self._model = model

    def run(self):
        # fit will be blocking
        self._trainer.fit(model=self._model, datamodule=self._datamodule)


class SweepFlow:
    def __init__(self):
        self.pipeline_work = PipelineWorker(LitDataModule)
        self.training_work = TrainingWorker(
            LitModel(),
            self.pipeline_work._datamodule,
            WandbLogger(experiment=self.pipeline_work._wb_run, project="lightning-pod"),
        )

    def run(self):
        self.pipeline_work.run()
        self.pipeline_work._datamodule.setup(stage="fit")
        # assume fit is indeed blocking and exit after complete
        self.training_work.run()
        sys.exit()


if __name__ == "__main__":
    sweep = SweepFlow()
    sweep.run()
