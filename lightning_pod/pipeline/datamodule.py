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

import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from lightning.pytorch import LightningDataModule
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from lightning_pod import conf
from lightning_pod.pipeline.dataset import PodDataset

filepath = Path(__file__)
PROJECTPATH = os.getcwd()
NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class PodDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Any = PodDataset,
        data_dir: str = "data",
        split: bool = True,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        transforms: Callable = transforms.ToTensor(),
        batch_size: int = 64,
    ):
        super().__init__()
        self.data_dir = os.path.join(PROJECTPATH, data_dir, "cache")
        self.dataset = dataset
        self.split = split
        self.train_size = train_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.batch_size = batch_size

    def prepare_data(self, logger: Optional[Logger] = None, log_preprocessing: bool = False) -> None:
        self.dataset(self.data_dir, download=True)

    def setup(self, stage: Union[str, None] = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = self.dataset(self.data_dir, train=True, transform=self.transforms)
            train_size = int(len(full_dataset) * self.train_size)
            test_size = len(full_dataset) - train_size
            self.train_data, self.val_data = random_split(full_dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.transforms)

    def persist_splits(self):
        """saves all splits for reproducibility"""
        torch.save(self.train_data, os.path.join(conf.SPLITSPATH, "train.pt"))
        torch.save(self.val_data, os.path.join(conf.SPLITSPATH, "val.pt"))
        if not hasattr(self, "test_data"):
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.transforms)
        torch.save(self.test_data, os.path.join(conf.SPLITSPATH, "test.pt"))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_data, num_workers=self.num_workers, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.batch_size)
