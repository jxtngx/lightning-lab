import multiprocessing
import os
from pathlib import Path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from lightninglab.pipeline.dataset import LabDataset

filepath = Path(__file__)
PROJECTPATH = os.getcwd()
NUMWORKERS = int(multiprocessing.cpu_count() // 2)


class LabDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset = LabDataset,
        data_dir: str = "data",
        split: bool = True,
        train_size: float = 0.8,
        num_workers: int = NUMWORKERS,
        transforms=None,
    ):
        super().__init__()
        self.data_dir = os.path.join(PROJECTPATH, data_dir, "cache")
        self.dataset = dataset
        self.split = split
        self.train_size = train_size
        self.num_workers = num_workers
        self.transforms = transforms

    def prepare_data(self):
        self.dataset(self.data_dir, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = self.dataset(self.data_dir, train=True, transform=self.transforms)
            train_size = int(len(full_dataset) * self.train_size)
            test_size = len(full_dataset) - train_size
            self.train_data, self.val_data = random_split(full_dataset, lengths=[train_size, test_size])
        if stage == "test" or stage is None:
            self.test_data = self.dataset(self.data_dir, train=False, transform=self.transforms)

    # def teardown(self):
    #     pass

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers)
