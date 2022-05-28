import os
import torch
from pathlib import Path
from lightning_pod.pipeline.datamodule import LitDataModule


def test_module_not_abstract():
    _ = LitDataModule()


def test_prepare_data():
    data_module = LitDataModule()
    data_module.prepare_data()
    networkpath = Path(__file__).parent
    projectpath = networkpath.parents[0]
    datapath = os.path.join(projectpath, "data", "cache")
    assert "LitDataset" in os.listdir(datapath)


def test_setup():
    data_module = LitDataModule()
    data_module.prepare_data()
    data_module.setup()
    data_keys = ["train_data", "test_data", "val_data"]
    assert all(key in dir(data_module) for key in data_keys)


def test_trainloader():
    data_module = LitDataModule()
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)


def test_testloader():
    data_module = LitDataModule()
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.test_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)


def test_valloader():
    data_module = LitDataModule()
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.val_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)
