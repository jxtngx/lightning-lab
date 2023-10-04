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

import os
from pathlib import Path

import torch

from lightninglab.pipeline.datamodule import LabDataModule


def test_module_not_abstract():
    _ = LabDataModule()


def test_prepare_data():
    datamodule = LabDataModule()
    datamodule.prepare_data()
    networkpath = Path(__file__).parent
    projectpath = networkpath.parents[0]
    datapath = os.path.join(projectpath, "data", "cache")
    assert "LabDataset" in os.listdir(datapath)


def test_setup():
    datamodule = LabDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    data_keys = ["train_data", "test_data", "val_data"]
    assert all(key in dir(datamodule) for key in data_keys)


def test_trainloader():
    datamodule = LabDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.train_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)


def test_testloader():
    datamodule = LabDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.test_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)


def test_valloader():
    datamodule = LabDataModule()
    datamodule.prepare_data()
    datamodule.setup()
    loader = datamodule.val_dataloader()
    sample = loader.dataset[0][0]
    assert isinstance(sample, torch.Tensor)
