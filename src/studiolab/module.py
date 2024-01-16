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

import pytorch_lightning as pl
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import optim  # noqa: F401


class LabModule(pl.LightningModule):
    """a custom PyTorch Lightning LightningModule"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch):
        pass

    def test_step(self, batch, *args):
        pass

    def validation_step(self, batch, *args):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        pass
