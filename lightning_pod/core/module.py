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

import lightning as L
import torch.nn.functional as F
from torch import nn, optim


class Encoder(nn.Module):
    """an encoder layer

    Args:
        None: no arguments are required at initialization.

    Returns:
        an encoded image.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(
                in_features=28 * 28,
                out_features=64,
                bias=True,
            ),
            nn.ReLU(
                inplace=False,
            ),
            nn.Linear(
                in_features=64,
                out_features=3,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    """a decoder layer

    Args:
        None: no arguments are required at initialization.

    Returns:
        a decoded image.
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(
                in_features=3,
                out_features=64,
                bias=True,
            ),
            nn.ReLU(
                inplace=False,
            ),
            nn.Linear(
                in_features=64,
                out_features=28 * 28,
                bias=True,
            ),
        )

    def forward(self, x):
        return self.l1(x)


class PodModule(L.LightningModule):
    """a custom PyTorch Lightning LightningModule

    Args:
        None: no arguments are required at initialization.
    """

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def training_step(self, batch):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)
        return loss

    def test_step(self, batch, *args):
        self._shared_eval(batch, "test")

    def validation_step(self, batch, *args):
        self._shared_eval(batch, "val")

    def _shared_eval(self, batch, prefix):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{prefix}_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
