from typing import Any

import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor, nn, optim


class Encoder(nn.Module):
    """an encoder layer

    Args:
        None: no arguments are required at initialization.

    Returns:
        an encoded image.
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(
                in_features=28 * 28,
                out_features=64,
                bias=True,  # default
            ),
            nn.ReLU(
                inplace=False,  # default
            ),
            nn.Linear(
                in_features=64,
                out_features=3,
                bias=True,  # default
            ),
        )

    def forward(self, x: Any) -> Any:
        return self.l1(x)


class Decoder(nn.Module):
    """a decoder layer

    Args:
        None: no arguments are required at initialization.

    Returns:
        a decoded image.
    """

    def __init__(self):  # type: ignore[no-untyped-def]
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

    def forward(self, x: Any) -> Any:
        return self.l1(x)


class LitModel(pl.LightningModule):
    """a custom PyTorch Lightning LightningModule

    Args:
        None: no arguments are required at initialization.
    """

    def __init__(self):  # type: ignore[no-untyped-def]
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def training_step(self, batch: Tensor) -> STEP_OUTPUT:  # type: ignore[override]
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("loss", loss)
        return loss

    def test_step(self, batch: Any, *args: Any) -> STEP_OUTPUT:  # type: ignore[override]
        self._shared_eval(batch, "test")

    def validation_step(self, batch: Any, *args: Any) -> STEP_OUTPUT:  # type: ignore[override]
        self._shared_eval(batch, "val")

    def _shared_eval(self, batch: Any, prefix: str) -> STEP_OUTPUT:
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log(f"{prefix}_loss", loss)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
