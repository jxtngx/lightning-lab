import lightning as L
import torch.nn.functional as F  # noqa: F401
import torchmetrics  # noqa: F401
from torch import nn, optim  # noqa: F401


class PodModule(L.LightningModule):
    """a custom PyTorch Lightning LightningModule"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def training_step(self, batch):
        pass

    def test_step(self, batch, *args):
        self._shared_eval(batch, "test")

    def validation_step(self, batch, *args):
        self._shared_eval(batch, "val")

    def _shared_eval(self, batch, prefix):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        pass
