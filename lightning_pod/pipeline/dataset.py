from typing import Any

from torchvision.datasets import MNIST


class LitDataset(MNIST):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
