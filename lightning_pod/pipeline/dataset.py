from torchvision.datasets import MNIST


class LitDataset(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
