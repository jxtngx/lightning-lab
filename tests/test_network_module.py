import torch
from lightning_pod.core.module import Encoder, Decoder
from lightning_pod.core.module import LitModel


def test_module_not_abstract():
    """
    example: https://github.com/PyTorchLightning/pytorch-lightning/blob/15fa5389387b3a220bc044dd30eb0be1e8f64944/tests/core/test_lightning_module.py#L29
    """
    _ = LitModel()


def test_module_forward():
    input_sample = torch.randn((1, 784))
    model = LitModel()
    output = model.forward(input_sample)
    assert output.shape == input_sample.shape


def test_module_training_step():
    input_sample = torch.randn((1, 784)), torch.randn((1, 784))
    model = LitModel()
    loss = model.training_step(input_sample)
    assert isinstance(loss, torch.Tensor)


def test_optimizer():
    model = LitModel()
    optimizer = model.configure_optimizers()
    optimizer_base_class = optimizer.__class__.__base__.__name__
    assert optimizer_base_class == "Optimizer"


def test_encoder_not_abstract():
    _ = Encoder()


def test_encoder_forward():
    input_sample = torch.randn((1, 784))
    model = Encoder()
    output = model.forward(input_sample)
    assert output.shape == torch.Size([1, 3])


def test_decoder_not_abstract():
    _ = Decoder()


def test_decoder_forward():
    input_sample = torch.randn((1, 3))
    model = Decoder()
    output = model.forward(input_sample)
    assert output.shape == torch.Size([1, 784])
