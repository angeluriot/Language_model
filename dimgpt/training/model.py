import torch
from torch import nn

from dimgpt.training.layers import *
from dimgpt.settings import *
from dimgpt import utils


# Model block
class Block(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def forward(self, x: torch.Tensor, images: torch.Tensor) -> torch.Tensor:

		return x


# Model
class Model(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		return x
