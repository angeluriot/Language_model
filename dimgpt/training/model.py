import torch
from torch import nn

from training.layers import *
from training.settings import *
from training import utils


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
