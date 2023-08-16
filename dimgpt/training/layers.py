import math
import numpy as np
import torch
from torch import nn

from dimgpt.settings import *


# Base class for all layers
class Module(nn.Module):

	# Give the number of parameters of the module
	def nb_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])


	# Give the number of trainable parameters of the module
	def nb_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])


	# Give the number of non-trainable parameters of the module
	def nb_non_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])


	# Summarize the module
	def summary(self) -> None:

		print(f'Number of parameters: {self.nb_parameters():,}')
		print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
		print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')


	# Remove NaNs from the module gradients
	def clean_nan(self) -> None:

		for p in self.parameters():
			if p.grad is not None:
				torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)

