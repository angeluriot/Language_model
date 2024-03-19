from abc import ABC
import torch

from dimgpt.data.tokenizer import Tokenizer
from dimgpt.settings import *


class Dataset(ABC):

	def __init__(self, tokenizer: Tokenizer):

		self.tokenizer = tokenizer


	def train_size(self) -> int:

		pass


	def val_size(self) -> int:

		pass


	def _random_document(self, val: bool) -> tuple[list[int], list[int]]:

		pass


	def _get_tokens(self, val: bool) -> tuple[list[int], list[int]]:

		pass


	def _next(self, val: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

		x = []
		y = []
		strengths = []

		for _ in range(BATCH_SIZE):

			xy, strength = self._get_tokens(val)

			x.append(xy[0:MAX_CONTEXT])
			y.append(xy[1:MAX_CONTEXT + 1])
			strengths.append(strength[1:MAX_CONTEXT + 1])

		x = torch.tensor(x, dtype = torch.long).pin_memory().to(DEVICE, non_blocking = True)
		y = torch.tensor(y, dtype = torch.long).pin_memory().to(DEVICE, non_blocking = True)
		strengths = torch.tensor(strengths, dtype = torch.float32).pin_memory().to(DEVICE, non_blocking = True)

		return x, y, strengths


	def next_train(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

		return self._next(False)


	def next_val(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

		return self._next(True)