import random
import numpy as np
import numpy.typing as npt
from keras.utils import Sequence

from gpt.settings import *


class BatchGenerator(Sequence):

	def __init__(self, data: npt.NDArray[np.uint16], size: int):

		self.data = data
		self.size = size


	def __len__(self):

		return self.size


	def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:

		x = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.uint16)
		y = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.uint16)

		for i in range(BATCH_SIZE):
			start = random.randint(0, len(self.data) - MAX_CONTEXT - 2)
			x[i, :] = self.data[start:start + MAX_CONTEXT]
			y[i, :] = self.data[start + 1:start + MAX_CONTEXT + 1]

		return x, y
