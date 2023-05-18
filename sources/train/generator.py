import random
import numpy as np
import numpy.typing as npt
from keras.utils import Sequence

from settings import *


class BatchGenerator(Sequence):

	def __init__(self, dataset: npt.NDArray[np.uint16], indexes: npt.NDArray[np.uint64], size: int):

		self.dataset = dataset
		self.indexes = indexes
		self.size = size


	def __len__(self):

		return self.size


	def __getitem__(self, idx: int) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:

		x = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.uint16)
		y = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.uint16)

		for i in range(BATCH_SIZE):
			start = int(self.indexes[random.randint(0, len(self.indexes) - 1)])
			x[i, :] = self.dataset[start: start + MAX_CONTEXT]
			y[i, :] = self.dataset[start + 1: start + MAX_CONTEXT + 1]

		return x, y
