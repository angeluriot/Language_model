import random
import numpy as np
from keras.utils import *
from settings import *


class BatchGenerator(Sequence):

	def __init__(self, dataset, indexes, size):

		self.dataset = dataset
		self.indexes = indexes
		self.size = size


	def __len__(self):

		return self.size


	def __getitem__(self, idx):

		x = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.int32)
		y = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.int32)
		starts = np.random.choice(self.indexes, BATCH_SIZE)

		for i, start in enumerate(starts):
			start = self.indexes[random.randint(0, len(self.indexes) - 1)]
			x[i, :] = self.dataset[start: start + MAX_CONTEXT]
			y[i, :] = self.dataset[start + 1: start + MAX_CONTEXT + 1]

		return x, y
