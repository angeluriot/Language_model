import random
import numpy as np
from keras.utils import *
from settings import *


class BatchGenerator(Sequence):

	def __init__(self, dataset, indexes, val_frequency = None, ratio = 1.0):

		self.dataset = dataset
		self.indexes = np.copy(indexes)
		self.val_frequency = val_frequency
		self.ratio = ratio
		self.epoch = 0
		random.shuffle(self.indexes)


	def __len__(self):

		size = len(self.indexes) // BATCH_SIZE

		if self.val_frequency is not None:
			size //= self.val_frequency

		if self.ratio < 1.0:
			size = int(size * self.ratio)

		return size


	def __getitem__(self, idx):

		x = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.int32)
		y = np.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = np.int32)

		if self.ratio < 1.0:

			for i in range(BATCH_SIZE):
				start = self.indexes[random.randint(0, len(self.indexes) - 1)]
				x[i, :] = self.dataset[start: start + MAX_CONTEXT]
				y[i, :] = self.dataset[start + 1: start + MAX_CONTEXT + 1]

			return x, y

		if self.val_frequency is not None:
			start = self.epoch * (len(self.indexes) // self.val_frequency)
		else:
			start = 0

		indexes = self.indexes[start + idx * BATCH_SIZE:start + (idx + 1) * BATCH_SIZE]

		for i in range(len(indexes)):
			x[i, :] = self.dataset[indexes[i]:indexes[i] + MAX_CONTEXT]
			y[i, :] = self.dataset[indexes[i] + 1:indexes[i] + MAX_CONTEXT + 1]

		return x, y


	def on_epoch_end(self):

		self.epoch += 1

		if self.val_frequency is not None and self.epoch >= self.val_frequency:
			random.shuffle(self.indexes)
			self.epoch = 0
