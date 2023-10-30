import os, random
import torch
import numpy as np

from dimgpt.settings import *


# Text dataset class
class Dataset():

	def __init__(self, paths: list[str] | list[tuple[str, float]]):

		if len(paths) == 1:
			self.ratios = [1.0]
		else:
			paths, self.ratios = zip(*paths)
			total_sum = sum(self.ratios)
			self.ratios = [ratio / total_sum for ratio in self.ratios]

		self.datasets = [np.memmap(path, dtype = np.uint16, mode = 'r') for path in paths]


	def size(self) -> int:

		return sum([len(data) for data in self.datasets])


	def next(self) -> torch.Tensor:

		x = []
		y = []

		for _ in range(BATCH_SIZE):

			dataset_i = np.random.choice(range(len(self.datasets)), p = self.ratios)
			i = random.randint(0, len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT)

			x.append(torch.from_numpy(self.datasets[dataset_i][i:i + MAX_CONTEXT].astype(np.int64)))
			y.append(torch.from_numpy(self.datasets[dataset_i][i + 1:i + 1 + MAX_CONTEXT].astype(np.int64)))

		x = torch.stack(x).pin_memory().to(DEVICE, non_blocking = True)
		y = torch.stack(y).pin_memory().to(DEVICE, non_blocking = True)

		return x, y


# Import pretraining datasets
def import_pretrain_datasets() -> tuple[Dataset, Dataset, list[Dataset]]:

	train_dataset = Dataset([
		(os.path.join(DATA_DIR, 'cc100', 'train.bin'), 1000),
		(os.path.join(DATA_DIR, 'wikipedia', 'train.bin'), 100),
		(os.path.join(DATA_DIR, 'fr_instructs', 'train.bin'), 100),
		(os.path.join(DATA_DIR, 'french_reddit', 'train.bin'), 10),
		(os.path.join(DATA_DIR, 'french_tweets', 'train.bin'), 1)
	])

	val_dataset = Dataset([
		(os.path.join(DATA_DIR, 'cc100', 'val.bin'), 1000),
		(os.path.join(DATA_DIR, 'wikipedia', 'val.bin'), 100),
		(os.path.join(DATA_DIR, 'fr_instructs', 'val.bin'), 100),
		(os.path.join(DATA_DIR, 'french_reddit', 'val.bin'), 10),
		(os.path.join(DATA_DIR, 'french_tweets', 'val.bin'), 1)
	])

	val_datasets = [
		Dataset([os.path.join(DATA_DIR, 'cc100', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'wikipedia', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'fr_instructs', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'french_reddit', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'french_tweets', 'val.bin')]),
	]

	return train_dataset, val_dataset, val_datasets


# Import finetuning datasets
def import_finetune_datasets() -> tuple[Dataset, Dataset, list[Dataset]]:

	train_dataset = Dataset([
		(os.path.join(DATA_DIR, 'cc100', 'train.bin'), 10),
		(os.path.join(DATA_DIR, 'wikipedia', 'train.bin'), 10),
		(os.path.join(DATA_DIR, 'fr_instructs', 'train.bin'), 100),
		(os.path.join(DATA_DIR, 'french_reddit', 'train.bin'), 100),
		(os.path.join(DATA_DIR, 'french_tweets', 'train.bin'), 1)
	])

	val_dataset = Dataset([
		(os.path.join(DATA_DIR, 'cc100', 'val.bin'), 10),
		(os.path.join(DATA_DIR, 'wikipedia', 'val.bin'), 10),
		(os.path.join(DATA_DIR, 'fr_instructs', 'val.bin'), 100),
		(os.path.join(DATA_DIR, 'french_reddit', 'val.bin'), 100),
		(os.path.join(DATA_DIR, 'french_tweets', 'val.bin'), 1)
	])

	val_datasets = [
		Dataset([os.path.join(DATA_DIR, 'cc100', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'wikipedia', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'fr_instructs', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'french_reddit', 'val.bin')]),
		Dataset([os.path.join(DATA_DIR, 'french_tweets', 'val.bin')]),
	]

	return train_dataset, val_dataset, val_datasets
