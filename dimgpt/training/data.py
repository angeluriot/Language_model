import os, random
import torch
import numpy as np

from dimgpt.settings import *


# Text dataset class
class Dataset():

	def __init__(self, paths: list[str], **kwargs):

		super().__init__(**kwargs)

		self.datasets = [np.memmap(path, dtype = np.uint16, mode = 'r') for path in paths]

		self.sizes = np.array([len(dataset) for dataset in self.datasets], dtype = np.float32)
		self.sizes /= np.sum(self.sizes)


	def size(self) -> int:

		return sum([len(data) for data in self.datasets])


	def next(self) -> torch.Tensor:

		x = []
		y = []

		for _ in range(BATCH_SIZE):

			dataset_i = np.random.choice(range(len(self.datasets)), p = self.sizes)
			i = random.randint(0, len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT)

			x.append(torch.from_numpy(self.datasets[dataset_i][i:i + MAX_CONTEXT].astype(np.int64)))
			y.append(torch.from_numpy(self.datasets[dataset_i][i + 1:i + 1 + MAX_CONTEXT].astype(np.int64)))

		x = torch.stack(x).pin_memory().to(DEVICE, non_blocking = True)
		y = torch.stack(y).pin_memory().to(DEVICE, non_blocking = True)

		return x, y


# Import pretraining datasets
def import_pretrain_datasets() -> tuple[Dataset]:

	train_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_train.bin'), os.path.join(DATA_DIR, 'wikipedia_train.bin')])
	val_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_val.bin'), os.path.join(DATA_DIR, 'wikipedia_val.bin')])
	cc100_val_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_val.bin')])
	wikipedia_val_dataset = Dataset([os.path.join(DATA_DIR, 'wikipedia_val.bin')])
	french_reddit_val_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_val.bin')])

	return train_dataset, val_dataset, cc100_val_dataset, wikipedia_val_dataset, french_reddit_val_dataset


# Import finetuning datasets
def import_finetune_datasets() -> tuple[Dataset]:

	train_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_train.bin')])
	val_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_val.bin')])
	cc100_val_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_val.bin')])
	wikipedia_val_dataset = Dataset([os.path.join(DATA_DIR, 'wikipedia_val.bin')])

	return train_dataset, val_dataset, cc100_val_dataset, wikipedia_val_dataset
