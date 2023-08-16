import os, random
import torch
import numpy as np

from dimgpt.settings import *


# Text dataset class
class Dataset():

	def __init__(self, paths: list[str], **kwargs):

		super().__init__(**kwargs)

		self.datas = [np.memmap(path, dtype = np.uint16, mode = 'r') for path in paths]

		self.sizes = np.array([len(data) for data in self.datas], dtype = np.float32)
		self.sizes /= np.sum(self.sizes)


	def size(self) -> int:

		return sum([len(data) for data in self.datas])


	def next(self) -> torch.Tensor:

		output = torch.zeros((BATCH_SIZE, MAX_CONTEXT), dtype = torch.long)

		for i in range(BATCH_SIZE):
			dataset_i = np.random.choice(range(len(self.datas)), p = self.sizes)
			example_i = random.randint(0, len(self.datas[dataset_i]) - 1 - MAX_CONTEXT)
			example = self.datas[dataset_i][example_i:example_i + MAX_CONTEXT]
			example = np.array(example, dtype = np.int32)
			output[i] = torch.tensor(example, dtype = torch.long)

		return output.to(DEVICE).detach()


# Import pretraining datasets
def import_pretrain_datasets() -> tuple[Dataset]:

	train_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_train.bin'), os.path.join(DATA_DIR, 'wikipedia_train.bin')])
	cc100_val_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_val.bin')])
	wikipedia_val_dataset = Dataset([os.path.join(DATA_DIR, 'wikipedia_val.bin')])
	french_reddit_val_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_val.bin')])

	return train_dataset, cc100_val_dataset, wikipedia_val_dataset, french_reddit_val_dataset


# Import finetuning datasets
def import_finetune_datasets() -> tuple[Dataset]:

	train_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_train.bin')])
	cc100_val_dataset = Dataset([os.path.join(DATA_DIR, 'cc100_val.bin')])
	wikipedia_val_dataset = Dataset([os.path.join(DATA_DIR, 'wikipedia_val.bin')])
	french_reddit_val_dataset = Dataset([os.path.join(DATA_DIR, 'french_reddit_val.bin')])

	return train_dataset, cc100_val_dataset, wikipedia_val_dataset, french_reddit_val_dataset
