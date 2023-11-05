import os, random
import torch
import numpy as np

from dimgpt.settings import *


# Text dataset class
class Dataset():

	def __init__(self, paths: list[tuple[str, float, float]]):

		paths, splits, self.ratios = zip(*paths)

		self.datasets = [np.memmap(os.path.join(DATA_DIR, paths[i], splits[i] + '.bin'), dtype = np.uint16, mode = 'r') for i in range(len(paths))]
		self.ids = [np.memmap(os.path.join(DATA_DIR, paths[i], splits[i] + '_ids.bin'), dtype = np.uint64, mode = 'r') for i in range(len(paths))]
		self.ratios = (np.array(self.ratios) / np.sum(self.ratios)).tolist()


	def size(self) -> int:

		return sum([len(data) for data in self.datasets])


	def next(self) -> tuple[torch.Tensor, torch.Tensor]:

		x = []
		y = []

		for _ in range(BATCH_SIZE):

			dataset_i = np.random.choice(range(len(self.datasets)), p = self.ratios)
			start_i = random.randint(0, len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT)
			xy = self.datasets[dataset_i][start_i:start_i + MAX_CONTEXT + 1].tolist()
			old_dataset_i = dataset_i
			i = 0

			while i < MAX_CONTEXT:

				if xy[i] == EOT_INDEX:

					dataset_i = np.random.choice(range(len(self.datasets)), p = self.ratios)

					if dataset_i == old_dataset_i:
						i += 1
						continue

					old_dataset_i = dataset_i
					start_i = int(self.ids[dataset_i][random.randint(0, len(self.ids[dataset_i]) - 1)])

					while start_i > len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT + i:
						start_i = int(self.ids[dataset_i][random.randint(0, len(self.ids[dataset_i]) - 1)])

					xy = xy[0: i + 1]
					xy.extend(self.datasets[dataset_i][start_i:start_i + MAX_CONTEXT - i].tolist())

				i += 1

			x.append(torch.tensor(xy[0:MAX_CONTEXT], dtype = torch.long))
			y.append(torch.tensor(xy[1:MAX_CONTEXT + 1], dtype = torch.long))

		x = torch.stack(x).pin_memory().to(DEVICE, non_blocking = True)
		y = torch.stack(y).pin_memory().to(DEVICE, non_blocking = True)

		return x, y


# Import datasets
def import_datasets() -> tuple[Dataset, Dataset, list[Dataset]]:

	train_dataset = Dataset([
		('cc100', 'train',			1000	/ 183),
		('wikipedia', 'train',		100		/ 184),
		('fr_instructs', 'train',	100		/ 44),
		('french_reddit', 'train',	10		/ 79),
		('french_tweets', 'train',	1		/ 11)
	])

	val_dataset = Dataset([
		('cc100', 'val',			1000	/ 183),
		('wikipedia', 'val',		100		/ 184),
		('fr_instructs', 'val',		100		/ 44),
		('french_reddit', 'val',	10		/ 79),
		('french_tweets', 'val',	1		/ 11)
	])

	val_datasets = [
		Dataset([('cc100', 'val', 1)]),
		Dataset([('wikipedia', 'val', 1)]),
		Dataset([('fr_instructs', 'val', 1)]),
		Dataset([('french_reddit', 'val', 1)]),
		Dataset([('french_tweets', 'val', 1)])
	]

	return train_dataset, val_dataset, val_datasets
