import os, random
import torch
import numpy as np

from dimgpt.settings import *


# Text dataset class
class Dataset():

	def __init__(self, paths: list[tuple[str, float, float]]):

		paths, doc_sizes, self.ratios = zip(*paths)

		self.ratios = [self.ratios[i] / doc_sizes[i] for i in range(self.ratios)]
		total_sum = sum(self.ratios)
		self.ratios = [ratio / total_sum for ratio in self.ratios]

		self.datasets = [np.memmap(path, dtype = np.uint16, mode = 'r') for path in paths]


	def size(self) -> int:

		return sum([len(data) for data in self.datasets])


	def next(self) -> torch.Tensor:

		x = []
		y = []

		for _ in range(BATCH_SIZE):

			old_dataset_i = dataset_i
			dataset_i = np.random.choice(range(len(self.datasets)), p = self.ratios)
			start_i = random.randint(0, len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT)
			xy = self.datasets[dataset_i][start_i:start_i + MAX_CONTEXT + 1].tolist()
			i = 0

			while i < MAX_CONTEXT:

				if xy[i] == EOT_INDEX:

					dataset_i = np.random.choice(range(len(self.datasets)), p = self.ratios)

					if dataset_i == old_dataset_i:
						i += 1
						continue

					old_dataset_i = dataset_i
					start_i = random.randint(0, len(self.datasets[dataset_i]) - 2 - MAX_CONTEXT)
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

	cc100 = lambda split: (os.path.join(DATA_DIR, 'cc100', f'{split}.bin'), 7000)
	wikipedia = lambda split: (os.path.join(DATA_DIR, 'wikipedia', f'{split}.bin'), 7000)
	fr_instructs = lambda split: (os.path.join(DATA_DIR, 'fr_instructs', f'{split}.bin'), 1000)
	french_reddit = lambda split: (os.path.join(DATA_DIR, 'french_reddit', f'{split}.bin'), 1000)
	french_tweets = lambda split: (os.path.join(DATA_DIR, 'french_tweets', f'{split}.bin'), 100)

	train_dataset = Dataset(
		(*cc100('train'), 1000),
		(*wikipedia('train'), 100),
		(*fr_instructs('train'), 100),
		(*french_reddit('train'), 10),
		(*french_tweets('train'), 1)
	)

	val_dataset = Dataset(
		(*cc100('val'), 1000),
		(*wikipedia('val'), 100),
		(*fr_instructs('val'), 100),
		(*french_reddit('val'), 10),
		(*french_tweets('val'), 1)
	)

	val_datasets = [
		Dataset([(cc100('val'), 1)]),
		Dataset([(wikipedia('val'), 1)]),
		Dataset([(fr_instructs('val'), 1)]),
		Dataset([(french_reddit('val'), 1)]),
		Dataset([(french_tweets('val'), 1)]),
	]

	return train_dataset, val_dataset, val_datasets
