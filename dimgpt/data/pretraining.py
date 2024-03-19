import os
import numpy as np
from tqdm import tqdm

from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer
from dimgpt.data.datasets.pretraining import *
from dimgpt.data.datasets import Dataset


class Pretraining:

	def __init__(self):

		self.datasets: list[Dataset] = [CommonCrawlDataset(), WikipediaDataset(), BooksDataset(), NewsDataset(), InstitutionsDataset(), OthersDataset()]


	def get_document(self) -> str:

		probabilities = np.array([dataset.size['train'] * dataset.multiplier for dataset in self.datasets])
		probabilities /= np.sum(probabilities)

		dataset = np.random.choice(self.datasets, p = probabilities)

		return dataset.get_document()


	def create_tokenizer_data(self, epsilon: float = 1e-8) -> tuple[list[int], list[str]]:

		if os.path.exists(os.path.join(DATA_DIR, 'tokenizer_data.txt')):

			return [0] * len(self.datasets), ['']

		target_ratios = np.array([dataset.size['train'] * dataset.multiplier for dataset in self.datasets])
		target_ratios = (target_ratios / np.sum(target_ratios)).tolist()

		with open(os.path.join(DATA_DIR, 'tokenizer_data.txt'), 'w', encoding = 'utf-8') as file:

			file.truncate(0)
			chars = {}
			current_sizes = [0] * len(self.datasets)
			pbar = tqdm(total = TOKENIZER_DATA_SIZE)

			while True:

				current_ratios = [size / (sum(current_sizes) + epsilon) for size in current_sizes]
				ratio_errors = [target_ratios[i] - current_ratios[i] for i in range(len(self.datasets))]
				dataset_index = np.argmax(ratio_errors)
				dataset = self.datasets[dataset_index]

				document = dataset.get_document()

				if len(document) == 0:
					continue

				file.write(document)
				current_sizes[dataset_index] += len(document)

				for char in document:
					chars[char] = chars.get(char, 0) + 1

				pbar.update(len(document))

				if sum(current_sizes) >= TOKENIZER_DATA_SIZE:
					break

			document = ' ' + ' '.join(list(POSSIBLE_CHARS))
			file.write(document)

			for char in document:
				chars[char] = chars.get(char, 0) + 1

			pbar.close()

		chars = sorted(chars.items(), key = lambda item: item[1], reverse = True)
		chars = [char for char, _ in chars]

		return current_sizes, chars


	def save(self, tokenizer: Tokenizer) -> None:

		for dataset in self.datasets:
			dataset.save(tokenizer)


	def summary(self) -> None:

		for dataset in self.datasets:
			print(f'{dataset.name}: {len(dataset.dataset):,} documents | {dataset.size["train"]:,} characters | {dataset.multiplier:.1f}x')