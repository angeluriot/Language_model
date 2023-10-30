import os, random, csv
import numpy as np
from tqdm import tqdm

from dimgpt.datasets.dataset import Dataset
from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class FrenchTweets(Dataset):

	def __init__(self):

		super().__init__('french_tweets')


	def import_dataset(self) -> None:

		print('Importing dataset...')

		with open('E:/Angel/Development/Datasets/French_tweets/dataset.csv', encoding = 'utf-8') as f:

			data = csv.reader(f, delimiter = ',')

			print('Parsing dataset...')

			self.dataset = []

			for row in data:
				self.dataset.append(str(row[1]))

		self.dataset = self.dataset[1:]


	def save(self, tokenizer: Tokenizer) -> None:

		for i in tqdm(range(len(self.dataset)), desc = f'Tokenizing {self.name}'):
			self.dataset[i] = self.document_to_tokens(self.dataset[i], tokenizer)

		random.shuffle(self.dataset)

		split = int(len(self.dataset) * VAL_RATIO * 10)
		splits = {
			'train': self.dataset[:-split],
			'val': self.dataset[-split:]
		}

		folder = os.path.join(DATA_DIR, self.name)

		if not os.path.exists(folder):
			os.mkdir(folder)

		for split, documents in splits.items():

			size = np.sum([doc['size'] for doc in documents], dtype = np.uint64)
			path = os.path.join(folder, f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
			i = 0

			for doc in tqdm(documents, desc = f'Saving {self.name} {split}'):
				tokens = doc['tokens']
				file[i:i + len(tokens)] = tokens
				i += len(tokens)

			file.flush()