import os, random, pickle
import torch
import numpy as np

from dimgpt.data.tokenizer import Tokenizer
from dimgpt.settings import *
from dimgpt.training.datasets import Dataset


class PretrainingDataset(Dataset):

	def __init__(self, tokenizer: Tokenizer):

		super().__init__(tokenizer)

		datasets = os.listdir(os.path.join(DATA_DIR, 'pretraining'))
		self.datasets = []

		for dataset in datasets:

			if not os.path.isdir(os.path.join(DATA_DIR, 'pretraining', dataset)):
				continue

			meta = pickle.load(open(os.path.join(DATA_DIR, 'pretraining', dataset, f'metadata.pkl'), 'rb'))

			self.datasets.append({
				'train': {
					'data': np.memmap(os.path.join(DATA_DIR, 'pretraining', dataset, f'train.bin'), dtype = np.uint16, mode = 'r'),
					'ids': pickle.load(open(os.path.join(DATA_DIR, 'pretraining', dataset, f'train_ids.pkl'), 'rb')),
					'size': meta['size']['train']
				},
				'val': {
					'data': np.memmap(os.path.join(DATA_DIR, 'pretraining', dataset, f'val.bin'), dtype = np.uint16, mode = 'r'),
					'ids': pickle.load(open(os.path.join(DATA_DIR, 'pretraining', dataset, f'val_ids.pkl'), 'rb')),
					'size': meta['size']['val']
				},
				'training_part': meta['training_part'],
				'name': meta['name'],
				'multiplier': meta['multiplier']
			})

		self.probas = [dataset['train']['size'] * dataset['multiplier'] for dataset in self.datasets]
		self.probas = (np.array(self.probas) / np.sum(self.probas)).tolist()


	def train_size(self) -> int:

		return sum([dataset['train']['size'] for dataset in self.datasets])


	def val_size(self) -> int:

		return sum([dataset['val']['size'] for dataset in self.datasets])


	def _get_random_document(self, val: bool) -> tuple[list[int], list[int]]:

		dataset = np.random.choice(self.datasets, p = self.probas)
		ids = dataset['val']['ids'] if val else dataset['train']['ids']
		data = dataset['val']['data'] if val else dataset['train']['data']

		i = random.randint(0, len(ids) - 1)
		xy = data[ids[i]['start']:ids[i]['start'] + ids[i]['size']]
		strength = [1.0] * ids[i]['size']

		return xy, strength


	def _get_tokens(self, val: bool) -> tuple[torch.Tensor, torch.Tensor]:

		dataset = np.random.choice(self.datasets, p = self.probas)
		data = dataset['val']['data'] if val else dataset['train']['data']

		start = random.randint(0, len(data) - 1 - (MAX_CONTEXT + 1))
		xy = []

		for i in range(MAX_CONTEXT + 1):

			token = data[start + i]
			xy.append(token)

			if token == self.tokenizer.end_of_text_token:
				break

		strength = [1.0] * len(xy)

		while len(xy) < MAX_CONTEXT + 1:

			_xy, _strength = self._get_random_document(val)

			xy.extend(_xy)
			strength.extend(_strength)

		xy = xy[0:MAX_CONTEXT + 1]
		strength = strength[0:MAX_CONTEXT + 1]

		return xy, strength