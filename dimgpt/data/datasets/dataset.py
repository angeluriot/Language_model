import os, random, pickle
from tqdm import tqdm
from abc import ABC
import numpy as np
import numpy.typing as npt
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class Dataset(ABC):

	def __init__(self) -> None:

		self.dataset = None
		self.training_part = ''
		self.name = ''
		self.size = {'train': 0, 'val': 0}
		self.multiplier = 1.0


	def get_document(self, i: int | None = None) -> str:

		if i is None:
			i = random.randint(0, len(self.dataset) - 1)

		return '⮜start-of-text⮞' + clean_string(self.dataset[i]['text']) + '⮜end-of-text⮞'


	def document_to_tokens(self, document: dict[str, str], tokenizer: Tokenizer) -> dict[str, npt.NDArray[np.uint16] | int]:

		tokens = [tokenizer.start_of_text_token, *tokenizer.encode(document['text']), tokenizer.end_of_text_token]

		return {'tokens': np.array(tokens, dtype = np.uint16), 'size': len(tokens)}


	def save(self, tokenizer: Tokenizer) -> None:

		if os.path.exists(os.path.join(DATA_DIR, self.training_part, self.name, f'train.bin')):
			return

		os.makedirs(os.path.join(DATA_DIR, self.training_part, self.name), exist_ok = True)

		split_dataset = self.dataset.train_test_split(test_size = PRETRAINING_VAL_RATIO, shuffle = True)
		split_dataset['val'] = split_dataset.pop('test')

		tokenized = split_dataset.map(
			lambda doc: self.document_to_tokens(doc, tokenizer),
			desc = f'Tokenizing {self.name}',
			num_proc = NUM_THREADS
		)

		for split, documents in tokenized.items():

			total = 0
			ids = []

			for doc in tqdm(documents, desc = f'Saving {self.name} {split} ids'):

				ids.append({
					'start': total,
					'size': doc['size']
				})

				total += doc['size']

			with open(os.path.join(DATA_DIR, self.training_part, self.name, f'{split}_ids.pkl'), 'wb') as file:
				pickle.dump(ids, file)

			batch_size = 1_024

			while batch_size >= len(documents):
				batch_size //= 2

			self.size[split] = int(np.sum(documents['size'], dtype = np.uint64))
			path = os.path.join(DATA_DIR, self.training_part, self.name, f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (self.size[split],))
			i = 0

			for batch_i in tqdm(range(batch_size), desc = f'Saving {self.name} {split}'):

				batch = documents.shard(num_shards = batch_size, index = batch_i, contiguous = True).with_format('numpy')
				file_batch = np.concatenate(batch['tokens'])
				file[i:i + len(file_batch)] = file_batch
				i += len(file_batch)

			file.flush()

		with open(os.path.join(DATA_DIR, self.training_part, self.name, f'metadata.pkl'), 'wb') as file:
			pickle.dump({
				'training_part': self.training_part,
				'name': self.name,
				'size': self.size,
				'multiplier': self.multiplier
			}, file)
