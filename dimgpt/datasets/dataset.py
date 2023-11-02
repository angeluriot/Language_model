import os, random
from abc import ABC, abstractmethod
import numpy as np

from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class Dataset(ABC):

	def __init__(self, name: str):

		self.name = name
		self.dataset = None
		self.import_dataset()


	@abstractmethod
	def import_dataset(self) -> None:
		pass


	def size(self) -> int:

		return len(self.dataset)


	def get_document(self, i: int | None = None, decode: bool = False) -> str:

		if i is None:
			i = random.randint(0, self.size() - 1)

		doc = self.process_document(self.dataset[i])

		if decode:
			return decode_string(doc)

		return doc


	def process_document(self, document) -> str:

		text = clean_string(document)

		return '' if len(text) == 0 else text + '<eot>'


	def document_to_tokens(self, document, tokenizer: Tokenizer) -> list[int]:

		tokens = tokenizer.encode(self.process_document(document))

		return {'tokens': tokens, 'size': len(tokens)}


	@abstractmethod
	def save(self, tokenizer: Tokenizer) -> None:
		pass


	def save_ids(self) -> None:

		folder = os.path.join(DATA_DIR, self.name)

		for split in ['train', 'val']:

			data = np.memmap(os.path.join(folder, f'{split}.bin'), dtype = np.uint16, mode = 'r')
			ids = np.where(data == EOT_INDEX)[0] + 1

			file = np.memmap(os.path.join(folder, f'{split}_ids.bin'), dtype = np.uint64, mode = 'w+', shape = ids.shape)
			file[0] = 0
			file[1:len(ids)] = ids.astype(np.uint64)[:len(ids) - 1]

			file.flush()