import random
from abc import ABC, abstractmethod

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
