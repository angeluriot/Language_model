import os, random
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from markdown import markdown
from bs4 import BeautifulSoup

from dimgpt.datasets.dataset import Dataset
from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class FrInstructs(Dataset):

	def __init__(self):

		super().__init__('fr_instructs')


	def import_dataset(self) -> None:

		self.dataset = load_dataset('Enno-Ai/fr-instructs', num_proc = NUM_THREADS)


	def size(self) -> int:

		return len(self.dataset['train'])


	def get_document(self, i: int | None = None, decode: bool = False) -> str:

		if i is None:
			i = random.randint(0, self.size() - 1)

		doc = self.process_document(self.dataset['train'][i])

		if decode:
			return decode_string(doc)

		return doc


	def _process_markdown(self, text: str, clean_start: bool = False) -> str:

		text = text.strip()

		html = markdown(text)
		soup = BeautifulSoup(html, features = 'html.parser')
		text = soup.get_text()

		text = '\n' + text
		text = text.replace('\n * • ', '\n • ')
		text = text.replace('\n * •', '\n • ')
		text = text.replace('\n* • ', '\n • ')
		text = text.replace('\n* •', '\n • ')
		text = text.replace('\n * ', '\n • ')
		text = text.replace('\n *', '\n • ')
		text = text.replace('\n* ', '\n • ')
		text = text.replace('\n*', '\n • ')
		text = text[1:]

		if clean_start:
			if len(text) >= 5 and ((not text[0].isalpha() and not text[0].isascii()) or text[0].isdecimal()):
				if text[1].isalpha():
					text = text[1:]
				elif text[2].isalpha():
					text = text[2:]

			elif len(text) >= 5 and ((not text[0].isalpha() and not text[0].isascii()) or text[0].isdecimal()) \
				and ((not text[1].isalpha() and not text[1].isascii()) or text[1] == '-' or text[1] == '\n'):
				if text[2].isalpha():
					text = text[2:]
				elif text[3].isalpha():
					text = text[3:]

		for i in range(len(text)):
			if text[i:].startswith('\n-----'):
				j = i + 1
				while j < len(text) and text[j] == '-':
					j += 1
				text = text[:i] + text[j:]
				break

		return text.strip()


	def process_document(self, document) -> str:

		context = clean_string(self._process_markdown(document['input']))
		instruction = clean_string(self._process_markdown(document['instruction'], True))
		answer = clean_string(self._process_markdown(document['output']))

		if len(instruction) == 0 or len(answer) == 0:
			return ''

		if len(context) == 0:
			return f'<user>{instruction}<bot>{answer}<eot>'

		return f'<user>{context}<nl><nl>{instruction}<bot>{answer}<eot>'


	def save(self, tokenizer: Tokenizer) -> None:

		batch_size = 256
		split_dataset = self.dataset['train'].train_test_split(test_size = VAL_RATIO, shuffle = True)
		split_dataset['val'] = split_dataset.pop('test')

		tokenized = split_dataset.map(
			lambda doc: self.document_to_tokens(doc, tokenizer),
			desc = f'Tokenizing {self.name}',
			num_proc = NUM_THREADS
		)

		folder = os.path.join(DATA_DIR, self.name)

		if not os.path.exists(folder):
			os.mkdir(folder)

		for split, documents in tokenized.items():

			size = np.sum(documents['size'], dtype = np.uint64)
			path = os.path.join(folder, f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
			i = 0

			for batch_i in tqdm(range(batch_size), desc = f'Saving {self.name} {split}'):

				batch = documents.shard(num_shards = batch_size, index = batch_i, contiguous = True).with_format('numpy')
				file_batch = np.concatenate(batch['tokens'])
				file[i:i + len(file_batch)] = file_batch
				i += len(file_batch)

			file.flush()

		self.save_ids()