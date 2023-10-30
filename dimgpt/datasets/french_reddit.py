import os, random
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from markdown import markdown
from bs4 import BeautifulSoup

from dimgpt.datasets.dataset import Dataset
from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class FrenchReddit(Dataset):

	def __init__(self):

		super().__init__('french_reddit')


	def import_dataset(self) -> None:

		print('Importing dataset...')

		data = ET.parse('E:/Angel/Development/Datasets/French_reddit/dataset.xml').getroot()

		print('Parsing dataset...')

		self.dataset = []

		for document in data:

			self.dataset.append([])

			for message in document:
				self.dataset[-1].append(str(message.text))


	def _process_markdown(self, text: str) -> str:

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

		return text.strip()


	def _process_message(self, message: str) -> str:

		message = self._process_markdown(message)
		message = clean_string(message)

		if len(message) == 0:
			return ''

		message = message.replace('<tab>', '\t')
		message = message.replace('<nl>', '\n')

		while True:

			if message[0] == '>':
				start = 0
			elif message.find('\n>') != -1:
				start = message.find('\n>') + 1
			else:
				start = None

			if start is None:
				break

			end = message.find('\n', start + 1)

			if end == -1:
				break

			message = message[:start] + message[end:]

		return clean_string(decode_string(message, True))


	def process_document(self, document) -> str:

		for i in range(len(document)):
			document[i] = self._process_message(document[i])

		document = [m for m in document if len(m) > 0]

		if len(document) < 2:
			return ''

		text = ''

		for i in range(len(document)):
			if i % 2 == 0:
				text += '<user>' + document[i]
			else:
				text += '<bot>' + document[i]

		return text + '<eot>'


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