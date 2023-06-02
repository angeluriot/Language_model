import numpy as np
import numpy.typing as npt
import tokenizers as tk
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import BertNormalizer
from tqdm import tqdm

import gpt.data.pretokenizer as my_pretokenizer
from gpt.settings import *


class Tokenizer:

	def __init__(self):

		self.vocab: list[str] = []
		self.to_index: dict[str, int] = {}
		self.to_token: dict[int, str] = {}
		self.max_token_length: int = 0
		self.control_tokens: dict[str, int] = {}


	def load_from_vocab(self, vocab: list[str]) -> None:

		self.vocab = vocab.copy()
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])
		self.control_tokens = {c: self.to_index[c] for c in CONTROL_CHARS}


	def create(self, data_path: str) -> None:

		self._create_vocab(data_path)
		dataset = open(data_path, 'r', encoding = 'utf-8').read()
		self._sort_vocab(dataset)


	def _create_vocab(self, data_path: str) -> None:

		print('Creating vocab...')

		tokenizer = tk.Tokenizer(BPE(unk_token = CONTROL_CHARS[-1]))
		tokenizer.normalizer = BertNormalizer(strip_accents = False, lowercase = False)
		tokenizer.pre_tokenizer = PreTokenizer.custom(my_pretokenizer.PreTokenizer())

		trainer = BpeTrainer(
			vocab_size = VOCAB_SIZE,
			show_progress = True,
			special_tokens = CONTROL_CHARS
		)

		tokenizer.train([data_path], trainer)

		self.vocab = tokenizer.get_vocab().keys()
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])

		print('Max token length:', self.max_token_length)


	def _sort_vocab(self, dataset: str) -> None:

		print('Pretokenize...')
		data = my_pretokenizer.split(dataset)

		print('Sorting vocab...')
		vocab = {v: 0 for v in self.vocab}

		for i in tqdm(range(len(data))):

			if data[i] in self.to_index:
				vocab[data[i]] += 1
				continue

			j = 0

			while j < len(data[i]):

				found = False

				for k in reversed(range(min(self.max_token_length, len(data[i]) - j))):

					word = data[i][j:j + k + 1]

					if word in self.to_index:
						vocab[word] += 1
						j += k
						found = True
						break

				if not found:
					vocab['<unk>'] += 1

				j += 1

		self.vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
		self.vocab = [v[0] for v in self.vocab if v[1] > 0 or v[0] in CONTROL_CHARS]
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])
		self.control_tokens = {c: self.to_index[c] for c in CONTROL_CHARS}



	def encode(self, text: str, verbose: bool = False) -> npt.NDArray[np.uint16]:

		if verbose:
			print('Pretokenize...')

		data = my_pretokenizer.split(text)

		if verbose:
			print('Encoding dataset...')

		output = []

		for i in tqdm(range(len(data)), disable = not verbose):

			if data[i] in self.to_index:
				output.append(self.to_index[data[i]])
				continue

			j = 0

			while j < len(data[i]):

				found = False

				for k in reversed(range(min(self.max_token_length, len(data[i]) - j))):

					word = data[i][j:j + k + 1]

					if word in self.to_index:
						output.append(self.to_index[word])
						j += k
						found = True
						break

				if not found:
					output.append(self.control_tokens['<unk>'])

				j += 1

		return np.array(output, dtype = np.uint16)


	def decode(self, tokens: npt.NDArray[np.uint16] | list[np.uint16] | list[int] | np.uint16 | int,
		keep_token_names: bool = False, token_array: bool = False) -> str | list[str]:

		if type(tokens) == int or type(tokens) == np.uint16:
			tokens = [tokens]
		elif type(tokens) != list:
			tokens = list(tokens)

		text = []

		if keep_token_names:
			for t in tokens:
				if t < 0 or t >= len(self.vocab):
					continue
				text.append(self.to_token[t])

		else:
			for t in tokens:
				if t < 0 or t >= len(self.vocab):
					continue
				if t == self.control_tokens['<nl>']:
					text.append('\n')
				elif t == self.control_tokens['<unk>']:
					text.append('�')
				elif t == self.control_tokens['<eom>']:
					text.append('\n\n<<< END OF MESSAGE >>>\n\n')
				elif t == self.control_tokens['<eot>']:
					text.append('\n\n<<<<<<<<<<<<<<< END OF TEXT >>>>>>>>>>>>>>>\n\n')
				else:
					text.append(self.to_token[t])

		if token_array:
			return text

		return ''.join(text)
