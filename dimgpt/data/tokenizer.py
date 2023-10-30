import numpy as np
import numpy.typing as npt
import tokenizers as tk
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer
from tqdm import tqdm

from dimgpt.data.clean import *
import dimgpt.data.pretokenizer as pretk
from dimgpt.settings import *


class Tokenizer:

	def __init__(self):

		self.vocab: list[str] = []
		self.to_index: dict[str, int] = {}
		self.to_token: dict[int, str] = {}


	def load_from_vocab(self, vocab: list[str]) -> None:

		self.vocab = vocab.copy()
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}


	def create(self, data_path: str) -> None:

		self._create_vocab(data_path)
		dataset = open(data_path, 'r', encoding = 'utf-8').read()
		self._sort_vocab(dataset)


	def _create_vocab(self, data_path: str) -> None:

		print('Creating vocab...')

		tokenizer = tk.Tokenizer(BPE(unk_token = '<unk>'))
		tokenizer.pre_tokenizer = PreTokenizer.custom(pretk.PreTokenizer())

		trainer = BpeTrainer(
			vocab_size = int(VOCAB_SIZE * 1.1),
			show_progress = True,
			special_tokens = CONTROL_CHARS
		)

		tokenizer.train([data_path], trainer)

		self.vocab = list(tokenizer.get_vocab().keys())
		vocab_size = len(self.vocab)

		def is_valid(word: str) -> bool:

			if len(word) > MAX_TOKEN_LENGTH:
				return False

			if word.endswith(' ') and len(word) > 4:
				return False

			nb_digits = 0

			for char in word:
				if char.isdigit():
					nb_digits += 1

			return nb_digits <= 2

		self.vocab = list(filter(lambda v: is_valid(v), self.vocab))

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({vocab_size - len(self.vocab):,} big tokens removed)')
		vocab_size = len(self.vocab)

		for i in range(100):
			if str(i) not in self.vocab:
				self.vocab.append(str(i))
			if ' ' + str(i) not in self.vocab:
				self.vocab.append(' ' + str(i))
			if i < 10 and '0' + str(i) not in self.vocab:
				self.vocab.append('0' + str(i))
			if i < 10 and ' 0' + str(i) not in self.vocab:
				self.vocab.append(' 0' + str(i))

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({len(self.vocab) - vocab_size:,} number tokens added)')

		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}


	def _sort_vocab(self, dataset: str) -> None:

		print('Pretokenize...')
		data = pretk.split(dataset)

		print('Sorting vocab...')
		vocab = {v: 0 for v in self.vocab}
		nb_tokens = 0
		total_tokens_length = 0

		for i in tqdm(range(len(data))):

			if data[i] in self.to_index:
				vocab[data[i]] += 1
				nb_tokens += 1
				total_tokens_length += len(data[i])
				continue

			j = 0

			while j < len(data[i]):

				found = False

				for k in reversed(range(min(MAX_TOKEN_LENGTH, len(data[i]) - j))):

					word = data[i][j:j + k + 1]

					if word in self.to_index:
						vocab[word] += 1
						nb_tokens += 1
						total_tokens_length += len(word)
						j += k
						found = True
						break

				if not found:
					vocab['<unk>'] += 1
					nb_tokens += 1
					total_tokens_length += 5

				j += 1

		self.vocab = list(sorted(vocab.items(), key = lambda x: x[1], reverse = True))
		vocab_size = len(self.vocab)

		while len(self.vocab) > VOCAB_SIZE:

			for i in range(len(self.vocab) - 1, -1, -1):

				if len(self.vocab[i][0]) > 1 and self.vocab[i][0] not in CONTROL_CHARS and not (self.vocab[i][0][-1].isdigit() and len(self.vocab[i][0]) <= 2):
					self.vocab.pop(i)
					break

		self.vocab = [v[0] for v in self.vocab]
		self.vocab = list(filter(lambda x: x not in CONTROL_CHARS, self.vocab))
		self.vocab = self.vocab + CONTROL_CHARS

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({vocab_size - len(self.vocab):,} unused tokens removed)')

		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}

		print(f'Number of tokens: {nb_tokens:,}')
		print(f'Average token length: {total_tokens_length / nb_tokens:.2f}')


	def encode(self, text: str, clean_text: bool = False, to_vector: bool = False, verbose: bool = False) -> npt.NDArray[np.uint16]:

		if verbose:
			print('Pretokenize...')

		if clean_text:
			text = clean_string(text)

		data = pretk.split(text)

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

				for k in reversed(range(min(MAX_TOKEN_LENGTH, len(data[i]) - j))):

					word = data[i][j:j + k + 1]

					if word in self.to_index:
						output.append(self.to_index[word])
						j += k
						found = True
						break

				if not found:
					output.append(self.to_index['<unk>'])

				j += 1

		if to_vector:
			return torch.tensor(output, dtype = torch.long)

		return np.array(output, dtype = np.uint16)


	def decode(self, tokens: npt.NDArray[np.uint16] | list[np.uint16] | list[int] | np.uint16 | int,
		keep_token_names: bool = False, token_array: bool = False) -> str | list[str]:

		if type(tokens) == int or type(tokens) == np.uint16:
			tokens = [tokens]
		if type(tokens) == torch.Tensor:
			tokens = tokens.detach().to('cpu').tolist()
		elif type(tokens) != list:
			tokens = list(tokens)

		text = []

		for t in tokens:

			if t < 0 or t >= len(self.vocab):
				continue

			if keep_token_names:
				text.append(self.to_token[t])
			else:
				text.append(decode_string(self.to_token[t]))

		if token_array:
			return text

		return ''.join(text)
