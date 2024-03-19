import os
import numpy as np
import numpy.typing as npt
import tokenizers as tk
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import PreTokenizer
from tqdm import tqdm

from dimgpt.data.clean import *
from dimgpt.utils import *
import dimgpt.data.pretokenizer as pretk
from dimgpt.settings import *

class Tokenizer:

	def __init__(self):

		self.vocab: list[str] = []
		self.to_index: dict[str, int] = {}
		self.to_token: dict[int, str] = {}

		if os.path.exists(os.path.join(DATA_DIR, 'vocab.txt')):
			self.load_from_vocab(load_text_array(os.path.join(DATA_DIR, 'vocab.txt')))
		else:
			self.create(os.path.join(DATA_DIR, 'tokenizer_data.txt'))
			save_text_array(self.vocab, os.path.join(DATA_DIR, 'vocab.txt'))


	def _set_control_tokens(self) -> None:

		self.unknown_token = self.to_index['⮜unknown⮞']
		self.padding_token = self.to_index['⮜padding⮞']
		self.start_of_text_token = self.to_index['⮜start-of-text⮞']
		self.tab_token = self.to_index['⮜tab⮞']
		self.new_line_token = self.to_index['⮜new-line⮞']
		self.human_token = self.to_index['⮜human⮞']
		self.system_token = self.to_index['⮜system⮞']
		self.user_token = self.to_index['⮜user⮞']
		self.assistant_token = self.to_index['⮜assistant⮞']
		self.end_of_text_token = self.to_index['⮜end-of-text⮞']


	def load_from_vocab(self, vocab: list[str]) -> None:

		self.vocab = vocab.copy()
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self._set_control_tokens()


	def create(self, data_path: str) -> None:

		self._create_vocab(data_path)
		dataset = open(data_path, 'r', encoding = 'utf-8').read()
		self._sort_vocab(dataset)
		self._set_control_tokens()


	def _create_vocab(self, data_path: str) -> None:

		print('Creating vocab...')

		tokenizer = tk.Tokenizer(BPE(unk_token = '⮜unknown⮞'))
		tokenizer.pre_tokenizer = PreTokenizer.custom(pretk.PreTokenizer())

		trainer = BpeTrainer(
			vocab_size = int(VOCAB_SIZE * 1.1),
			show_progress = True,
			special_tokens = CONTROL_TOKENS
		)

		tokenizer.train([data_path], trainer)

		self.vocab = list(tokenizer.get_vocab().keys())
		vocab_size = len(self.vocab)

		def is_valid(word: str) -> bool:

			if len(word) > MAX_TOKEN_LENGTH:
				return False

			if word.endswith(' ') and len(word) > 4:
				return False

			if any(c not in POSSIBLE_CHARS for c in word):
				return False

			nb_digits = 0

			for char in word:
				if char.isdigit():
					nb_digits += 1

			return nb_digits < 2

		self.vocab = list(filter(lambda v: is_valid(v), self.vocab))

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({vocab_size - len(self.vocab):,} invalid tokens removed)')
		vocab_size = len(self.vocab)

		for i in range(10):
			if str(i) not in self.vocab:
				self.vocab.append(str(i))
			if ' ' + str(i) not in self.vocab:
				self.vocab.append(' ' + str(i))

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({len(self.vocab) - vocab_size:,} number tokens added)')
		vocab_size = len(self.vocab)

		for token in FORCED_TOKENS:
			if token not in self.vocab:
				self.vocab.append(token)

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({len(self.vocab) - vocab_size:,} forced tokens added)')
		vocab_size = len(self.vocab)

		self.vocab = CONTROL_TOKENS + self.vocab

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({len(self.vocab) - vocab_size:,} control tokens added)')

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
					vocab['⮜unknown⮞'] += 1
					nb_tokens += 1
					total_tokens_length += 5

				j += 1

		self.vocab = list(sorted(vocab.items(), key = lambda x: x[1], reverse = True))
		vocab_size = len(self.vocab)
		self.vocab = list(filter(lambda x: x[0] not in CONTROL_TOKENS, self.vocab))

		while len(self.vocab) > VOCAB_SIZE - len(CONTROL_TOKENS):

			for i in range(len(self.vocab) - 1, -1, -1):

				if len(self.vocab[i][0]) > 1 and self.vocab[i][0] not in FORCED_TOKENS and not (self.vocab[i][0][-1].isdigit() and len(self.vocab[i][0]) <= 2):
					self.vocab.pop(i)
					break

		self.vocab = [v[0] for v in self.vocab]
		self.vocab = CONTROL_TOKENS + self.vocab

		print(f'Vocab size: {vocab_size:,} -> {len(self.vocab):,} ({vocab_size - len(self.vocab):,} unused tokens removed)')

		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}

		print(f'Number of tokens: {nb_tokens:,}')
		print(f'Average token length: {total_tokens_length / nb_tokens:.2f}')


	def encode(self, text: str, clean_text: bool = True, keep_control_tokens: bool = False, verbose: bool = False) -> list[int]:

		if verbose:
			print('Pretokenize...')

		if clean_text:
			text = clean_string(text, keep_control_tokens)

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
					output.append(self.to_index['⮜unknown⮞'])

				j += 1

		return output


	def decode(self, tokens: list[int] | npt.NDArray[np.uint16] | torch.Tensor | int, keep_control_tokens: bool = False,
		token_array: bool = False) -> str | list[str]:

		if type(tokens) == int:
			tokens = [tokens]
		if type(tokens) == torch.Tensor:
			tokens = tokens.detach().to('cpu').tolist()
		elif type(tokens) != list:
			tokens = list(tokens)

		text = []

		for t in tokens:

			if t < 0 or t >= len(self.vocab):
				continue

			text.append(unclean_string(self.to_token[t], keep_control_tokens))

		if token_array:
			return text

		return ''.join(text)
