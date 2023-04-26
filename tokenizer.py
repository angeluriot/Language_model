import os, pickle
import numpy as np
import tokenizers as tk
from tokenizers.models import *
from tokenizers.trainers import *
from tokenizers.pre_tokenizers import *
from tokenizers.normalizers import *

import pretokenizer as mypretk
from settings import *


class Tokenizer:

	def __init__(self, dataset):

		self.vocab = []
		self.to_index = {}
		self.to_token = {}
		self.max_token_length = 0

		if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl')):
			print('Loading vocab...')
			self.vocab = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl'), 'rb'))
			self.to_index = {v: i for i, v in enumerate(self.vocab)}
			self.to_token = {i: v for i, v in enumerate(self.vocab)}
			self.max_token_length = max([len(v) for v in self.vocab])

		else:
			self.create_vocab()
			self.sort_vocab(dataset)
			pickle.dump(self.vocab, open(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl'), 'wb'))


	def create_vocab(self):

		print('Creating vocab...')
		tokenizer = tk.Tokenizer(BPE(unk_token = CONTROL_CHARS[-1]))
		tokenizer.normalizer = BertNormalizer(strip_accents = False, lowercase = False)
		tokenizer.pre_tokenizer = PreTokenizer.custom(mypretk.PreTokenizer())

		trainer = BpeTrainer(
			vocab_size = VOCAB_SIZE,
			show_progress = True,
			special_tokens = CONTROL_CHARS
		)

		tokenizer.train([os.path.join(PROCESSED_DATA_DIR, 'dataset.txt')], trainer)

		self.vocab = tokenizer.get_vocab().keys()
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])

		print('Max token length:', self.max_token_length)


	def sort_vocab(self, dataset):

		print('Pretokenize...')
		data = mypretk.split(dataset)

		print('Sorting vocab...')
		vocab = {v: 0 for v in self.vocab}

		for i in range(len(data)):

			if i % int(len(data) / 100) == 0:
				print('Progress:', str(int((i / len(data)) * 100)) + '%               ', end = '\r')

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

		print('Progress: 100%               ')
		self.vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
		self.vocab = [v[0] for v in self.vocab if v[1] > 0]
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])


	def encode(self, text, verbose = False):

		if verbose:
			print('Encoding dataset...')

		vocab = {v: 0 for v in self.vocab}

		if verbose:
			print('Pretokenize...')
		data = mypretk.split(dataset)

		for i in range(len(data)):

			if i % int(len(data) / 100) == 0:
				print('Progress:', str(int((i / len(data)) * 100)) + '%               ', end = '\r')

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

		print('Progress: 100%               ')
		self.vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
		self.vocab = [v[0] for v in self.vocab if v[1] > 0]
		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])
		i += 1

		return np.array(output, dtype = np.int32)


	def decode(self, tokens, keep_token_names = False, token_array = False):

		if type(tokens) == int or type(tokens) == np.int16 or type(tokens) == np.int32 or type(tokens) == np.int64:
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
				if t == self.to_index['<nl>']:
					text.append('\n')
				elif t == self.to_index['<unk>']:
					text.append('�')
				elif t == self.to_index['<eom>']:
					text.append('\n\n')
				elif t == self.to_index['<eod>']:
					text.append('\n\n-------------------------------------------------\n\n')
				else:
					text.append(self.to_token[t])

		if token_array:
			return text

		return ''.join(text)
