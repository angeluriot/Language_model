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

		else:
			self.create_vocab()
			self.sort_vocab(dataset)
			pickle.dump(self.vocab, open(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl'), 'wb'))

		self.to_index = {v: i for i, v in enumerate(self.vocab)}
		self.to_token = {i: v for i, v in enumerate(self.vocab)}
		self.max_token_length = max([len(v) for v in self.vocab])


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
		self.max_token_length = max([len(v) for v in self.vocab])


	def sort_vocab(self, dataset):

		print('Sorting vocab...')

		text = list(dataset)
		vocab = {v: 0 for v in self.vocab}
		i = 0

		while i < len(text):

			if text[i] == '':
				continue

			found = False

			for j in reversed(range(self.max_token_length)):

				word = ''.join(text[i:i + j + 1])

				if word in self.to_index:
					vocab[word] += 1
					i += j
					found = True
					break

			if not found:
				vocab['<unk>'] += 1

			i += 1

		self.vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
		self.vocab = [v[0] for v in self.vocab]


	def encode(self, text):

		if type(text) != list:
			text = list(text)

		output = []
		i = 0

		while i < len(text):

			if text[i] == '':
				continue

			if text[i] == '\n':
				output.append(self.to_index['<nl>'])
				continue

			found = False

			for j in reversed(range(self.longest_token)):
				if ''.join(text[i:i + j + 1]) in self.to_index:
					output.append(self.to_index[''.join(text[i:i + j + 1])])
					i += j
					found = True
					break

			if not found:
				print('found')

			if not found:
				output.append(self.to_index['<unk>'])

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
