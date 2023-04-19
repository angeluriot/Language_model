import os, pickle
import numpy as np
from settings import *
import tokenizers as tk
from tokenizers.models import *
from tokenizers.trainers import *
from tokenizers.pre_tokenizers import *
from tokenizers.normalizers import *


class Tokenizer:

	def __init__(self, dataset):

		if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl')):

			print('Loading vocab...')
			vocab = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, 'vocab.pkl'), 'rb'))

			tokenizer = tk.Tokenizer(BPE(vocab = vocab, unk_token = '<unk>'))
			tokenizer.normalizer = BertNormalizer(strip_accents = False, lowercase = False)
			tokenizer.pre_tokenizer = Metaspace(replacement = " ", add_prefix_space = False)

		else:

			tokenizer = tk.Tokenizer(BPE(unk_token = '<unk>'))
			tokenizer.normalizer = BertNormalizer(strip_accents = False, lowercase = False)
			tokenizer.pre_tokenizer = Metaspace(replacement = " ", add_prefix_space = False)

			trainer = BpeTrainer(
				vocab_size = VOCAB_SIZE,
				show_progress = True,
				special_tokens = ['<eom>', '<eod>']
			)

			tokenizer.train([os.path.join(PROCESSED_DATA_DIR, 'dataset.txt')], trainer)

			tokenizer.model.save(os.path.join(PROCESSED_DATA_DIR, 'tokenizer.json'))

			if not os.path.exists(PROCESSED_DATA_DIR):
				os.mkdir(PROCESSED_DATA_DIR)


for v in vocab:
	print(f'[{v}]', end = ' ')


	def pretokenize(self, dataset):

		print('Pretokenize...')
		words = {}
		word = ''
		i = 0

		while i < len(dataset):

			if dataset[i] in CONTROL_CHARS:

				if len(word) > 0:
					words[word] = words.get(word, 0) + 1
					word = ''

				words[dataset[i]] = words.get(dataset[i], 0) + 1
				i += 1
				continue

			if len(word) > 0 and word[-1] != ' ' and dataset[i] == ' ':
				words[word] = words.get(word, 0) + 1
				word = ''
				continue

			word += dataset[i]
			i += 1

		if len(word) > 0:
			words[word] = words.get(word, 0) + 1

		return words


	def merge(self, words):

		print('Merge...')
		vocab = {}
		temp = []

		# Get all the characters
		for (word, nb) in words.items():

			if word in CONTROL_CHARS:
				vocab[word] = vocab.get(word, 0) + nb
				continue

			for char in word:
				vocab[char] = vocab.get(char, 0) + nb

			if len(word) > 1:
				temp.append([list(word), nb])

		words = temp

		# Merge the characters
		while len(vocab) < VOCAB_SIZE:

			print(f'{len(vocab)} / {VOCAB_SIZE}          ', end = '\r')

			# Get all the possible merges
			merges = {}
			i = 0

			while i < len(words):

				word = words[i]

				if len(word[0]) == 1:
					words.pop(i)
					continue

				for j in range(len(word[0]) - 1):

					merge = word[0][j] + word[0][j + 1]

					if len(merge) <= MAX_TOKEN_LENGTH:
						merges[merge] = merges.get(merge, 0) + word[1]

				i += 1

			if len(merges) == 0:
				break

			# Find the best merge
			best_merge = sorted(merges.items(), key = lambda x: x[1], reverse = True)[0][0]

			# Apply the best merge
			for word in words:
				for i in range(len(word[0]) - 1):

					if word[0][i] + word[0][i + 1] == best_merge:

						vocab[word[0][i]] = vocab.get(word[0][i], 0) - word[1]
						vocab[word[0][i + 1]] = vocab.get(word[0][i + 1], 0) - word[1]
						vocab[best_merge] = vocab.get(best_merge, 0) + word[1]

						word[0][i] = best_merge
						word[0].pop(i + 1)
						break

			for v, nb in vocab.copy().items():
				if nb == 0 and len(v) > 1 and v not in CONTROL_CHARS:
					vocab.pop(v)

		vocab = sorted(vocab.items(), key = lambda x: x[1], reverse = True)
		vocab = [t[0] for t in vocab]
		vocab.pop('<unk>')
		vocab.append('<unk>')
		print(f'{VOCAB_SIZE} / {VOCAB_SIZE}          ')

		return vocab


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
