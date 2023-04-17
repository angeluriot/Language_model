import os, pickle, random
import numpy as np
import xml.etree.ElementTree as ET
import emoji

from settings import *

def clean(text):

	text = text.strip()
	text = text.replace('’', "'")
	text = text.replace('‘', "'")
	text = text.replace('`', "'")
	text = text.replace('…', '...')
	text = text.replace('—', '-')
	text = text.replace('–', '-')
	text = text.replace('−', '-')
	text = text.replace('“', '"')
	text = text.replace('”', '"')
	text = text.replace('„', '"')
	text = text.replace('´', "'")
	text = text.replace('⁻', '-')
	text = text.replace('⁄', '/')
	text = text.replace('＊', '*')
	text = text.replace('	', '    ')
	text = text.replace('º', '°')
	text = text.replace('―', '-')
	text = text.replace('ˉ', '-')
	text = text.replace('¯', '-')
	text = text.replace('＾', '^')
	text = text.replace('）', ') ')

	ban_space_chars = [' ', ' ', ' ', ' ', '­', '️', '‍']
	ban_chars = ['̃', '̈ ​', '﻿', '́', '͟', '​', '̂', '͡', '‎', '︎', '̀', '͜', '̶', '̿', '̲', '̯', '̅', '‏', '', '‪', '‬', '‮', '卐']

	for c in ban_space_chars:
		text = text.replace(c, ' ')

	for c in ban_chars:
		text = text.replace(c, '')

	return text

def parse_dataset(dataset_path):

	print('Importing dataset...')

	if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'dataset.pkl')) and os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'chars.pkl')):
		dataset = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, 'dataset.pkl'), 'rb'))
		chars = pickle.load(open(os.path.join(PROCESSED_DATA_DIR, 'chars.pkl'), 'rb'))
		return dataset, chars

	dataset = []
	chars = {}
	data = ET.parse(dataset_path).getroot()

	print('Parsing dataset...')

	for i in range(len(data)):
		for j in range(len(data[i])):

			message = list(clean(data[i][j].text))

			for c in message:
				if c == '\n':
					dataset.append('<nl>')
					chars['<nl>'] = chars.get('<nl>', 0) + 1
				elif c != '':
					dataset.append(c)
					chars[c] = chars.get(c, 0) + 1

			if len(dataset) > 0 and dataset[-1] != '<eom>' and dataset[-1] != '<eod>':
				dataset.append('<eom>')
				chars['<eom>'] = chars.get('<eom>', 0) + 1

		if len(dataset) > 0 and dataset[-1] != '<eod>':
			dataset.append('<eod>')
			chars['<eod>'] = chars.get('<eod>', 0) + 1

		if i % (len(data) // 100) == 0:
			print(f'\rProgress: {int((i / len(data)) * 100)}%     ', end = '')

		if DATASET_MAX_SIZE != None and len(dataset) > DATASET_MAX_SIZE:
			break

	print('\rProgress: 100%     ')

	chars = sorted(chars.items(), key = lambda x: x[1], reverse = True)
	chars = [t[0] for t in chars]

	for c in CONTROL_CHARS:
		if c not in chars:
			chars.append(c)

	print('Cleaning chars...')

	if len(chars) > NUM_CHARS_MAX:

		i = len(chars) - 1

		while i >= 0 and len(chars) > NUM_CHARS_MAX:

			if chars[i] not in CONTROL_CHARS and emoji.emoji_count(chars[i]) == 0:
				chars.pop(i)

			i -= 1

	print('Cleaning dataset...')

	char_dict = {c: i for i, c in enumerate(chars)}

	for i in range(len(dataset)):
		if dataset[i] not in char_dict:
			dataset[i] = '<unk>'

	print('Saving dataset...')

	if not os.path.exists(PROCESSED_DATA_DIR):
		os.makedirs(PROCESSED_DATA_DIR)

	pickle.dump(dataset, open(os.path.join(PROCESSED_DATA_DIR, 'dataset.pkl'), 'wb'))
	pickle.dump(chars, open(os.path.join(PROCESSED_DATA_DIR, 'chars.pkl'), 'wb'))

	return dataset, chars


def split_dataset(dataset):

	sub_y_size = int((len(dataset) * Y_RATIO) / NUM_Y_PARTS)
	train_indexes = []
	val_indexes = []
	val_start_indexes = [random.randint(2 * MAX_CONTEXT, len(dataset) - sub_y_size - 3 * MAX_CONTEXT) for _ in range(NUM_Y_PARTS)]
	i = 0

	while i < len(dataset) - MAX_CONTEXT:

		if (i + MAX_CONTEXT - 1) in val_start_indexes:
			i += MAX_CONTEXT - 1
			for _ in range(sub_y_size):
				i += 1
				val_indexes.append(i)
			i += MAX_CONTEXT + 1

		train_indexes.append(i)
		i += 1

	return np.array(train_indexes), np.array(val_indexes)
