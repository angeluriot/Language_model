import os, random
import numpy as np
import xml.etree.ElementTree as ET

from settings import *

def clean(text):

	text = text.strip()
	text = text.replace('’', "'"); text = text.replace('‘', "'"); text = text.replace('“', '"'); text = text.replace('”', '"'); text = text.replace('„', '"'); text = text.replace('´', "'"); text = text.replace('`', "'"); text = text.replace('ʹ', "'")
	text = text.replace('—', '-'); text = text.replace('–', '-'); text = text.replace('−', '-'); text = text.replace('⁻', '-'); text = text.replace('―', '-'); text = text.replace('ˉ', '-'); text = text.replace('¯', '-')
	text = text.replace('…', '...')
	text = text.replace('⁄', '/')
	text = text.replace('＊', '*')
	text = text.replace('\t', '    ')
	text = text.replace('º', '°')
	text = text.replace('＾', '^')
	text = text.replace('（', ' ('); text = text.replace('）', ') ')
	text = text.replace('［', ' ['); text = text.replace('］', '] ')
	text = text.replace('｛', ' {'); text = text.replace('｝', '} ')
	text = text.replace('＜', '<'); text = text.replace('＞', '>')
	text = text.replace('＝', '='); text = text.replace('＋', '+'); text = text.replace('％', '%'); text = text.replace('＄', '$'); text = text.replace('＃', '#')
	text = text.replace('¸', ',')

	ban_space_chars = [' ', ' ', ' ', ' ', '­', '️', '‍']
	ban_chars = ['̃', '̈ ​', '﻿', '́', '͟', '​', '̂', '͡', '‎', '︎', '̀', '͜', '̶', '̿', '̲', '̯', '̅', '‏', '', '‪', '‬', '‮', '卐']

	for c in ban_space_chars:
		text = text.replace(c, ' ')

	for c in ban_chars:
		text = text.replace(c, '')

	while ' \n' in text:
		text = text.replace(' \n', '\n')

	text = text.replace('\n', '<nl>')

	return text


def parse_dataset(dataset_path):

	if not os.path.exists(PROCESSED_DATA_DIR):
		os.makedirs(PROCESSED_DATA_DIR)

	if os.path.exists(os.path.join(PROCESSED_DATA_DIR, 'dataset.txt')):
		print('Importing parsed dataset...')
		return open(os.path.join(PROCESSED_DATA_DIR, 'dataset.txt'), 'r', encoding = 'utf-8').read().strip()

	with open(os.path.join(PROCESSED_DATA_DIR, 'dataset.txt'), 'w', encoding = 'utf-8') as file:

		print('Importing dataset...')
		data = ET.parse(dataset_path).getroot()
		file.truncate(0)
		dataset_size = 0

		print('Parsing dataset...')

		for i in range(len(data)):
			for j in range(len(data[i])):

				message = clean(data[i][j].text)
				file.write(message)
				dataset_size += len(message)

				if dataset_size > 0:
					file.write('<eom>')
					dataset_size += 5

			if dataset_size > 0:
				file.write('<eod>')
				dataset_size += 5

			if DATASET_MAX_SIZE != None and dataset_size >= DATASET_MAX_SIZE:
				break

	print('Importing parsed dataset...')
	return open(os.path.join(PROCESSED_DATA_DIR, 'dataset.txt'), 'r', encoding = 'utf-8').read().strip()



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
