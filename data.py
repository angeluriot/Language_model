import os, random
import numpy as np
import numpy.typing as npt
import xml.etree.ElementTree as ET

from settings import *


def clean(text: str) -> str:

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


def parse_dataset(dataset_path: str) -> str:

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

			dialog = ''

			for j in range(len(data[i])):

				dialog += clean(data[i][j].text)

				if len(dialog) > 0:
					dialog += '<eom>'

			if len(dialog) > 0:
				dialog += '<eod>'

			dialog = dialog.replace('<eom><eod>', '<eod>')
			file.write(dialog)
			dataset_size += len(dialog)

			if DATASET_MAX_SIZE != None and dataset_size >= DATASET_MAX_SIZE:
				break

	print('Importing parsed dataset...')
	return open(os.path.join(PROCESSED_DATA_DIR, 'dataset.txt'), 'r', encoding = 'utf-8').read().strip()


def split_dataset(dataset: npt.NDArray[np.uint16]) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:

	if not os.path.exists(OUTPUT_DIR):
		os.makedirs(OUTPUT_DIR)

	if os.path.exists(os.path.join(OUTPUT_DIR, 'train_indexes.npy')) and os.path.exists(os.path.join(OUTPUT_DIR, 'val_indexes.npy')):
		return np.load(os.path.join(OUTPUT_DIR, 'train_indexes.npy')), np.load(os.path.join(OUTPUT_DIR, 'val_indexes.npy'))

	sub_val_size = int((len(dataset) * VAL_RATIO) / NUM_VAL_PARTS)
	train_indexes = []
	val_indexes = []
	val_start_indexes = [random.randint(3 * MAX_CONTEXT, len(dataset) - sub_val_size - 6 * MAX_CONTEXT) for _ in range(NUM_VAL_PARTS)]
	i = 0

	while i < len(dataset) - 3 * MAX_CONTEXT:

		if (i + MAX_CONTEXT - 1) in val_start_indexes:
			i += MAX_CONTEXT - 1
			for _ in range(sub_val_size):
				i += 1
				val_indexes.append(i)
			i += MAX_CONTEXT + 1

		train_indexes.append(i)
		i += 1

	train_indexes = np.array(train_indexes, dtype = np.uint64)
	val_indexes = np.array(val_indexes, dtype = np.uint64)

	np.save(os.path.join(OUTPUT_DIR, 'train_indexes.npy'), train_indexes)
	np.save(os.path.join(OUTPUT_DIR, 'val_indexes.npy'), val_indexes)

	return train_indexes, val_indexes
