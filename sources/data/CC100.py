import os, shutil, random
import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from tqdm import tqdm

from sources.data.tokenizer import Tokenizer
from sources.settings import *
from sources import utils
from sources.data import shared


def get_document_indexes(dataset) -> list[int]:

	idx = []

	for i, document in enumerate(dataset['train']):

		if document['text'].strip() == '' and i < len(dataset['train']) - 1:

			idx.append(i + 1)

			if len(idx) > 1 and idx[-2] == idx[-1] - 1:
				idx.pop(-2)

	return idx


def create_tokenizer_data(dataset, idx: list[int]) -> None:

	with open(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'), 'w', encoding = 'utf-8') as f:

		size = 0

		for i in tqdm(idx):

			j = 0

			while i + j < len(dataset['train']):

				text = shared.clean(dataset['train'][i + j]['text'])

				if text == '':
					f.write('<eot>')
					size += len(text) + 5
					break

				f.write(text + '<nl><nl>')
				size += len(text) + 8
				j += 1

			if TOKENIZER_DATA_SIZE != None and size >= TOKENIZER_DATA_SIZE:
				break


def convert_paragraph(tokenizer: Tokenizer, document: dict[str, str]) -> dict[str, npt.NDArray[np.uint16]]:

	if document['text'].strip() == '':
		return {'tokens': np.array([], dtype = np.uint16)}

	tokens = tokenizer.encode(shared.clean(document['text']))

	return {'tokens': tokens}


def save_dataset(dataset, idx: list[int]) -> None:


	for split in ['train', 'val']:

		filename = os.path.join(DATA_DIR, f'{split}.bin')
		size = sum([len(document['tokens']) for document in documents])

			size = 0

			for i in tqdm(idx):

				j = 0

				while i + j < len(dataset['train']):

					text = shared.clean(dataset['train'][i + j]['text'])

					if text == '':
						f.write('<eot>')
						size += len(text) + 5
						break

					f.write(text + '<nl><nl>')
					size += len(text) + 8
					j += 1

				if TOKENIZER_DATA_SIZE != None and size >= TOKENIZER_DATA_SIZE:
					break


# Main function that process and import the dataset
def get_data() -> tuple[Tokenizer, npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:

	# Import already processed data if it exists
	if os.path.exists(os.path.join(DATA_DIR, 'CC100/vocab.txt')) and \
		os.path.exists(os.path.join(DATA_DIR, 'CC100/train.bin')) and \
		os.path.exists(os.path.join(DATA_DIR, 'CC100/val.bin')):

		print('Importing dataset...')

		vocab = utils.load_text_array(os.path.join(DATA_DIR, 'CC100/vocab.txt'))
		train_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/train.bin'), dtype = np.uint16, mode = 'r')
		val_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/val.bin'), dtype = np.uint16, mode = 'r')

		toknenizer = Tokenizer()
		toknenizer.load_from_vocab(vocab)

		return toknenizer, train_tokens, val_tokens

	# Delete old data if it exists
	if os.path.exists(os.path.join(DATA_DIR, 'CC100')):
		shutil.rmtree(os.path.join(DATA_DIR, 'CC100'))

	# Create missing directories
	if not os.path.exists(DATA_DIR):
		os.mkdir(DATA_DIR)

	if not os.path.exists(os.path.join(DATA_DIR, 'CC100')):
		os.mkdir(os.path.join(DATA_DIR, 'CC100'))

	# Download dataset
	print('Downloading dataset...')

	dataset = load_dataset('cc100', lang = 'fr', num_proc = NUM_THREADS)
	print('Dataset nb paragraphs:', '{:,.0f}'.format(len(dataset['train'])))

	# Find document indexes
	print('Finding document indexes...')

	idx = get_document_indexes(dataset)

	# Create vocab
	print('Creating tokenizer data...')

	random.shuffle(idx)
	create_tokenizer_data(dataset, idx)

	tokenizer = Tokenizer()
	tokenizer.create(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'))
	print('Vocab size:', '{:,.0f}'.format(len(tokenizer.vocab)), '\n')

	os.remove(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'))

	# Encode dataset
	print('Encode dataset...')

	dataset = dataset.map(
		lambda x: convert_paragraph(tokenizer, x),
		remove_columns = ['text'],
		num_proc = NUM_THREADS
	)

	# Save dataset
	print('Save dataset...')

	random.shuffle(idx)
	train_idx = idx[:int(len(idx) * VAL_RATIO)]
	val_idx = idx[int(len(idx) * VAL_RATIO):]

	save_dataset(dataset, train_idx, val_idx)

	print('Importing dataset...')

	train_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/train.bin'), dtype = np.uint16, mode = 'r')
	val_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/val.bin'), dtype = np.uint16, mode = 'r')

	return toknenizer, train_tokens, val_tokens
