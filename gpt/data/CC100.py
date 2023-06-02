import os, shutil, random
import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from tqdm import tqdm

from gpt.data.tokenizer import Tokenizer
from gpt.settings import *
from gpt import utils
from gpt.data import shared


class Document:

	def __init__(self, start: int, end: int, size: int):

		self.start: int = start
		self.end: int = end
		self.size: int = size


def clean_dataset(paragraph: dict[str, int | str]) -> dict[str, str | int]:

	text = shared.clean(paragraph['text'])
	return {'text': text, 'size': len(text)}


def get_document_indexes(dataset, is_tokens) -> list[Document]:

	documents = [Document(0, 0, 0)]

	for i, paragraph in tqdm(enumerate(dataset['train']), total = len(dataset['train'])):

		if paragraph['size'] == 0:

			if documents[-1].size == 0:
				documents[-1].start = i + 1
			else:
				documents[-1].end = i - 1
				documents[-1].size += 0 if is_tokens else 1
				documents.append(Document(i + 1, 0, 0))

		else:
			documents[-1].size += paragraph['size'] + (1 if is_tokens else 4)

	if documents[-1].size == 0:
		documents.pop()
	else:
		documents[-1].end = i

	return documents


def create_tokenizer_data(dataset, documents: list[Document]) -> None:

	with open(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'), 'w', encoding = 'utf-8') as f:

		size = 0
		pbar = tqdm(total = TOKENIZER_DATA_SIZE)

		for document in documents:

			for i in range(document.start, document.end):
				f.write(dataset['train'][i]['text'] + '<nl>')

			f.write(dataset['train'][document.end]['text'] + '<eot>')
			size += document.size
			pbar.update(document.size)

			if size >= TOKENIZER_DATA_SIZE:
				break

		pbar.close()


def encode_dataset(paragraph: dict[str, int | str], tokenizer: Tokenizer) -> dict[str, npt.NDArray[np.uint16] | int]:

	tokens = tokenizer.encode(paragraph['text'])
	return {'tokens': tokens, 'size': len(tokens)}


def save_dataset(dataset, tokenizer: Tokenizer, train_documents: list[Document], val_documents: list[Document]) -> None:

	all_documents = {'train': train_documents, 'val': val_documents}

	for split in ['train', 'val']:

		documents = all_documents[split]
		filename = os.path.join(DATA_DIR, f'CC100/{split}.bin')
		size = sum([document.size for document in documents])
		file = np.memmap(filename, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for document in tqdm(documents, total = len(documents)):

			tokens = []

			for j in range(document.start, document.end):
				tokens.append(dataset['train'][j]['tokens'])
				tokens.append(np.array([tokenizer.control_tokens['<nl>']], dtype = np.uint16))

			tokens.append(dataset['train'][document.end]['tokens'])
			tokens.append(np.array([tokenizer.control_tokens['<eot>']], dtype = np.uint16))

			tokens = np.concatenate(tokens)

			file[i:i + len(tokens)] = tokens
			i += len(tokens)

		file.flush()


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
	#if os.path.exists(os.path.join(DATA_DIR, 'CC100')):
		#shutil.rmtree(os.path.join(DATA_DIR, 'CC100'))

	# Create missing directories
	#if not os.path.exists(DATA_DIR):
		#os.mkdir(DATA_DIR)

	#if not os.path.exists(os.path.join(DATA_DIR, 'CC100')):
		#os.mkdir(os.path.join(DATA_DIR, 'CC100'))

	# Download dataset
	print('Downloading dataset...')

	dataset = load_dataset('cc100', lang = 'fr', num_proc = NUM_THREADS)
	print('Dataset nb paragraphs:', '{:,.0f}'.format(len(dataset['train'])))

	# Clean dataset
	print('Cleaning dataset...')

	dataset = dataset.map(
		clean_dataset,
		num_proc = NUM_THREADS
	)

	# Find document indexes
	#print('Finding document indexes...')

	#documents = get_document_indexes(dataset, False)

	# Create vocab
	#print('Creating tokenizer data...')

	#random.shuffle(documents)
	#create_tokenizer_data(dataset, documents)

	tokenizer = Tokenizer()
	tokenizer.load_from_vocab(utils.load_text_array(os.path.join(DATA_DIR, 'CC100/vocab.txt')))
	#tokenizer.create(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'))
	print('Vocab size:', '{:,.0f}'.format(len(tokenizer.vocab)), '\n')

	#utils.save_text_array(tokenizer.vocab, os.path.join(DATA_DIR, 'CC100/vocab.txt'))

	#os.remove(os.path.join(DATA_DIR, 'CC100/tokenizer_data.txt'))

	# Encode dataset
	print('Encoding dataset...')

	dataset = dataset.map(
		lambda paragraph: encode_dataset(paragraph, tokenizer),
		num_proc = NUM_THREADS
	)

	# Update document indexes
	print('Updating document indexes...')

	documents = []
	documents = get_document_indexes(dataset, True)

	# Save dataset
	print('Saving dataset...')

	random.shuffle(documents)
	train_documents = documents[:int(len(documents) * (1 - VAL_RATIO))]
	val_documents = documents[int(len(documents) * (1 - VAL_RATIO)):]

	print('Train documents:', '{:,.0f}'.format(len(train_documents)))
	print('Val documents:', '{:,.0f}'.format(len(val_documents)))

	save_dataset(dataset, tokenizer, train_documents, val_documents)

	print('Importing dataset...')

	train_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/train.bin'), dtype = np.uint16, mode = 'r')
	val_tokens = np.memmap(os.path.join(DATA_DIR, 'CC100/val.bin'), dtype = np.uint16, mode = 'r')

	return tokenizer, train_tokens, val_tokens
