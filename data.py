import os, random
import numpy as np
import numpy.typing as npt
import tqdm
import tiktoken
from datasets import load_dataset

from settings import *


def convert_document(tokenizer: tiktoken.Encoding, document: str) -> dict[str, npt.NDArray[np.uint16]]:

	tokens = tokenizer.encode_ordinary(document['text'])
	tokens.append(tokenizer.eot_token)
	tokens = np.array(tokens, dtype = np.uint16)

	return {'tokens': tokens}


def import_dataset() -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:

	if not os.path.exists(DATA_DIR):
		os.makedirs(DATA_DIR)

	if os.path.exists(os.path.join(DATA_DIR, 'train.bin')) and os.path.exists(os.path.join(DATA_DIR, 'val.bin')):

		print('Importing dataset...')

		train_tokens = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype = np.uint16, mode = 'r')
		val_tokens = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype = np.uint16, mode = 'r')

		return train_tokens, val_tokens

	print('Downloading dataset...')

	dataset = load_dataset('openwebtext', num_proc = NUM_THREADS)
	dataset = dataset['train'].train_test_split(test_size = VAL_RATIO, shuffle = True, seed = random.randint(0, 100000))
	dataset['val'] = dataset.pop('test')

	tokenizer = tiktoken.get_encoding('gpt2')

	print('Tokenizing...')

	dataset = dataset.map(
		lambda x: convert_document(tokenizer, x),
		remove_columns = ['text']
	)

	print('Saving dataset...')

	for split, documents in dataset.items():

		filename = os.path.join(DATA_DIR, f'{split}.bin')
		size = sum([len(document['tokens']) for document in documents])
		file = np.memmap(filename, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for batch_i in tqdm(range(DOCUMENT_BATCH_SIZE), desc = split):

			batch = documents.shard(num_shards = DOCUMENT_BATCH_SIZE, index = batch_i, contiguous = True).with_format('numpy')
			batch = np.concatenate(batch['tokens'])
			file[i:i + len(batch)] = batch
			i += len(batch)

		file.flush()

	train_tokens = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype = np.uint16, mode = 'r')
	val_tokens = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype = np.uint16, mode = 'r')

	return train_tokens, val_tokens
