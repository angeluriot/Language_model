import os, random
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from dimgpt.datasets.dataset import Dataset
from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class Wikipedia(Dataset):

	def __init__(self):

		super().__init__('wikipedia')


	def import_dataset(self) -> None:

		self.dataset = load_dataset('wikipedia', '20220301.fr', num_proc = NUM_THREADS)
		self.dataset = self.dataset.filter(lambda x: len(x['text']) > MAX_CONTEXT * AVERAGE_TOKEN_LENGTH, num_proc = NUM_THREADS)


	def size(self) -> int:

		return len(self.dataset['train'])


	def get_document(self, i: int | None = None, decode: bool = False) -> str:

		if i is None:
			i = random.randint(0, self.size() - 1)

		doc = self.process_document(self.dataset['train'][i])

		if decode:
			return decode_string(doc)

		return doc


	def process_document(self, document) -> str:

		text = clean_string(document['text'])

		return '' if len(text) == 0 else text + '<eot>'


	def save(self, tokenizer: Tokenizer) -> None:

		split_dataset = self.dataset['train'].train_test_split(test_size = VAL_RATIO, shuffle = True)
		split_dataset['val'] = split_dataset.pop('test')

		tokenized = split_dataset.map(
			lambda doc: self.document_to_tokens(doc, tokenizer),
			desc = f'Tokenizing {self.name}',
			num_proc = NUM_THREADS
		)

		folder = os.path.join(DATA_DIR, self.name)

		if not os.path.exists(folder):
			os.mkdir(folder)

		for split, documents in tokenized.items():

			size = np.sum(documents['size'], dtype = np.uint64)
			path = os.path.join(folder, f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
			i = 0

			for batch_i in tqdm(range(SAVING_BATCHS // 8), desc = f'Saving {self.name} {split}'):

				batch = documents.shard(num_shards = SAVING_BATCHS // 8, index = batch_i, contiguous = True).with_format('numpy')
				file_batch = np.concatenate(batch['tokens'])
				file[i:i + len(file_batch)] = file_batch
				i += len(file_batch)

			file.flush()
