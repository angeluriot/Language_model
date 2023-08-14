import os, random
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import xml.etree.ElementTree as ET
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

from dimgpt.data.clean import *
from dimgpt.settings import *


def create_tokenizer_data(cc100: DatasetDict, wikipedia: DatasetDict) -> tuple[int]:

	with open(os.path.join(DATA_DIR, 'tokenizer_data.txt'), 'w', encoding = 'utf-8') as file:

		file.truncate(0)
		chars = {}
		nb_from_cc100 = 0
		nb_from_wikipedia = 1
		pbar = tqdm(total = TOKENIZER_DATA_SIZE)

		while True:

			if nb_from_cc100 / nb_from_wikipedia < CC100_NB_CHARS / WIKIPEDIA_NB_CHARS:

				document = cc100['train'][random.randint(0, CC100_NB_DOCUMENTS - 1)]
				document = clean_document(document['text'])

				if len(document) > 0:
					file.write(document)
					nb_from_cc100 += len(document)

			else:

				document = wikipedia['train'][random.randint(0, WIKIPEDIA_NB_DOCUMENTS - 1)]
				document = clean_document(document['text'])

				if len(document) > 0:
					file.write(document)
					nb_from_wikipedia += len(document)

			for char in document:
				chars[char] = chars.get(char, 0) + 1

			pbar.update(len(document))

			if nb_from_cc100 + nb_from_wikipedia >= TOKENIZER_DATA_SIZE:
				break

		document = ' ' + ' '.join(list(POSSIBLE_CHARS))
		file.write(document)

		for char in document:
			chars[char] = chars.get(char, 0) + 1

		pbar.close()

	chars = sorted(chars.items(), key = lambda item: item[1], reverse = True)
	chars = [char for char, _ in chars]

	return nb_from_cc100, nb_from_wikipedia, chars


def prepare_dataset(dataset: DatasetDict) -> None:

	split_dataset = dataset['train'].train_test_split(test_size = 0.0005, shuffle = True)
	split_dataset['val'] = split_dataset.pop('test')


