import os, random, csv, json, pytz
from datetime import datetime
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import xml.etree.ElementTree as ET

from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer
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


def import_xml_dataset(path: str) -> list[list[str]]:

	print('Importing dataset...')

	data = ET.parse(path).getroot()

	print('Parsing dataset...')

	dataset = []

	for document in data:

		dataset.append([])

		for message in document:
			dataset[-1].append(message.text)

	return dataset


def import_csv_dataset(path: str) -> list[list[str]]:

	print('Importing dataset...')

	with open(path, encoding = 'utf-8') as f:

		data = csv.reader(f, delimiter = ',')

		print('Parsing dataset...')

		dataset = []

		for row in data:
			dataset.append(row[1])

		dataset = dataset[1:]

	return dataset


def import_json_dataset(path: str) -> list[list[str]]:

	dataset = []

	with open(os.path.join(path, 'tweets.json'), 'r', encoding = 'utf-8') as f:
		tweets = json.load(f)

	with open(os.path.join(path, 'deleted-tweets.json'), 'r', encoding = 'utf-8') as f:
		deleted_tweets = json.load(f)

	with open(os.path.join(path, 'direct-messages.json'), 'r', encoding = 'utf-8') as f:
		direct_messages = json.load(f)

	tweets += deleted_tweets

	for tweet in tweets:
		doc = str(tweet['tweet']['full_text'])
		if doc.startswith('RT @'):
			continue
		dataset.append([
			doc,
			datetime.strptime(str(tweet['tweet']['created_at']).strip(), '%a %b %d %H:%M:%S %z %Y').astimezone(pytz.utc)
		])

	for conv in direct_messages:
		for msg in conv['dmConversation']['messages']:
			if 'messageCreate' not in msg or str(msg['messageCreate']['senderId']).strip() != '4736048116':
				continue
			dataset.append([
				str(msg['messageCreate']['text']),
				datetime.strptime(str(msg['messageCreate']['createdAt']).strip(), '%Y-%m-%dT%H:%M:%S.%fZ').astimezone(pytz.utc)
			])

	dataset.sort(key = lambda x: x[1])
	dataset = [doc for doc, _ in dataset]

	return dataset


def prepare_hf_text_dataset(tokenizer: Tokenizer, dataset: DatasetDict, name: str) -> None:

	def process(document: dict, tokenizer: Tokenizer):

		text = clean_document(document['text'])
		ids = tokenizer.encode(text)

		return {'ids': ids, 'size': len(ids)}

	split_dataset = dataset['train'].train_test_split(test_size = VAL_RATIO, shuffle = True)
	split_dataset['val'] = split_dataset.pop('test')

	tokenized = split_dataset.map(
		lambda doc: process(doc, tokenizer),
		remove_columns = ['text'],
		desc = f'Tokenizing {name}',
		num_proc = NUM_THREADS,
	)

	for split, documents in tokenized.items():

		size = np.sum(documents['size'], dtype = np.uint64)
		path = os.path.join(DATA_DIR, f'{name}_{split}.bin')
		file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for batch_i in tqdm(range(SAVING_BATCHS), desc = f'Saving {name}'):

			batch = documents.shard(num_shards = SAVING_BATCHS, index = batch_i, contiguous = True).with_format('numpy')
			file_batch = np.concatenate(batch['ids'])
			file[i:i + len(file_batch)] = file_batch
			i += len(file_batch)

		file.flush()


def prepare_hf_instructions_dataset(tokenizer: Tokenizer, dataset: DatasetDict, name: str) -> None:

	def process(document: dict, tokenizer: Tokenizer):

		context = clean_string(document['input'].strip())
		instruction = clean_string(document['instruction'].strip())
		answer = clean_string(document['output'].strip())

		if len(instruction) >= 3 and not instruction[0].isalpha() and instruction[1] == ' ' and instruction[2].isalpha():
			instruction = instruction[2:]

		if len(context) == 0:
			text = f'{instruction}␃{answer}␄'
		else:
			text = f'{context}↲↲{instruction}␃{answer}␄'

		while '↲↲↲' in text:
			text = text.replace('↲↲↲', '↲↲')

		ids = tokenizer.encode(text)

		return {'ids': ids, 'size': len(ids)}

	split_dataset = dataset['train'].train_test_split(test_size = VAL_RATIO, shuffle = True)
	split_dataset['val'] = split_dataset.pop('test')

	tokenized = split_dataset.map(
		lambda doc: process(doc, tokenizer),
		desc = f'Tokenizing {name}',
		num_proc = NUM_THREADS,
	)

	for split, documents in tokenized.items():

		size = np.sum(documents['size'], dtype = np.uint64)
		path = os.path.join(DATA_DIR, f'{name}_{split}.bin')
		file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for batch_i in tqdm(range(SAVING_BATCHS), desc = f'Saving {name}'):

			batch = documents.shard(num_shards = SAVING_BATCHS, index = batch_i, contiguous = True).with_format('numpy')
			file_batch = np.concatenate(batch['ids'])
			file[i:i + len(file_batch)] = file_batch
			i += len(file_batch)

		file.flush()


def prepare_xml_dataset(tokenizer: Tokenizer, dataset: list[list[str]], name: str) -> None:

	for i in tqdm(range(len(dataset)), desc = f'Tokenizing {name}'):
		document = clean_chat(dataset[i])
		dataset[i] = tokenizer.encode(document)

	random.shuffle(dataset)

	split = int(len(dataset) * VAL_RATIO * 10)
	splits = {
		'train': dataset[:-split],
		'val': dataset[-split:]
	}

	for split, documents in tqdm(splits.items(), desc = f'Saving {name}'):

		size = np.sum([len(doc) for doc in documents], dtype = np.uint64)
		path = os.path.join(DATA_DIR, f'{name}_{split}.bin')
		file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for doc in documents:
			file[i:i + len(doc)] = doc
			i += len(doc)

		file.flush()


def prepare_csv_dataset(tokenizer: Tokenizer, dataset: list[list[str]], name: str) -> None:

	for i in tqdm(range(len(dataset)), desc = f'Tokenizing {name}'):
		document = clean_document(dataset[i])
		dataset[i] = tokenizer.encode(document)

	random.shuffle(dataset)

	split = int(len(dataset) * VAL_RATIO * 10)
	splits = {
		'train': dataset[:-split],
		'val': dataset[-split:]
	}

	for split, documents in tqdm(splits.items(), desc = f'Saving {name}'):

		size = np.sum([len(doc) for doc in documents], dtype = np.uint64)
		path = os.path.join(DATA_DIR, f'{name}_{split}.bin')
		file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for doc in documents:
			file[i:i + len(doc)] = doc
			i += len(doc)

		file.flush()


def prepare_json_dataset(tokenizer: Tokenizer, dataset: list[list[str]], name: str) -> None:

	for i in tqdm(range(len(dataset)), desc = f'Tokenizing {name}'):
		document = clean_document(dataset[i].strip())
		dataset[i] = tokenizer.encode(document)

	splits = {
		'train': [],
		'val': []
	}

	for doc in dataset:
		if random.random() < VAL_RATIO * 10:
			splits['val'].append(doc)
		else:
			splits['train'].append(doc)

	for split, documents in tqdm(splits.items(), desc = f'Saving {name}'):

		size = np.sum([len(doc) for doc in documents], dtype = np.uint64)
		path = os.path.join(DATA_DIR, f'{name}_{split}.bin')
		file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
		i = 0

		for doc in documents:
			file[i:i + len(doc)] = doc
			i += len(doc)

		file.flush()
