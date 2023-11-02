import os, random, json, pytz
from datetime import datetime
import numpy as np
from tqdm import tqdm

from dimgpt.datasets.dataset import Dataset
from dimgpt.settings import *
from dimgpt.data.clean import *
from dimgpt.data.tokenizer import Tokenizer


class MyTweets(Dataset):

	def __init__(self):

		super().__init__('my_tweets')


	def import_dataset(self) -> None:

		print('Importing dataset...')

		path = 'D:/Datasets/My_tweets'
		self.dataset = []

		with open(os.path.join(path, 'tweets.json'), 'r', encoding = 'utf-8') as f:
			tweets = json.load(f)

		with open(os.path.join(path, 'deleted-tweets.json'), 'r', encoding = 'utf-8') as f:
			deleted_tweets = json.load(f)

		with open(os.path.join(path, 'direct-messages.json'), 'r', encoding = 'utf-8') as f:
			direct_messages = json.load(f)

		print('Parsing dataset...')

		tweets += deleted_tweets

		for tweet in tweets:
			doc = str(tweet['tweet']['full_text'])
			if doc.startswith('RT @'):
				continue
			self.dataset.append([
				doc,
				datetime.strptime(str(tweet['tweet']['created_at']).strip(), '%a %b %d %H:%M:%S %z %Y').astimezone(pytz.utc)
			])

		for conv in direct_messages:
			for msg in conv['dmConversation']['messages']:
				if 'messageCreate' not in msg or str(msg['messageCreate']['senderId']).strip() != '4736048116':
					continue
				self.dataset.append([
					str(msg['messageCreate']['text']),
					datetime.strptime(str(msg['messageCreate']['createdAt']).strip(), '%Y-%m-%dT%H:%M:%S.%fZ').astimezone(pytz.utc)
				])

		self.dataset.sort(key = lambda x: x[1])
		self.dataset = [doc for doc, _ in self.dataset]


	def save(self, tokenizer: Tokenizer) -> None:

		for i in tqdm(range(len(self.dataset)), desc = f'Tokenizing {self.name}'):
			self.dataset[i] = self.document_to_tokens(self.dataset[i], tokenizer)

		splits = {
			'train': [],
			'val': []
		}

		for doc in self.dataset:
			if random.random() < VAL_RATIO * 50:
				splits['val'].append(doc)
			else:
				splits['train'].append(doc)

		folder = os.path.join(DATA_DIR, self.name)

		if not os.path.exists(folder):
			os.mkdir(folder)

		for split, documents in splits.items():

			size = np.sum([doc['size'] for doc in documents], dtype = np.uint64)
			path = os.path.join(folder, f'{split}.bin')
			file = np.memmap(path, dtype = np.uint16, mode = 'w+', shape = (size,))
			i = 0

			for doc in tqdm(documents, desc = f'Saving {self.name} {split}'):
				tokens = doc['tokens']
				file[i:i + len(tokens)] = tokens
				i += len(tokens)

			file.flush()

		self.save_ids()