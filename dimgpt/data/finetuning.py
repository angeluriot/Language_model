import os, pickle
from datasets import load_dataset, DownloadConfig
from tqdm import tqdm

from dimgpt.settings import *
from dimgpt.data.tokenizer import Tokenizer


class Finetuning:

	def __init__(self):

		self.import_dataset()


	def import_dataset(self) -> None:

		if os.path.exists(os.path.join(DATA_DIR, 'finetuning', 'chatbot_conversations_train.pkl')):
			return

		self.datasets = {}

		for name in ['human_conversations', 'chatbot_conversations', 'dimension_gpt_conversations', 'human_preprompts', 'chatbot_preprompts', 'dimension_gpt_preprompts']:

			self.datasets[name] = load_dataset(
				path = 'angeluriot/DimensionGPT_instruct',
				name = name,
				download_config = DownloadConfig(max_retries = 10),
				num_proc = NUM_THREADS
			)


	def document_to_tokens(self, document: dict[str, str], tokenizer: Tokenizer, preprompts: bool) -> dict[str, list[int] | int]:

		if preprompts:

			tokens = [tokenizer.system_token, *tokenizer.encode(document['preprompt'])]

			return {'tokens': tokens, 'size': len(tokens)}

		tokens = []

		for msg in document['conversation']:

			if msg['role'] == 'user':
				tokens.append(tokenizer.user_token)
			elif msg['role'] == 'assistant':
				tokens.append(tokenizer.assistant_token)
			else:
				tokens.append(tokenizer.human_token)

			tokens.extend(tokenizer.encode(msg['text']))

		return {'tokens': tokens, 'size': len(tokens)}


	def save(self, tokenizer: Tokenizer) -> None:

		if os.path.exists(os.path.join(DATA_DIR, 'finetuning', 'chatbot_conversations_train.pkl')):
			return

		if not os.path.exists(os.path.join(DATA_DIR, 'finetuning')):
			os.makedirs(os.path.join(DATA_DIR, 'finetuning'))

		for name, dataset in self.datasets.items():

			if name == 'chatbot_conversations':
				dataset = dataset['train'].train_test_split(test_size = FINETUNING_VAL_RATIO, shuffle = True)
				dataset['val'] = dataset.pop('test')

			tokenized = dataset.map(
				lambda doc: self.document_to_tokens(doc, tokenizer, name.endswith('preprompts')),
				desc = f'Tokenizing {name}',
				num_proc = NUM_THREADS
			)

			for split, documents in tokenized.items():

				docs = []

				for doc in tqdm(documents, desc = f'Saving finetuning dataset {name}_{split}'):
					docs.append(doc['tokens'])

				with open(os.path.join(DATA_DIR, 'finetuning', f'{name}_{split}.pkl'), 'wb') as file:
					pickle.dump(docs, file)

