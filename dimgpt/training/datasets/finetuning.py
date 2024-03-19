import os, random, pickle
import torch
import numpy as np

from dimgpt.data.tokenizer import Tokenizer
from dimgpt.settings import *
from dimgpt.training.datasets import Dataset


class FinetuningDataset(Dataset):

	def __init__(self, tokenizer: Tokenizer):

		self.tokenizer = tokenizer

		self.train_data = {
			'human': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'human_conversations_train.pkl'), 'rb')),
			'chatbot': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'chatbot_conversations_train.pkl'), 'rb')),
			'dimension_gpt': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'dimension_gpt_conversations_train.pkl'), 'rb'))
		}

		self.val_data =  pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'chatbot_conversations_val.pkl'), 'rb'))

		self.train_preprompts = {
			'human': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'human_preprompts_train.pkl'), 'rb')),
			'chatbot': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'chatbot_preprompts_train.pkl'), 'rb')),
			'dimension_gpt': pickle.load(open(os.path.join(DATA_DIR, 'finetuning', 'dimension_gpt_preprompts_train.pkl'), 'rb'))
		}

		self.final_preprompt = [self.tokenizer.system_token, *self.tokenizer.encode(PREPROMPT)]

		self.preprompt_ratios = {
			'human': HUMAN_PREPROMPT_RATIOS,
			'chatbot': CHATBOT_PREPROMPT_RATIOS,
			'dimension_gpt': DIMENSION_GPT_PREPROMPT_RATIOS
		}

		h = [len(i) for i in self.train_data['human']]
		c = [len(i) for i in self.train_data['chatbot']]
		d = [len(i) for i in self.train_data['dimension_gpt']]

		self.train_data_p = {
			'human': (np.array(h) / np.sum(h)).tolist(),
			'chatbot': (np.array(c) / np.sum(c)).tolist(),
			'dimension_gpt': (np.array(d) / np.sum(d)).tolist()
		}

		print(sum(self.train_data_p['human']))
		print(sum(self.train_data_p['chatbot']))
		print(sum(self.train_data_p['dimension_gpt']))

		v = [len(i) for i in self.val_data]

		self.val_data_p = (np.array(v) / np.sum(v)).tolist()

		self.train_ids = {
			'human': list(range(len(self.train_data['human']))),
			'chatbot': list(range(len(self.train_data['chatbot']))),
			'dimension_gpt': list(range(len(self.train_data['dimension_gpt'])))
		}

		self.val_ids = list(range(len(self.val_data)))


	def train_size(self) -> int:

		return sum([len(self.train_data[key]) for key in self.train_data])


	def val_size(self) -> int:

		return len(self.val_data)


	def __get_strength(self, doc: list[int], val: bool) -> list[int]:

		assistant = False
		instruction_loss_strength = 0.0 if val else INSTRUCTION_LOSS_STRENGTH
		strength = []

		for token in doc:

			strength.append(1.0 if assistant else instruction_loss_strength)

			if token == self.tokenizer.user_token or token == self.tokenizer.end_of_text_token:
				assistant = False

			if token == self.tokenizer.assistant_token or token == self.tokenizer.human_token:
				assistant = True

		return strength


	def __get_document(self, val: bool, first: bool) -> tuple[list[int], list[int]]:

		if val:
			data_ids = self.val_ids
			data = self.val_data
			data_p = self.val_data_p

		else:
			data_split = np.random.choice(['human', 'chatbot', 'dimension_gpt'], p = SPLIT_RATIOS)
			data_ids = self.train_ids[data_split]
			data = self.train_data[data_split]
			data_p = self.train_data_p[data_split]

		if first:
			id = np.random.choice(data_ids, p = data_p)
			conversation = data[id]
		else:
			conversation = data[random.randint(0, len(data) - 1)]

		if val:
			xy = [self.tokenizer.start_of_text_token, *self.final_preprompt, *conversation, self.tokenizer.end_of_text_token]
			strength = self.__get_strength(xy, val)
			return xy, strength

		preprompt_ratio = self.preprompt_ratios[data_split]
		preprompt_split = np.random.choice(['human', 'chatbot', 'dimension_gpt', 'none'], p = preprompt_ratio)

		if preprompt_split != 'none':
			preprompt = self.train_preprompts[preprompt_split][random.randint(0, len(self.train_preprompts[preprompt_split]) - 1)]
			conversation = [*preprompt, *conversation]

		xy = [self.tokenizer.start_of_text_token, *conversation, self.tokenizer.end_of_text_token]
		strength = self.__get_strength(xy, val)

		return xy, strength


	def _get_random_document(self, val: bool) -> tuple[list[int], list[int]]:

		xy, strength = self.__get_document(val, False)

		return xy, strength


	def _get_tokens(self, val: bool) -> tuple[torch.Tensor, torch.Tensor]:

		xy, strength = self.__get_document(val, True)

		i = random.randint(0, len(xy) - 1)
		xy = xy[i:]
		strength = strength[i:]

		while len(xy) < MAX_CONTEXT + 1:

			_xy, _strength = self._get_random_document(val)

			xy.extend(_xy)
			strength.extend(_strength)

		xy = xy[0:MAX_CONTEXT + 1]
		strength = strength[0:MAX_CONTEXT + 1]

		return xy, strength