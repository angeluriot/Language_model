import re
from datasets import load_dataset, DownloadConfig, concatenate_datasets
from dimgpt.data.datasets import Dataset
from dimgpt.settings import *

class OthersDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'others'
		self.multiplier = 2.0

		print('Downloading Others dataset...')

		ted_talks = load_dataset(
			path = 'bigscience-data/roots_fr_ted_talks_iwslt',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		ted_talks = ted_talks.remove_columns('meta')

		bloom_lm = load_dataset(
			path = 'sil-ai/bloom-lm',
			name = 'fra',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		bloom_lm = bloom_lm.remove_columns(['title', 'license', 'pageCount', 'bookInstanceId', 'bookLineage'])

		self.dataset = concatenate_datasets([ted_talks, bloom_lm])
		self.dataset = self.dataset.filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

		print(f'Others dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')
