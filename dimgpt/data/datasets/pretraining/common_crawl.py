import os, json
from datasets import load_dataset, DownloadConfig
from dimgpt.data.datasets import Dataset
from dimgpt.settings import *

class CommonCrawlDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'common_crawl'
		self.multiplier = 1.0

		print('Downloading Common Crawl dataset...')

		if not os.path.exists(os.path.join(DATA_DIR, self.training_part, self.name, 'raw.json')):

			dataset = load_dataset(
				path = 'ontocord/CulturaY',
				name = 'fr',
				split = 'train',
				download_config = DownloadConfig(max_retries = 10),
				streaming = True
			)

			os.makedirs(os.path.join(DATA_DIR, self.training_part, self.name), exist_ok = True)

			with open(os.path.join(DATA_DIR, self.training_part, self.name, 'raw.json'), 'w', encoding = 'utf-8') as file:

				file.truncate(0)
				i = 0
				self.size['train'] = 0

				for record in dataset:

					text = str(record['text']).strip()

					if len(text) < MIN_DOCUMENT_SIZE:
						continue

					file.write(json.dumps({'text': text}, ensure_ascii = False) + '\n')

					self.size['train'] += len(text)
					i += 1

					if i % 1_000 == 0:
						print(f'{i:,} documents | {self.size["train"]:,} characters            ', end = '\r')

					if self.size['train'] >= 150_000_000_000:
						break

		self.dataset = load_dataset(
			path = 'json',
			split = 'train',
			data_files = os.path.join(DATA_DIR, self.training_part, self.name, 'raw.json'),
			num_proc = NUM_THREADS
		)

		if self.size['train'] == 0:
			self.size['train'] = 150_000_000_000

		print(f'Common Crawl dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')
