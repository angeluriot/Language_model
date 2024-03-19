from datasets import load_dataset, DownloadConfig, concatenate_datasets
from dimgpt.data.datasets import Dataset
from dimgpt.settings import *

class InstitutionsDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'institutions'
		self.multiplier = 2.0

		print('Downloading Institutions dataset...')

		europarl = load_dataset(
			path = 'bigscience-data/roots_fr_the_pile_europarl',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		qr_an = load_dataset(
			path = 'cassandra-themis/QR-AN',
			name = 'qran_generation',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		qr_an = qr_an.map(
			lambda doc: {'text': (str(doc['question']).strip() + '\n\n' + str(doc['answer']).strip()).strip()},
			remove_columns = ['question', 'answer'],
			desc = 'Cleaning QR-AN',
			num_proc = NUM_THREADS
		)

		self.dataset = concatenate_datasets([europarl, qr_an])
		self.dataset = self.dataset.filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

		print(f'Institutions dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')