import re
from datasets import load_dataset, DownloadConfig, concatenate_datasets
from dimgpt.data.datasets import Dataset
from dimgpt.settings import *

class WikipediaDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'wikipedia'
		self.multiplier = 4.0

		print('Downloading Wikipedia dataset...')

		wikipedia_fr = load_dataset(
			path = 'eckendoerffer/wikipedia_fr',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		wikipedia_fr = wikipedia_fr.map(
			lambda doc: {'text': self._clean_wikipedia_fr(doc['text'])},
			desc = 'Cleaning wikipedia_fr',
			num_proc = NUM_THREADS
		)

		roots_fr_wikipedia = load_dataset(
			path = 'bigscience-data/roots_fr_wikipedia',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		roots_fr_wikipedia = roots_fr_wikipedia.remove_columns('meta')

		roots_fr_wikivoyage = load_dataset(
			path = 'bigscience-data/roots_fr_wikivoyage',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		roots_fr_wikivoyage = roots_fr_wikivoyage.remove_columns('meta')

		self.dataset = concatenate_datasets([wikipedia_fr, roots_fr_wikipedia, roots_fr_wikivoyage])
		self.dataset = self.dataset.filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

		print(f'Wikipedia dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')


	def _clean_wikipedia_fr(self, text: str) -> str:

		text = text.replace(' ,', ',')
		text = text.replace(' .', '.')
		text = text.replace(' )', ')')
		text = text.replace('( ', '(')
		text = text.replace(' ]', ']')
		text = text.replace('[ ', '[')

		text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)

		array = list(text)
		start = True

		for i in range(len(array)):
			if array[i] == '"':
				array[i] = '«' if start else '»'
				start = not start

		return ''.join(array)