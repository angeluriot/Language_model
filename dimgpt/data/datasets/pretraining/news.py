import re, json
from datasets import load_dataset, DownloadConfig, concatenate_datasets
from dimgpt.data.datasets import Dataset
from dimgpt.settings import *

class NewsDataset(Dataset):

	def __init__(self) -> None:

		super().__init__()

		self.training_part = 'pretraining'
		self.name = 'news'
		self.multiplier = 0.3

		print('Downloading News dataset...')

		news_fr = load_dataset(
			path = 'eckendoerffer/news_fr',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		news_fr = news_fr.map(
			lambda doc: {'text': self._clean_news_fr(doc['text'])},
			desc = 'Cleaning news_fr',
			num_proc = NUM_THREADS
		)

		wikinews = load_dataset(
			path = 'bigscience-data/roots_fr_wikinews',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		wikinews = wikinews.map(
			lambda doc: {'text': self._clean_wikinews(doc)},
			remove_columns = ['meta'],
			desc = 'Cleaning wikinews',
			num_proc = NUM_THREADS
		)

		cc_news = load_dataset(
			path = 'intfloat/multilingual_cc_news',
			name = 'fr',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		cc_news = cc_news.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['maintext']).strip()).strip()},
			remove_columns = ['title', 'maintext', 'url', 'date_publish'],
			desc = 'Cleaning cc_news',
			num_proc = NUM_THREADS
		)

		xlsum = load_dataset(
			path = 'csebuetnlp/xlsum',
			name = 'french',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		xlsum_summaries = xlsum.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['summary']).strip()).strip()},
			remove_columns = ['id', 'url', 'title', 'summary'],
			desc = 'Cleaning xlsum_summaries',
			num_proc = NUM_THREADS
		)

		xlsum = xlsum.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['text']).strip()).strip()},
			remove_columns = ['id', 'url', 'title', 'summary'],
			desc = 'Cleaning xlsum',
			num_proc = NUM_THREADS
		)

		mlsum = load_dataset(
			path = 'mlsum',
			name = 'fr',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		mlsum_summaries = mlsum.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['summary']).strip()).strip()},
			remove_columns = ['summary', 'topic', 'url', 'title', 'date'],
			desc = 'Cleaning mlsum_summaries',
			num_proc = NUM_THREADS
		)

		mlsum = mlsum.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['text']).strip()).strip()},
			remove_columns = ['summary', 'topic', 'url', 'title', 'date'],
			desc = 'Cleaning mlsum',
			num_proc = NUM_THREADS
		)

		orange_sum = load_dataset(
			path = 'orange_sum',
			name = 'title',
			split = 'train+validation+test',
			download_config = DownloadConfig(max_retries = 10)
		)

		orange_sum = orange_sum.map(
			lambda doc: {'text': (str(doc['summary']).strip() + '\n\n' + str(doc['text']).strip()).strip()},
			remove_columns = ['summary'],
			desc = 'Cleaning orange_sum',
			num_proc = NUM_THREADS
		)

		covid_news = load_dataset(
			path = 'gustavecortal/fr_covid_news',
			split = 'train',
			download_config = DownloadConfig(max_retries = 10)
		)

		covid_news = covid_news.map(
			lambda doc: {'text': (str(doc['title']).strip() + '\n\n' + str(doc['text']).strip()).strip()},
			remove_columns = ['title', 'description', 'domain', 'url', 'labels'],
			desc = 'Cleaning covid_news',
			num_proc = NUM_THREADS
		)

		self.dataset = concatenate_datasets([news_fr, wikinews, cc_news, xlsum, xlsum_summaries, mlsum, mlsum_summaries, orange_sum, covid_news])
		self.dataset = self.dataset.filter(lambda doc: len(str(doc['text']).strip()) >= MIN_DOCUMENT_SIZE)
		self.size['train'] = 0

		for doc in self.dataset:
			self.size['train'] += len(str(doc['text']).strip())

		print(f'News dataset downloaded: {len(self.dataset):,} documents | {self.size["train"]:,} characters')


	def _clean_news_fr(self, text: str) -> str:

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


	def _clean_wikinews(self, document) -> str:

		meta = str(document['meta']).strip()
		start = meta.find(", 'title': ") + 12
		end = meta.find(", 'type':") - 1

		if start != 11 and end != -2:
			title = meta[start:end].strip()
		else:
			title = ''

		text = str(document['text']).strip()

		if len(text) < 32:
			return text

		index = text[:30].find('–')

		if index != -1:
			text = text[index + 1:]

		output = title + '\n\n' + text.strip()

		return output.strip()