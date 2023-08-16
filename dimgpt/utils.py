import random, platform, psutil
import datetime as dt
import numpy as np
import torch

from dimgpt.settings import *


# Reset the random seed
def reset_rand() -> None:

	now = dt.datetime.now()
	seconds_since_midnight = int((now - now.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds())
	random.seed(seconds_since_midnight)
	np.random.seed(seconds_since_midnight)
	torch.manual_seed(seconds_since_midnight)


# Check if there is a GPU available
def check_gpu() -> None:

	if GPU_ENABLED:
		torch.cuda.empty_cache()
		nb_gpu = torch.cuda.device_count()
		memory = torch.cuda.mem_get_info()[0] / 1024 ** 3
		print(f'{nb_gpu} GPU {"are" if nb_gpu > 1 else "is"} available! Using GPU: "{torch.cuda.get_device_name()}" ({memory:.2f} GB available)')

	else:
		memory = psutil.virtual_memory().available / 1024 ** 3
		print(f'No GPU available... Using CPU: "{platform.processor()}" ({memory:.2f} GB available)')


def save_text_array(array: list[str], path: str) -> None:

	with open(path, 'w', encoding = 'utf-8') as f:

		f.truncate(0)

		for i in range(len(array)):

			f.write(array[i])

			if i != len(array) - 1:
				f.write('\n')


def load_text_array(path: str) -> list[str]:

	with open(path, 'r', encoding = 'utf-8') as f:

		return f.read().split('\n')


def print_tokens(tokens: list[str]) -> None:

	for i in range(len(tokens)):

		if i % 4 == 0:
			print('\033[91m' + tokens[i].replace(' ', '_') + '\033[0m', end = '')

		elif i % 4 == 1:
			print('\033[94m' + tokens[i].replace(' ', '_') + '\033[0m', end = '')

		elif i % 4 == 2:
			print('\033[92m' + tokens[i].replace(' ', '_') + '\033[0m', end = '')

		else:
			print('\033[93m' + tokens[i].replace(' ', '_') + '\033[0m', end = '')

	print()