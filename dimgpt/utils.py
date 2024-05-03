import random, platform, psutil, time
import datetime as dt
import numpy as np
import torch
from sys import exit

from dimgpt.settings import *


# Reset the random seed
def reset_rand() -> None:

	now = dt.datetime.now()
	milliseconds_since_midnight = (now.hour * 3600 + now.minute * 60 + now.second) * 1000 + now.microsecond // 1000
	random.seed(milliseconds_since_midnight)
	np.random.seed(milliseconds_since_midnight)
	torch.manual_seed(milliseconds_since_midnight)


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


def split_keep(text: str, delimiter: str) -> list[str]:

	words = text.split(delimiter)

	temp = []

	for i in range(len(words) - 1):
		temp.extend([words[i], delimiter])

	temp.append(words[-1])

	return temp


class Timer:

	def __init__(self, wait_steps: int = 0, num_steps: int = 1, exit_on_end: bool = False):

		self.wait_steps = wait_steps
		self.num_steps = num_steps
		self.exit_on_end = exit_on_end
		self.times = [0.0] * num_steps
		self.wait_step = 0
		self.step = 0


	def __enter__(self):

		if self.wait_step < self.wait_steps:
			return

		self.times[self.step] = time.time()


	def __exit__(self, exc_type, exc_value, traceback):

		if self.wait_step < self.wait_steps:
			self.wait_step += 1
			return

		self.times[self.step] = time.time() - self.times[self.step]
		self.step += 1

		if self.step >= self.num_steps:

			print(f'\nDuration: {sum(self.times) / self.num_steps:.2f}s')

			if self.exit_on_end:
				exit(0)