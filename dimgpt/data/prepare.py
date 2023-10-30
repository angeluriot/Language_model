import os
from tqdm import tqdm
import numpy as np

from dimgpt.data.clean import *
from dimgpt.settings import *
from dimgpt.datasets.dataset import Dataset


def create_tokenizer_data(datasets: list[tuple[Dataset, float]], epsilon: float = 1e-8) -> tuple[list[int], list[str]]:

	datasets, target_ratios = zip(*datasets)
	total_sum = sum(target_ratios)
	target_ratios = [ratio / total_sum for ratio in target_ratios]

	if not os.path.exists(DATA_DIR):
		os.makedirs(DATA_DIR)

	with open(os.path.join(DATA_DIR, 'tokenizer_data.txt'), 'w', encoding = 'utf-8') as file:

		file.truncate(0)
		chars = {}
		current_sizes = [0] * len(datasets)
		pbar = tqdm(total = TOKENIZER_DATA_SIZE)

		while True:

			current_ratios = [size / (sum(current_sizes) + epsilon) for size in current_sizes]
			ratio_errors = [target_ratios[i] - current_ratios[i] for i in range(len(datasets))]
			dataset_index = np.argmax(ratio_errors)
			dataset = datasets[dataset_index]

			document = dataset.get_document()

			if len(document) == 0:
				continue

			file.write(document)
			current_sizes[dataset_index] += len(document)

			for char in document:
				chars[char] = chars.get(char, 0) + 1

			pbar.update(len(document))

			if sum(current_sizes) >= TOKENIZER_DATA_SIZE:
				break

		document = ' ' + ' '.join(list(POSSIBLE_CHARS))
		file.write(document)

		for char in document:
			chars[char] = chars.get(char, 0) + 1

		pbar.close()

	chars = sorted(chars.items(), key = lambda item: item[1], reverse = True)
	chars = [char for char, _ in chars]

	return current_sizes, chars
