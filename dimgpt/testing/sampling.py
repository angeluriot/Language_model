import torch
import numpy as np
import numpy.typing as npt

from dimgpt.training.model import Model
from dimgpt.data.tokenizer import Tokenizer
from dimgpt.settings import *


class Sampler():

	def __init__(self, model: Model, tokenizer: Tokenizer):

		self.model = model
		self.tokenizer = tokenizer


	def get_probabilities(self, input: list[int]) -> npt.NDArray[np.float32]:

		probabilities = self.model(torch.tensor([input], dtype = torch.long, device = DEVICE), only_last = True)[0].detach().to('cpu').numpy()
		probabilities = np.log(np.exp(probabilities) / np.sum(np.exp(probabilities)))

		return probabilities


	def sample(self, input: list[int], temperature: float = 1.0, top_p: float = 1.0, no_repeat_strength: float = 0.0) -> int:

		probabilities = self.get_probabilities(input)
		proximity = MAX_CONTEXT

		for i in reversed(range(max(len(input) - MAX_CONTEXT, 0), len(input))):
			strength = no_repeat_strength * (proximity / MAX_CONTEXT)
			probabilities[input[i]] *= (1 + strength)
			proximity -= 1

		if temperature == 0.0:
			return np.argmax(probabilities)

		probabilities /= temperature
		probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

		sorted_indices = np.argsort(-probabilities)
		cumsum_probabilites = np.cumsum(probabilities[sorted_indices])
		cutoff_index = np.searchsorted(cumsum_probabilites, max(top_p, cumsum_probabilites[0] + 1e-6))
		temp = np.zeros_like(probabilities)
		temp[sorted_indices[:cutoff_index]] = probabilities[sorted_indices[:cutoff_index]]
		probabilities = temp / np.sum(temp)

		return np.random.choice(range(len(probabilities)), p = probabilities)


	def generate(self, input: str, max_length: int, keep_input = False, temperature: float = 1.0,
		top_p: float = 1.0, no_repeat: float = 0.0, verbose: bool = False, max_print_line_length = 0) -> str:

		self.model.eval()

		with torch.no_grad():

			input = self.tokenizer.encode(input).tolist()
			output = []
			to_print = []
			last_line_length = 0

			if len(input) == 0:
				input = [EOT_INDEX]
			elif input[0] != EOT_INDEX:
				input = [EOT_INDEX] + input

			if keep_input:
				output = input[1:].copy()
				to_print = input[1:].copy()
				text = self.tokenizer.decode(to_print)
				last_line_length = len(text) - 1 - text.rfind('\n')

			for _ in range(max_length):

				index = self.sample(input, temperature, top_p, no_repeat)
				input.append(index)
				output.append(index)
				to_print.append(index)

				if verbose:

					text = self.tokenizer.decode(to_print)

					if '\n' in text:
						last_line_length = len(text) - 1 - text.rfind('\n')
					else:
						last_line_length += len(text)

					if max_print_line_length > 0 and last_line_length >= max_print_line_length and text.startswith(' '):
						print()
						text = text[1:]
						last_line_length = 0

					print(text, end = '')
					to_print = []

		return self.tokenizer.decode(output)
