from settings import *


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
