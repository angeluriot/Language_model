import regex

from tokenizers import *
from dimgpt.settings import *
from dimgpt.utils import *


def split(text: str) -> list[str]:

	if text == '':
		return []

	# Split in words

	reg = r'(' + r'|'.join(CONTROL_CHARS) + r'|\d+|\s+|\p{L}+|[^\d\p{L}\s' + r''.join([f'[{i}]' for i in CONTROL_CHARS]) + r']+)'
	words = regex.split(reg, text, flags = regex.UNICODE, concurrent = False)
	words = list(filter(None, words))

	# Add beginning spaces

	temp = []
	i = 0

	while i < len(words) - 1:

		if words[i] == ' ':
			temp.append(' ' + words[i + 1])
			i += 2
			continue

		if words[i].endswith(' '):
			spaces = list(filter(None, split_keep(words[i][:-1], '    ')))
			temp.extend(spaces + [' ' + words[i + 1]])
			i += 2
			continue

		temp.append(words[i])
		i += 1

	if i == len(words) - 1 and words[i].endswith(' '):
		spaces = list(filter(None, split_keep(words, '    ')))
		temp.extend(spaces)

	words = temp
	words = list(filter(None, words))

	return words


class PreTokenizer:

	def split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:

		print('Pretokenize...')

		words = split(str(normalized_string))
		words = [NormalizedString(word) for word in words]

		print('Nb words:', '{:,.0f}'.format(len(words)))
		print('Merges...')

		return words


	def pre_tokenize(self, pretok: PreTokenizedString) -> None:

		pretok.split(self.split)
