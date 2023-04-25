from tokenizers import *
from settings import *

class PreTokenizer:

	def split(self, i: int, normalized_string: NormalizedString):

		# Split control characters

		words = [str(normalized_string)]
		temp = []
		i = 0

		for control in CONTROL_CHARS:
			for i in range(len(words)):

				parts = words[i].split(control)

				for part in parts:
					if len(part) > 0:
						temp.append(part)

					temp.append(control)

				if len(parts) > 0:
					temp.pop()

			words = temp
			temp = []

		# Split words

		for i in range(len(words)):

			if words[i] in CONTROL_CHARS:
				temp.append(words[i])
				continue

			word = ''

			for j in range(len(words[i])):

				if len(word) == 0:
					word += words[i][j]
					continue

				if word[-1] != ' ' and words[i][j] == ' ':
					temp.append(word)
					word = words[i][j]
					continue

				if word[-1].isalpha() and not words[i][j].isalpha():
					temp.append(word)
					word = words[i][j]
					continue

				if word[-1] != ' ' and not word[-1].isalpha() and words[i][j].isalpha():
					temp.append(word)
					word = words[i][j]
					continue

				word += words[i][j]

			if len(word) > 0:
				temp.append(word)

		words = temp
		temp = []

		# Split multiple spaces

		for i in range(len(words)):

			j = 0

			while j < len(words[i]) and words[i][j] == ' ':
				j += 1

			if j > 1 and j < len(words[i]):
				temp.append(words[i][:j - 1])
				temp.append(words[i][j - 1:])

			else:
				temp.append(words[i])

		words = temp
		words = [NormalizedString(word) for word in words]

		return words


	def pre_tokenize(self, pretok: PreTokenizedString):
		pretok.split(self.split)
