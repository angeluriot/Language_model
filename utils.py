import numpy as np

from settings import *

def decode(tokenizer, tokens):

	text = ''

	for i in tokens:
		t = tokenizer.decode([i], False)
		if t == '<eom>':
			text += '\n'
		elif t == '<eod>':
			text += '\n-------------------\n'
		else:
			text += t

	return text
