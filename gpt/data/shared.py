import os, random
import numpy as np
import numpy.typing as npt
from datasets import load_dataset

from gpt.settings import *
from gpt.data import CC100


def clean(text: str) -> str:

	text = text.strip()
	text = text.replace('’', "'"); text = text.replace('‘', "'"); text = text.replace('“', '"'); text = text.replace('”', '"'); text = text.replace('„', '"'); text = text.replace('´', "'"); text = text.replace('`', "'"); text = text.replace('ʹ', "'")
	text = text.replace('—', '-'); text = text.replace('–', '-'); text = text.replace('−', '-'); text = text.replace('⁻', '-'); text = text.replace('―', '-'); text = text.replace('ˉ', '-'); text = text.replace('¯', '-')
	text = text.replace('…', '...')
	text = text.replace('⁄', '/')
	text = text.replace('＊', '*')
	text = text.replace('\t', '    ')
	text = text.replace('º', '°')
	text = text.replace('＾', '^')
	text = text.replace('（', ' ('); text = text.replace('）', ') ')
	text = text.replace('［', ' ['); text = text.replace('］', '] ')
	text = text.replace('｛', ' {'); text = text.replace('｝', '} ')
	text = text.replace('＜', '<'); text = text.replace('＞', '>')
	text = text.replace('＝', '='); text = text.replace('＋', '+'); text = text.replace('％', '%'); text = text.replace('＄', '$'); text = text.replace('＃', '#')
	text = text.replace('¸', ',')

	ban_space_chars = [' ', ' ', ' ', ' ', '­', '️', '‍']
	ban_chars = ['̃', '̈ ​', '﻿', '́', '͟', '​', '̂', '͡', '‎', '︎', '̀', '͜', '̶', '̿', '̲', '̯', '̅', '‏', '', '‪', '‬', '‮', '卐']

	for c in ban_space_chars:
		text = text.replace(c, ' ')

	for c in ban_chars:
		text = text.replace(c, '')

	while ' \n' in text:
		text = text.replace(' \n', '\n')

	text = text.replace('\n', '<nl>')

	if len(text) >= 2 and text[0] == '-' and text[1].isalpha():
		text = '- ' + text[1:]

	return text


def get_data():

	if DATASET == 'CC100':
		return CC100.get_data()
