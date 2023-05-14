import pickle
import tensorflow as tf
from keras import backend
from keras.optimizers import Optimizer
from gradient_accumulator import GradientAccumulateModel

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


def save_state(optimizer: Optimizer, path: str) -> None:

	variables = optimizer.variables()
	weights = [backend.get_value(var) for var in variables]
	pickle.dump(weights, open(path, 'wb'))


def load_state(optimizer: Optimizer, path: str) -> None:

	variables = optimizer.variables()
	weights = pickle.load(open(path, 'rb'))

	for var, weight in zip(variables, weights):
		backend.set_value(var, weight)


def reset_accumulator(model: GradientAccumulateModel) -> None:

	model.accum_step_counter.assign(0)

	for i in range(len(model.gradient_accumulation)):

		model.gradient_accumulation[i].assign(
			tf.zeros_like(model.trainable_variables[i], dtype = model.dtype_value),
			read_value = False,
		)
