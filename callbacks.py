import os, pickle, typing
from keras import backend
from keras.callbacks import Callback

from utils import *
from settings import *


class LRScheduler(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)
		self.epoch = 0


	def on_batch_begin(self, batch: int, logs: dict[str, typing.Any] = {}) -> None:

		step = (self.epoch * STEP_PER_EPOCH + batch) / NUM_ACCUMULATIONS

		if step <= INCREASE_STEPS:
			lr = MAX_LEARNING_RATE * (step / INCREASE_STEPS)
		else:
			lr = max((MAX_LEARNING_RATE * (DECAY_STEPS - step)) / (DECAY_STEPS - INCREASE_STEPS), MIN_LEARNING_RATE)

		backend.set_value(self.model.optimizer.lr, backend.get_value(lr))


	def on_epoch_begin(self, epoch: int, logs: dict[str, typing.Any] = {}) -> None:

		self.epoch = epoch


	def on_batch_end(self, batch: int, logs: dict[str, typing.Any] = {}) -> None:

		logs['lr'] = backend.get_value(self.model.optimizer.lr)


class SaveModel(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)
		self.best_val_loss = float('inf')

		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)

		if os.path.exists(os.path.join(OUTPUT_DIR, 'logs.pkl')):
			logs = pickle.load(open(os.path.join(OUTPUT_DIR, 'logs.pkl'), 'rb'))
			self.best_val_loss = min(logs['val_loss'])


	def on_epoch_end(self, epoch: int, logs: dict[str, typing.Any] = {}) -> None:

		self.model.save_weights(os.path.join(OUTPUT_DIR, 'model.h5'))
		save_state(self.model.optimizer, os.path.join(OUTPUT_DIR, 'optimizer.pkl'))

		if logs['val_loss'] <= self.best_val_loss:
			self.best_val_loss = logs['val_loss']
			self.model.save_weights(os.path.join(OUTPUT_DIR, 'best_model.h5'))


class SaveLogs(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.logs = {
			'loss': [],
			'val_loss': [],
			'accuracy': [],
			'val_accuracy': [],
			'steps': [],
			'epochs': [],
			'tokens': []
		}

		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)

		if os.path.exists(os.path.join(OUTPUT_DIR, 'logs.pkl')):
			self.logs = pickle.load(open(os.path.join(OUTPUT_DIR, 'logs.pkl'), 'rb'))


	def on_epoch_end(self, epoch: int, logs: dict[str, typing.Any] = {}) -> None:

		self.logs['loss'].append(logs['loss'])
		self.logs['val_loss'].append(logs['val_loss'])
		self.logs['accuracy'].append(logs['accuracy'])
		self.logs['val_accuracy'].append(logs['val_accuracy'])
		self.logs['steps'].append(((epoch + 1) * STEP_PER_EPOCH) / NUM_ACCUMULATIONS)
		self.logs['epochs'].append(epoch + 1)
		self.logs['tokens'].append((epoch + 1) * STEP_PER_EPOCH * BATCH_SIZE * MAX_CONTEXT)

		pickle.dump(self.logs, open(os.path.join(OUTPUT_DIR, 'logs.pkl'), 'wb'))
