from keras.callbacks import *
from keras import backend
from settings import *

class LRScheduler(Callback):

	def __init__(self, train_size, **kwargs):
		super().__init__(**kwargs)
		self.train_size = train_size
		self.epoch = 0


	def on_batch_begin(self, batch, logs = None):

		step = self.epoch * (self.train_size // BATCH_SIZE) + batch

		if step <= INCREASE_STEPS:
			lr = MAX_LEARNING_RATE * (step / INCREASE_STEPS)

		else:
			float_epoch = self.epoch + batch / (self.train_size // BATCH_SIZE)
			lr = max(MAX_LEARNING_RATE * (MIN_LEARNING_RATE / MAX_LEARNING_RATE) ** (float_epoch / DECAY_EPOCHS), MIN_LEARNING_RATE)

		backend.set_value(self.model.optimizer.lr, backend.get_value(lr))


	def on_epoch_begin(self, epoch, logs = None):

		self.epoch = epoch


	def on_batch_end(self, batch, logs = None):

		logs["lr"] = backend.get_value(self.model.optimizer.lr)
