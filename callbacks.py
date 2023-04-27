from keras.callbacks import *
from keras import backend
from settings import *

class LRScheduler(Callback):

	def __init__(self, train_size, **kwargs):
		super().__init__(**kwargs)
		self.train_size = train_size
		self.epoch = 0


	def on_batch_begin(self, batch, logs = None):

		step = self.epoch * self.train_size + batch

		if step <= INCREASE_STEPS:
			lr = MAX_LEARNING_RATE * (step / INCREASE_STEPS)

		else:
			float_epoch = self.epoch + batch / self.train_size
			lr = max(MAX_LEARNING_RATE * (MIN_LEARNING_RATE / MAX_LEARNING_RATE) ** (float_epoch / DECAY_EPOCHS), MIN_LEARNING_RATE)

		backend.set_value(self.model.optimizer.lr, backend.get_value(lr))


	def on_epoch_begin(self, epoch, logs = None):

		self.epoch = epoch


	def on_batch_end(self, batch, logs = None):

		logs['lr'] = backend.get_value(self.model.optimizer.lr)


class LossSaver(Callback):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)
		self.losses = []
		self.val_losses = []


	def on_epoch_end(self, epoch, logs = {}):

		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))

		np.save('losses.npy', self.losses)
		np.save('val_losses.npy', self.val_losses)

