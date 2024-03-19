import os, pickle, math, time
import torch
from torch import nn

from dimgpt.settings import *
from dimgpt.training.datasets import Dataset
from dimgpt.training.model import Model
from dimgpt.training.optimizer import AdamW


class Trainer():

	def __init__(self, model: Model, dataset: Dataset):

		self.model = model
		model.train()

		self.dataset = dataset

		self.time = None
		self.step = 0
		self.tokens = 0
		self.epochs = 0.0
		self.learning_rate = 0.0
		self.loss = 0.0
		self.accuracy = 0.0
		self.val_loss = 0.0
		self.val_accuracy = 0.0
		self.loss_ema = None
		self.accuracy_ema = None
		self.best_val_loss = float('inf')

		self.optimizer = AdamW(self.model.parameters(), self.learning_rate)

		self.metrics_history = {
			'time': [],
			'step': [],
			'tokens': [],
			'epochs': [],
			'loss': [],
			'accuracy': [],
			'val_loss': [],
			'val_accuracy': []
		}


	# Save the models
	def save_model(self, path: str) -> None:

		if not os.path.exists(path):
			os.makedirs(path)

		torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
		torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))


	# Load the models
	def load_model(self, path) -> None:

		if not os.path.exists(path):
			return

		self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location = DEVICE))
		self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location = DEVICE))


	# Find previous session
	def find_previous_session(self) -> None:

		if os.path.exists(os.path.join(OUTPUT_DIR, 'last')):
			self.load_model(os.path.join(OUTPUT_DIR, 'last'))

		if os.path.exists(os.path.join(OUTPUT_DIR, 'metrics.pkl')):
			self.load_metrics()


	# Print
	def print(self) -> None:

		print(f'Epochs: {self.epochs:.4f} | Steps: {self.step:,} | Tokens: {self.tokens:,} | LR: {self.learning_rate:.5f}   ||   ' \
			f'Loss: {self.loss_ema:.5f} | Accuracy: {self.accuracy_ema * 100.0:.4f} % | ' \
			f'Val loss: {self.val_loss:.5f} | Val accuracy: {self.val_accuracy * 100.0:.4f} %       ', end = '\r')


	# Save metrics
	def save_metrics(self) -> None:

		if self.time is None:
			self.metrics_history["time"].append(0.0)
		else:
			self.metrics_history["time"].append(self.metrics_history["time"][-1] + (time.time() - self.time))

		self.time = time.time()

		self.metrics_history["step"].append(self.step)
		self.metrics_history["tokens"].append(self.tokens)
		self.metrics_history["epochs"].append(self.epochs)
		self.metrics_history["loss"].append(self.loss_ema)
		self.metrics_history["accuracy"].append(self.accuracy_ema)
		self.metrics_history["val_loss"].append(self.val_loss)
		self.metrics_history["val_accuracy"].append(self.val_accuracy)

		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)

		pickle.dump(self.metrics_history, open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'wb'))


	# Load metrics
	def load_metrics(self) -> None:

		self.metrics_history = pickle.load(open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'rb'))

		self.step = self.metrics_history["step"][-1]
		self.tokens = self.metrics_history["tokens"][-1]
		self.epochs = self.metrics_history["epochs"][-1]
		self.loss_ema = self.metrics_history["loss"][-1]
		self.accuracy_ema = self.metrics_history["accuracy"][-1]
		self.val_loss = self.metrics_history["val_loss"][-1]
		self.val_accuracy = self.metrics_history["val_accuracy"][-1]
		self.best_val_loss = min(self.metrics_history["val_loss"])
		self.time = time.time()


	# Update learning rate
	def update_learning_rate(self) -> None:

		if self.step < WARMUP_STEPS:
			ratio = self.step / WARMUP_STEPS
			self.learning_rate = MAX_LEARNING_RATE * ratio
		elif self.step < WARMUP_STEPS + DECAY_STEPS:
			ratio = (self.step - WARMUP_STEPS) / DECAY_STEPS
			ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
			self.learning_rate = ratio * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) + MIN_LEARNING_RATE
		else:
			self.learning_rate = MIN_LEARNING_RATE

		for g in self.optimizer.param_groups:
			g['lr'] = self.learning_rate


	def apply_ema(self, value_1: float, value_2: float) -> float:

		if value_1 is None:
			return value_2

		return value_1 * METRICS_BETA + value_2 * (1.0 - METRICS_BETA)


	# Train the model
	def train(self) -> None:

		# Training loop
		while True:

			# Update step
			self.step += 1
			self.tokens += (MAX_CONTEXT + 1) * BATCH_SIZE * NUM_ACCUMULATIONS
			self.epochs += ((MAX_CONTEXT + 1) * BATCH_SIZE * NUM_ACCUMULATIONS) / self.dataset.train_size()

			# Update learning rate
			self.update_learning_rate()

			# ----- Training ----- #

			self.model.train()
			self.loss = 0.0
			self.accuracy = 0.0

			# First load data (asyncronous)
			x, y, strength = self.dataset.next_train()

			for i in range(NUM_ACCUMULATIONS):

				with CONTEXT:

					# Forward pass
					prediction = self.model(x)

					# Loss
					loss = nn.functional.cross_entropy(
						input = prediction.reshape(-1, prediction.shape[-1]),
						target = y.reshape(-1),
						ignore_index = PADDING_TOKEN,
						reduction = 'none'
					)
					loss = ((loss * strength.reshape(-1)).sum() / (strength.sum() + 1e-8)) / NUM_ACCUMULATIONS
					self.loss += loss.item()

					# Accuracy
					accuracy = (prediction.argmax(dim = 2) == y).to(dtype = torch.float32)
					self.accuracy += (((accuracy * strength).sum() / (strength.sum() + 1e-8)) / NUM_ACCUMULATIONS).item()

				# Next load data (asyncronous)
				if i < NUM_ACCUMULATIONS - 1:
					x, y, strength = self.dataset.next_train()

				# Backward pass
				loss.backward()

			# Update weights
			self.model.clean_nan()
			self.model.clip_gradient(CLIP_GRADIENT)
			self.optimizer.step()
			self.optimizer.zero_grad(set_to_none = True)

			# Update ema values
			self.loss_ema = self.apply_ema(self.loss_ema, self.loss)
			self.accuracy_ema = self.apply_ema(self.accuracy_ema, self.accuracy)

			# ----- Validations ----- #

			if self.step % VAL_INTERVAL == 0:

				self.model.eval()

				with torch.no_grad():

					self.val_loss = 0.0
					self.val_accuracy = 0.0

					for _ in range(NUM_ACCUMULATIONS):

						# Load data
						x, y, strength = self.dataset.next_val()

						with CONTEXT:

							# Forward pass
							prediction = self.model(x)

							# Loss
							loss = nn.functional.cross_entropy(
								input = prediction.reshape(-1, prediction.shape[-1]),
								target = y.reshape(-1),
								ignore_index = PADDING_TOKEN,
								reduction = 'none'
							)
							self.val_loss += (((loss * strength.reshape(-1)).sum() / (strength.sum() + 1e-8)) / NUM_ACCUMULATIONS).item()

							# Accuracy
							accuracy = (prediction.argmax(dim = 2) == y).to(dtype = torch.float32)
							self.val_accuracy += (((accuracy * strength).sum() / (strength.sum() + 1e-8)) / NUM_ACCUMULATIONS).item()

				# Save
				self.save_metrics()
				self.save_model(os.path.join(OUTPUT_DIR, 'last'))

				# Save best
				if self.val_loss <= self.best_val_loss:
					self.best_val_loss = self.val_loss
					self.save_model(os.path.join(OUTPUT_DIR, 'best'))

			# -------------------- #

			# Print
			self.print()
