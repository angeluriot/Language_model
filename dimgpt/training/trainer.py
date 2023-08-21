import os, pickle, math
import torch
from torch import nn

from dimgpt.settings import *
from dimgpt.training.data import *
from dimgpt.training.model import *


class Trainer():

	def __init__(self, model: Model, train_dataset: Dataset, val_datasets: list[Dataset]):

		self.model = model
		self.train_dataset = train_dataset
		self.val_datasets = val_datasets

		self.step = 0
		self.tokens = 0
		self.epochs = 0.0
		self.learning_rate = 0.0
		self.loss = 0.0
		self.accuracy = 0.0
		self.loss_ema = None
		self.accuracy_ema = None
		self.val_losses = [0.0] * len(self.val_datasets)
		self.val_accuracies = [0.0] * len(self.val_datasets)
		self.best_val_loss = float('inf')

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr = self.learning_rate,
			betas = (BETA_1, BETA_2),
			weight_decay = WEIGHT_DECAY
		)

		self.metrics_history = {
			'step': [],
			'tokens': [],
			'loss': [],
			'accuracy': [],
			'val_losses': [[]] * len(self.val_datasets),
			'val_accuracies': [[]] * len(self.val_datasets)
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

		if os.path.exists(os.path.join(OUTPUT_DIR, 'model')):
			self.load_model(os.path.join(OUTPUT_DIR, 'model'))

		if os.path.exists(os.path.join(OUTPUT_DIR, 'metrics.pkl')):
			self.load_metrics()


	# Print
	def print(self) -> None:

		print(f'Epochs: {self.epochs:.4f} | Steps: {self.step:,} | Tokens: {self.tokens:,} | LR: {self.learning_rate:.5f}   ||   ' \
			f'Loss: {self.loss_ema:.5f} | Accuracy: {self.accuracy_ema * 100.0:.4f} % | ' \
			f'Val loss: {self.val_losses[0]:.5f} | Val accuracy: {self.val_accuracies[0] * 100.0:.4f} %       ', end = '\r')


	# Save metrics
	def save_metrics(self) -> None:

		self.metrics_history["step"].append(self.step)
		self.metrics_history["tokens"].append(self.tokens)
		self.metrics_history["loss"].append(self.loss_ema)
		self.metrics_history["accuracy"].append(self.accuracy_ema)

		for i in range(len(self.val_datasets)):
			self.metrics_history["val_losses"][i].append(self.val_losses[i])
			self.metrics_history["val_accuracies"][i].append(self.val_accuracies[i])

		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)

		pickle.dump(self.metrics_history, open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'wb'))


	# Load metrics
	def load_metrics(self) -> None:

		self.metrics_history = pickle.load(open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'rb'))

		self.step = self.metrics_history["step"][-1]
		self.tokens = self.metrics_history["tokens"][-1]
		self.epochs = self.tokens / self.train_dataset.size()
		self.loss_ema = self.metrics_history["loss"][-1]
		self.accuracy_ema = self.metrics_history["accuracy"][-1]

		for i in range(len(self.val_datasets)):
			self.val_losses[i] = self.metrics_history["val_losses"][i][-1]
			self.val_accuracies[i] = self.metrics_history["val_accuracies"][i][-1]

		self.best_val_loss = min(self.metrics_history["val_losses"][0])
		self.update_learning_rate()


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

			# ----- Training ----- #

			self.model.train()
			self.loss = 0.0
			self.accuracy = 0.0

			# First load data (asyncronous)
			x, y = self.train_dataset.next()

			for _ in range(NUM_ACCUMULATIONS):

				# Forward pass
				prediction = self.model(x)

				# Loss
				loss = nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), y.reshape(-1)) / NUM_ACCUMULATIONS
				self.loss += loss.item()

				# Accuracy
				self.accuracy += ((prediction.argmax(dim = 2) == y).to(dtype = torch.float32).mean() / NUM_ACCUMULATIONS).item()

				# Next load data (asyncronous)
				x, y = self.train_dataset.next()

				# Backward pass
				loss.backward()

			# Update weights
			self.model.clean_nan()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRADIENT)
			self.optimizer.step()
			self.optimizer.zero_grad(set_to_none = True)

			# Update ema values
			self.loss_ema = self.apply_ema(self.loss_ema, self.loss)
			self.accuracy_ema = self.apply_ema(self.accuracy_ema, self.accuracy)

			# ----- Validations ----- #

			if self.step % VAL_INTERVAL == 0:

				self.model.eval()

				with torch.no_grad():

					for i, val_dataset in enumerate(self.val_datasets):

						self.val_losses[i] = 0.0
						self.val_accuracies[i] = 0.0

						for _ in range(NUM_ACCUMULATIONS):

							# Load data
							x, y = val_dataset.next()

							# Forward pass
							prediction = self.model(x)

							# Loss and accuracy
							self.val_losses[i] += (nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), y.reshape(-1)) / NUM_ACCUMULATIONS).item()
							self.val_accuracies[i] += ((prediction.argmax(dim = 2) == y).to(dtype = torch.float32).mean() / NUM_ACCUMULATIONS).item()

				# Save
				self.save_metrics()
				self.save_model(os.path.join(OUTPUT_DIR, 'last'))

				# Save best
				if self.val_losses[0] <= self.best_val_loss:
					self.best_val_loss = self.val_losses[0]
					self.save_model(os.path.join(OUTPUT_DIR, 'best'))

			# -------------------- #

			# Print
			self.print()

			# Update step
			self.step += 1
			self.tokens = self.step * (MAX_CONTEXT + 1) * BATCH_SIZE * NUM_ACCUMULATIONS
			self.epochs = self.tokens / self.train_dataset.size()

			# Update learning rate
			self.update_learning_rate()
