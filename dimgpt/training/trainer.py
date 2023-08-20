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

		self.optimizer = torch.optim.Adam(
			self.model.parameters(),
			lr = 0,
			betas = (BETA_1, BETA_2),
			weight_decay = WEIGHT_DECAY
		)

		self.step = 0
		self.tokens = 0
		self.epochs = 0.0

		self.metrics_ema = {
			'loss': 0.0,
			'accuracy': 0.0,
			'val_losses': [0.0] * len(self.val_datasets),
			'val_accuracies': [0.0] * len(self.val_datasets)
		}

		self.metrics_history = {
			'loss': [],
			'accuracy': [],
			'val_losses': [],
			'val_accuracies': []
		}


	# Save the models
	def save_model(self, path: str) -> None:

		if not os.path.exists(path):
			os.makedirs(path)

		torch.save(self.model.state_dict(), os.path.join(path, 'model.pt'))
		torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))
		pickle.dump(self.step, open(os.path.join(path, 'step.pkl'), 'wb'))


	# Load the models
	def load_model(self, path) -> None:

		if not os.path.exists(path):
			return

		self.model.load_state_dict(torch.load(os.path.join(path, 'model.pt'), map_location = DEVICE))
		self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pt'), map_location = DEVICE))
		self.step = pickle.load(open(os.path.join(path, 'step.pkl'), 'rb'))
		self.tokens = self.step * (MAX_CONTEXT + 1) * BATCH_SIZE * NUM_ACCUMULATIONS
		self.epochs = self.tokens / self.train_dataset.size()
		self.update_learning_rate()


	# Find previous session
	def find_previous_session(self) -> None:

		if os.path.exists(os.path.join(OUTPUT_DIR, 'model')):
			self.load_model(os.path.join(OUTPUT_DIR, 'model'))

		if os.path.exists(os.path.join(OUTPUT_DIR, 'loss.npy')):
			self.load_metrics()


	# Print
	def print(self) -> None:

		print(f'Epochs: {self.epochs:.4f} | Steps: {self.step:,} | Tokens: {self.tokens:,}   ||   ' \
			f'Loss: {self.metrics_ema["loss"]:.5f} | Accuracy: {self.metrics_ema["accuracy"] * 100.0:.4f} % | ' \
			f'Val loss: {self.metrics_ema["val_losses"][0]:.5f} | Val accuracy: {self.metrics_ema["val_accuracies"][0] * 100.0:.4f} %       ', end = '\r')


	# Save metrics
	def save_metrics(self) -> None:

		self.metrics_history["loss"].append(self.metrics_ema["loss"])
		self.metrics_history["accuracy"].append(self.metrics_ema["accuracy"])
		self.metrics_history["val_losses"].append(self.metrics_ema["val_losses"])
		self.metrics_history["val_accuracies"].append(self.metrics_ema["val_accuracies"])

		if not os.path.exists(OUTPUT_DIR):
			os.makedirs(OUTPUT_DIR)

		pickle.dump(self.metrics_history, open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'wb'))


	# Load metrics
	def load_metrics(self) -> None:

		self.metrics_history = pickle.load(open(os.path.join(OUTPUT_DIR, 'metrics.pkl'), 'rb'))

		self.metrics_ema["loss"] = self.metrics_history["loss"][-1]
		self.metrics_ema["accuracy"] = self.metrics_history["accuracy"][-1]
		self.metrics_ema["val_losses"] = self.metrics_history["val_losses"][-1]
		self.metrics_ema["val_accuracies"] = self.metrics_history["val_accuracies"][-1]


	# Update learning rate
	def update_learning_rate(self) -> None:

		if self.step < WARMUP_STEPS:
			ratio = self.step / WARMUP_STEPS
			lr = MAX_LEARNING_RATE * ratio
		elif self.step < WARMUP_STEPS + DECAY_STEPS:
			ratio = (self.step - WARMUP_STEPS) / DECAY_STEPS
			ratio = 0.5 * (1.0 + math.cos(math.pi * ratio))
			lr = ratio * (MAX_LEARNING_RATE - MIN_LEARNING_RATE) + MIN_LEARNING_RATE
		else:
			lr = MIN_LEARNING_RATE

		for g in self.optimizer.param_groups:
			g['lr'] = lr


	# Train the model
	def train(self) -> None:

		# Training loop
		while True:

			# ----- Training ----- #

			self.model.train()

			loss = 0.0
			accuracy = 0.0

			# First load data (asyncronous)
			x, y = self.train_dataset.next()

			for _ in range(NUM_ACCUMULATIONS):

				# Forward pass
				prediction = self.model(x)

				# Loss
				loss_tensor = nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), y.reshape(-1)) / NUM_ACCUMULATIONS

				# Next load data (asyncronous)
				x, y = self.train_dataset.next()

				# Backward pass
				loss_tensor.backward()

				# Metrics
				loss += loss_tensor.item()
				accuracy += ((prediction.argmax(dim = 2) == y).to(dtype = torch.float32).mean() / NUM_ACCUMULATIONS).item()

			# Update weights
			self.model.clean_nan()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_GRADIENT)
			self.optimizer.step()
			self.optimizer.zero_grad(set_to_none = True)

			# Update print values
			if self.step == 0:
				self.metrics_ema['loss'] = loss
				self.metrics_ema['accuracy'] = accuracy
			else:
				self.metrics_ema['loss'] = self.metrics_ema['loss'] * PRINT_MA_BETA + loss * (1.0 - PRINT_MA_BETA)
				self.metrics_ema['accuracy'] = self.metrics_ema['accuracy'] * PRINT_MA_BETA + accuracy * (1.0 - PRINT_MA_BETA)

			# ----- Validations ----- #

			if self.step % VAL_INTERVAL == 0:

				self.model.eval()

				with torch.no_grad():

					for i, val_dataset in enumerate(self.val_datasets):

						val_loss = 0.0
						val_accuracy = 0.0

						# First load data (asyncronous)
						x, y = val_dataset.next()

						for _ in range(NUM_ACCUMULATIONS):

							# Forward pass
							prediction = self.model(x)

							# Next load data (asyncronous)
							x, y = val_dataset.next()

							# Loss and accuracy
							val_loss += nn.functional.cross_entropy(prediction.reshape(-1, prediction.shape[-1]), y.reshape(-1)).item() / NUM_ACCUMULATIONS
							val_accuracy += (prediction.argmax(dim = 2) == y).to(dtype = torch.float32).mean().item() / NUM_ACCUMULATIONS

						# Update print values
						if self.step == 0:
							self.metrics_ema['val_losses'][i] = val_loss
							self.metrics_ema['val_accuracies'][i] = val_accuracy
						else:
							self.metrics_ema['val_losses'][i] = self.metrics_ema['val_losses'][i] * PRINT_MA_BETA + val_loss * (1.0 - PRINT_MA_BETA)
							self.metrics_ema['val_accuracies'][i] = self.metrics_ema['val_accuracies'][i] * PRINT_MA_BETA + val_accuracy * (1.0 - PRINT_MA_BETA)

				# Save
				self.save_metrics()

			# -------------------- #

			# Print
			self.print()

			# Save
			if self.step % SAVE_INTERVAL == 0:
				self.save_model(os.path.join(OUTPUT_DIR, 'last'))

			# Save best
			if len(self.metrics_history['val_losses'][:-1]) == 0 or self.metrics_ema['val_losses'][0] <= min([l[0] for l in self.metrics_history['val_losses'][:-1]]):
				self.save_model(os.path.join(OUTPUT_DIR, 'best'))

			# Update step
			self.step += 1
			self.tokens = self.step * (MAX_CONTEXT + 1) * BATCH_SIZE * NUM_ACCUMULATIONS
			self.epochs = self.tokens / self.train_dataset.size()

			# Update learning rate
			self.update_learning_rate()
