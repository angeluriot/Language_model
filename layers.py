import math, typing
import tensorflow as tf
from keras.layers import Layer
from keras.initializers.initializers_v2 import RandomNormal

from settings import *


class TokenEmbedding(Layer):

	def __init__(self, vocab_size: int, embedding_dim: int, max_context: int, padding_token: int | None = None, **kwargs):

		super().__init__(**kwargs)
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.max_context = max_context
		self.padding_token = padding_token


	def get_config(self) -> dict[str, typing.Any]:

		config = super().get_config().copy()

		config.update({
			'vocab_size': self.vocab_size,
			'embedding_dim': self.embedding_dim,
			'max_context': self.max_context,
			'padding_token': self.padding_token
		})

		return config


	def build(self, input_shape) -> None:

		super().build(input_shape)

		self.embedding_matrix = self.add_weight(
			shape = (self.vocab_size, self.embedding_dim),
			initializer = RandomNormal(mean = 0.0, stddev = INIT_STDDEV),
			trainable = True,
			dtype = tf.float32,
			name = self.name + '_(embedding_matrix)'
		)


	def call(self, inputs, transpose: bool = False):

		if not transpose:

			if tf.shape(inputs)[1] > self.max_context:
				inputs = inputs[:, -self.max_context:]

			embedding = tf.nn.embedding_lookup(self.embedding_matrix, tf.cast(inputs, tf.int32))

			if self.padding_token is not None:
				mask = tf.cast(tf.not_equal(inputs, self.padding_token), tf.float32)
				mask = tf.expand_dims(mask, axis = -1)
				mask = tf.tile(mask, multiples = (1, 1, self.embedding_dim))
				return embedding * mask

			return embedding

		else:

			return tf.matmul(inputs, tf.transpose(self.embedding_matrix))


class PositionEmbedding(Layer):

	def __init__(self, max_context: int, embedding_dim: int, padding_token: int | None = None, **kwargs):

		super().__init__(**kwargs)
		self.max_context = max_context
		self.embedding_dim = embedding_dim
		self.padding_token = padding_token


	def get_config(self) -> dict[str, typing.Any]:

		config = super().get_config().copy()

		config.update({
			'max_context': self.max_context,
			'embedding_dim': self.embedding_dim,
			'padding_token': self.padding_token
		})

		return config


	def build(self, input_shape) -> None:

		super().build(input_shape)

		self.embedding_matrix = self.add_weight(
			shape = (self.max_context, self.embedding_dim),
			initializer = RandomNormal(mean = 0.0, stddev = INIT_STDDEV),
			trainable = True,
			dtype = tf.float32,
			name = self.name + '_(embedding_matrix)'
		)


	def call(self, inputs):

		if tf.shape(inputs)[1] > self.max_context:
			inputs = inputs[:, -self.max_context:]

		index = tf.range(tf.shape(inputs)[1], dtype = tf.int32)
		index = tf.expand_dims(index, axis = 0)
		index = tf.tile(index, multiples = (tf.shape(inputs)[0], 1))

		embedding = tf.nn.embedding_lookup(self.embedding_matrix, index)

		if self.padding_token is not None:
			mask = tf.cast(tf.not_equal(inputs, self.padding_token), tf.float32)
			mask = tf.expand_dims(mask, axis = -1)
			mask = tf.tile(mask, multiples = (1, 1, self.embedding_dim))
			return embedding * mask

		return embedding


class GeLU(Layer):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)


	def call(self, inputs):

		return 0.5 * inputs * (1.0 + tf.math.tanh(tf.math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * tf.math.pow(inputs, 3))))
