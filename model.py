import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.initializers import RandomNormal

from utils import *
from layers import *
from settings import *
from data import *
from utils import *


def create_block(inputs, i):

	model = LayerNormalization(
		epsilon = 1e-5,
		center = USE_BIAS,
		name = f'block_{i}_layer_norm_1'
	)(inputs)

	model = MultiHeadAttention(
		num_heads = NUM_HEADS,
		key_dim = EMBEDDING_DIM // NUM_HEADS,
		dropout = DROPOUT,
		kernel_initializer = RandomNormal(mean = 0., stddev = INIT_STDDEV),
		use_bias = USE_BIAS,
		name = f'block_{i}_causal_attention'
	)(model, model, model, use_causal_mask = True)

	model = Add(name = f'block_{i}_skip_1')([model, inputs])

	skip = LayerNormalization(
		epsilon = 1e-5,
		center = USE_BIAS,
		name = f'block_{i}_layer_norm_2'
	)(model)

	model = Dense(
		units = FFN_DIM,
		kernel_initializer = RandomNormal(mean = 0., stddev = INIT_STDDEV),
		use_bias = USE_BIAS,
		name = f'block_{i}_FFN_dense_1'
	)(skip)

	model = GeLU(name = f'block_{i}_FFN_gelu')(model)

	model = Dense(
		units = EMBEDDING_DIM,
		kernel_initializer = RandomNormal(mean = 0., stddev = INIT_STDDEV),
		use_bias = USE_BIAS,
		name = f'block_{i}_FFN_dense_2'
	)(model)

	model = Dropout(DROPOUT, name = f'block_{i}_FFN_dropout')(model)
	model = Add(name = f'block_{i}_skip_2')([model, skip])

	return model


def create_model(vocab_size = VOCAB_SIZE):

	input = Input(shape = (None,), dtype = tf.int32, name = 'input')

	embedding_layer = TokenEmbedding(vocab_size, EMBEDDING_DIM, MAX_CONTEXT, name = 'token_embedding')

	token_embedding = embedding_layer(input)
	position_embedding = PositionEmbedding(MAX_CONTEXT, EMBEDDING_DIM, name = 'position_embedding')(input)
	model = Add(name = 'embedding_add')([token_embedding, position_embedding])
	model = Dropout(DROPOUT, name = 'embedding_dropout')(model)

	for i in range(NUM_BLOCKS):
		model = create_block(model, i)

	model = LayerNormalization(
		epsilon = 1e-5,
		center = USE_BIAS,
		name = 'final_layer_norm'
	)(model)

	model = embedding_layer(model, transpose = True)
	model = Activation('softmax', name = 'final_softmax')(model)

	model = Model(inputs = input, outputs = model)

	return model


def predict(model, input, tokenizer, max_length, temperature = 1.0, top_p = 1.0, verbose = False):

	input = tokenizer.encode(input)
	output = []

	for _ in range(max_length):

		probabilities = model.predict(np.array([input]), verbose = 0)[0, -1]

		if temperature < 0.01:
			index = np.argmax(probabilities)

		else:
			probabilities = np.log(probabilities) / temperature
			probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

			#sorted_indices = np.argsort(probabilities)[::-1]
			#cumulative_probabilities = np.cumsum(probabilities[sorted_indices])
			#sorted_indices = sorted_indices[cumulative_probabilities <= top_p]
			#probabilities = probabilities[sorted_indices]
			#probabilities = probabilities / np.sum(probabilities)

			index = np.random.choice(range(len(probabilities)), p = probabilities)

		input = np.append(input, index)
		output.append(index)

		if verbose:
			print(tokenizer.decode(index), end = '')

	return tokenizer.decode(output)
