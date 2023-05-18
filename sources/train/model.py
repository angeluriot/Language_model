import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.initializers.initializers_v2 import RandomNormal
from gradient_accumulator import GradientAccumulateModel

from utils import *
from train.layers import *
from settings import *
from data.data import *
from utils import *
from data.tokenizer import Tokenizer


def create_block(inputs, i: int):

	model = LayerNormalization(
		epsilon = 1e-5,
		center = USE_BIAS,
		name = f'block_{i}_layer_norm_1'
	)(inputs)

	model = MultiHeadAttention(
		num_heads = NUM_HEADS,
		key_dim = EMBEDDING_DIM // NUM_HEADS,
		dropout = DROPOUT,
		use_bias = USE_BIAS,
		kernel_initializer = RandomNormal(mean = 0.0, stddev = INIT_STDDEV),
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
		use_bias = USE_BIAS,
		kernel_initializer = RandomNormal(mean = 0.0, stddev = INIT_STDDEV),
		name = f'block_{i}_FFN_dense_1'
	)(skip)

	model = GeLU(name = f'block_{i}_FFN_gelu')(model)

	model = Dense(
		units = EMBEDDING_DIM,
		use_bias = USE_BIAS,
		kernel_initializer = RandomNormal(mean = 0.0, stddev = INIT_STDDEV),
		name = f'block_{i}_FFN_dense_2'
	)(model)

	model = Dropout(DROPOUT, name = f'block_{i}_FFN_dropout')(model)
	model = Add(name = f'block_{i}_skip_2')([model, skip])

	return model


def create_model(vocab_size: int = VOCAB_SIZE) -> Model | GradientAccumulateModel:

	input = Input(shape = (None,), dtype = tf.uint16, name = 'input')

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

	if NUM_ACCUMULATIONS > 1:
		model = GradientAccumulateModel(accum_steps = NUM_ACCUMULATIONS, inputs = model.input, outputs = model.output)

	for i in range(len(model.weights)):
		model.weights[i]._handle_name = model.weights[i].name + "_" + str(i)

	return model


def predict(model: Model | GradientAccumulateModel, input: str, tokenizer: Tokenizer, max_length: int, keep_input = False,
	temperature: float = 1.0, top_p: float = 1.0, no_repeat: float = 0.0, verbose: bool = False, max_print_line_length = 0) -> str:

	input = list(tokenizer.encode(input))
	output = []
	last_line_length = 0

	if keep_input:
		output = input.copy()
		text = tokenizer.decode(input)
		last_line_length = len(text) - 1 - text.rfind('\n')

	for _ in range(max_length):

		probabilities = model.predict(np.array([input], dtype = np.uint16), verbose = 0)[0, -1]
		probabilities = np.log(probabilities)
		proximity = MAX_CONTEXT

		for i in reversed(range(max(len(input) - MAX_CONTEXT, 0), len(input))):
			strength = no_repeat * (proximity / MAX_CONTEXT)
			probabilities[input[i]] *= (1 + strength)
			proximity -= 1

		if temperature < 0.01:
			index = np.argmax(probabilities)

		else:
			probabilities /= temperature
			probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities))

			sorted_indices = np.argsort(-probabilities)
			cumsum_probabilites = np.cumsum(probabilities[sorted_indices])
			cutoff_index = np.searchsorted(cumsum_probabilites, max(top_p, cumsum_probabilites[0] + 1e-6))
			temp = np.zeros_like(probabilities)
			temp[sorted_indices[:cutoff_index]] = probabilities[sorted_indices[:cutoff_index]]
			probabilities = temp / np.sum(temp)

			index = np.random.choice(range(len(probabilities)), p = probabilities)

		input.append(index)
		output.append(index)

		if verbose:

			text = tokenizer.decode(input)

			if '\n' in text:
				last_line_length = len(text) - 1 - text.rfind('\n')
			else:
				last_line_length += len(text)

			if max_print_line_length > 0 and last_line_length >= max_print_line_length and text.startswith(' '):
				print()
				text = text[1:]
				last_line_length = 0

			print(text, end = '')

	return tokenizer.decode(output)
