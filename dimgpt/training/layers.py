import math
import numpy as np
import torch
from torch import nn

from dimgpt.settings import *


# Base class for all layers
class Module(nn.Module):

	# Give the number of parameters of the module
	def nb_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters()])


	# Give the number of trainable parameters of the module
	def nb_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if p.requires_grad])


	# Give the number of non-trainable parameters of the module
	def nb_non_trainable_parameters(self) -> int:

		return sum([np.prod(p.size(), dtype = np.int32) for p in self.parameters() if not p.requires_grad])


	# Summarize the module
	def summary(self) -> None:

		print(f'Number of parameters: {self.nb_parameters():,}')
		print(f'Number of trainable parameters: {self.nb_trainable_parameters():,}')
		print(f'Number of non-trainable parameters: {self.nb_non_trainable_parameters():,}')


	# Remove NaNs from the module gradients
	def clean_nan(self) -> None:

		for p in self.parameters():
			if p.grad is not None:
				torch.nan_to_num(p.grad, nan = 0, posinf = 1e5, neginf = -1e5, out = p.grad)


	# Clip the module gradients
	def clip_gradient(self, max_norm: float) -> None:

		nn.utils.clip_grad_norm_(self.parameters(), max_norm)


class Linear(nn.Linear):

	def __init__(self, in_features: int, out_features: int, bias: bool = USE_BIAS, **kwargs):

		super().__init__(in_features, out_features, bias, **kwargs)
		nn.init.normal_(self.weight, mean = 0.0, std = INIT_STDDEV)


class LayerNorm(Module):

	def __init__(self, shape: int, bias: bool = USE_BIAS, epsilon: float = 1e-5, **kwargs):

		super().__init__(**kwargs)

		self.shape = (shape,)
		self.weight = nn.Parameter(torch.ones(shape))
		self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
		self.epsilon = epsilon


	def forward(self, x: torch.Tensor):

		return nn.functional.layer_norm(x, self.shape, self.weight, self.bias, self.epsilon)


class Embedding(nn.Embedding):

	def __init__(self, num_embeddings: int, embedding_dim: int, **kwargs):

		super().__init__(num_embeddings, embedding_dim, **kwargs)
		nn.init.normal_(self.weight, mean = 0.0, std = INIT_STDDEV)


class CausalSelfAttention(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.causal_attention = Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM)
		self.causal_projection = Linear(EMBEDDING_DIM, EMBEDDING_DIM)
		nn.init.normal_(self.causal_projection.weight, mean = 0.0, std = INIT_STDDEV / math.sqrt(2 * NUM_BLOCKS))
		self.residual_dropout = nn.Dropout(DROPOUT)


	def forward(self, x):

		batch_size, context_size, _ = x.shape

		q, k, v  = self.causal_attention(x).split(EMBEDDING_DIM, dim = 2)

		k = k.reshape(batch_size, context_size, NUM_HEADS, EMBEDDING_DIM // NUM_HEADS).transpose(1, 2)
		q = q.reshape(batch_size, context_size, NUM_HEADS, EMBEDDING_DIM // NUM_HEADS).transpose(1, 2)
		v = v.reshape(batch_size, context_size, NUM_HEADS, EMBEDDING_DIM // NUM_HEADS).transpose(1, 2)

		x = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = DROPOUT if self.training else 0, is_causal = True)
		x = x.transpose(1, 2).contiguous().reshape(batch_size, context_size, EMBEDDING_DIM)
		x = self.residual_dropout(self.causal_projection(x))

		return x
