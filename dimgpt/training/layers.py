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

'''
class CausalSelfAttention(Module):

	def __init__(self):

		super().__init__()

		self.attention = nn.MultiheadAttention(
			embed_dim = EMBEDDING_DIM,
			num_heads = NUM_HEADS,
			dropout = DROPOUT if self.training else 0.0
		)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device = DEVICE)
		return self.attention(x, x, x, need_weights = False, attn_mask = mask, is_causal = True)[0]
'''


class Linear(nn.Linear):

	def __init__(self, in_features: int, out_features: int, bias: bool = USE_BIAS):

		super().__init__(in_features, out_features, bias)
		nn.init.normal_(self.weight, mean = 0.0, std = INIT_STDDEV)


class LayerNorm(Module):

	def __init__(self, shape: int, bias: bool = USE_BIAS, epsilon: float = 1e-5):

		super().__init__()

		self.shape = (shape,)
		self.weight = nn.Parameter(torch.ones(shape))
		self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
		self.epsilon = epsilon


	def forward(self, x: torch.Tensor):

		return nn.functional.layer_norm(x, self.shape, self.weight, self.bias, self.epsilon)


class Embedding(nn.Embedding):

	def __init__(self, num_embeddings: int, embedding_dim: int):

		super().__init__(num_embeddings, embedding_dim)
		nn.init.normal_(self.weight, mean = 0.0, std = INIT_STDDEV)


class CausalSelfAttention(Module):

	def __init__(self):

		super().__init__()

		self.c_attn = Linear(EMBEDDING_DIM, 3 * EMBEDDING_DIM)
		self.c_proj = Linear(EMBEDDING_DIM, EMBEDDING_DIM)
		nn.init.normal_(self.c_proj.weight, mean = 0.0, std = INIT_STDDEV / math.sqrt(2 * NUM_BLOCKS))
		self.resid_dropout = nn.Dropout(DROPOUT)


	def forward(self, x):

		B, T, C = x.shape

		q, k, v  = self.c_attn(x).split(EMBEDDING_DIM, dim = 2)

		k = k.reshape(B, T, NUM_HEADS, C // NUM_HEADS).transpose(1, 2)
		q = q.reshape(B, T, NUM_HEADS, C // NUM_HEADS).transpose(1, 2)
		v = v.reshape(B, T, NUM_HEADS, C // NUM_HEADS).transpose(1, 2)

		y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = DROPOUT if self.training else 0, is_causal = True)
		y = y.transpose(1, 2).contiguous().reshape(B, T, C)
		y = self.resid_dropout(self.c_proj(y))

		return y
