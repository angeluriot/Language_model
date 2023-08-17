import torch
from torch import nn

from dimgpt.training.layers import *
from dimgpt.settings import *
from dimgpt import utils


# Model block
class Block(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.norm_1 = nn.LayerNorm(EMBEDDING_DIM)

		self.attention = nn.MultiheadAttention(
			embed_dim = EMBEDDING_DIM,
			num_heads = NUM_HEADS,
			dropout = DROPOUT if self.training else 0.0
		)

		self.norm_2 = nn.LayerNorm(EMBEDDING_DIM)

		self.mlp = nn.Sequential(
			nn.Linear(EMBEDDING_DIM, FFN_DIM),
			nn.GELU(),
			nn.Linear(FFN_DIM, EMBEDDING_DIM),
			nn.Dropout(DROPOUT)
		)


	def forward(self, input: torch.Tensor) -> torch.Tensor:

		x = self.norm_1(input)
		x = input + self.attention(x, x, x, need_weights = False, is_causal = True)
		x = self.norm_2(x)
		x = x + self.mlp(x)

		return x


# Model
class Model(Module):

	def __init__(self, vocab_size: int, **kwargs):

		super().__init__(**kwargs)

		self.word_embedding = nn.Embedding(
			num_embeddings = vocab_size,
			embedding_dim = EMBEDDING_DIM
		)

		self.position_embedding = nn.Embedding(
			num_embeddings = MAX_CONTEXT,
			embedding_dim = EMBEDDING_DIM
		)

		self.init_dropout = nn.Dropout(DROPOUT)
		self.blocks = nn.Sequential(*[Block() for _ in range(NUM_BLOCKS)])
		self.final_norm = nn.LayerNorm(EMBEDDING_DIM)
		self.final_linear = nn.Linear(EMBEDDING_DIM, vocab_size, bias = False)
		self.word_embedding.weight = self.final_linear.weight


	def forward(self, input: torch.Tensor) -> torch.Tensor:

		if input.shape[1] > MAX_CONTEXT:
			input = input[:, -MAX_CONTEXT:]

		word_embeddings = self.word_embedding(input)

		positions = torch.arange(input.shape[1], dtype = torch.long, device = DEVICE)
		position_embeddings = self.position_embedding(positions)

		x = word_embeddings + position_embeddings
		x = self.init_dropout(x)

		x = self.blocks(x)

		x = self.final_norm(x)
		x = self.final_linear(x)

		return x
