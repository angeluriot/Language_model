import torch
from torch import nn

from dimgpt.training.layers import *
from dimgpt.settings import *


# Model block
class Block(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.norm_1 = LayerNorm(EMBEDDING_DIM)
		self.attention = CausalSelfAttention()
		self.norm_2 = LayerNorm(EMBEDDING_DIM)

		self.mlp = nn.Sequential(
			Linear(EMBEDDING_DIM, FFN_DIM),
			nn.GELU(),
			Linear(FFN_DIM, EMBEDDING_DIM),
			nn.Dropout(DROPOUT)
		)


	def forward(self, x: torch.Tensor) -> torch.Tensor:

		x = x + self.attention(self.norm_1(x))
		x = x + self.mlp(self.norm_2(x))

		return x


# Model
class Model(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.token_embedding = Embedding(VOCAB_SIZE, EMBEDDING_DIM)
		self.position_embedding = Embedding(MAX_CONTEXT, EMBEDDING_DIM)
		self.init_dropout = nn.Dropout(DROPOUT)
		self.blocks = nn.Sequential(*[Block() for _ in range(NUM_BLOCKS)])
		self.final_norm = LayerNorm(EMBEDDING_DIM)
		self.final_linear = Linear(EMBEDDING_DIM, VOCAB_SIZE, bias = False)
		self.token_embedding.weight = self.final_linear.weight


	def forward(self, input: torch.Tensor, only_last: bool = False) -> torch.Tensor:

		if input.shape[1] > MAX_CONTEXT:
			input = input[:, -MAX_CONTEXT:]

		token_embeddings = self.token_embedding(input)

		positions = torch.arange(input.shape[1], dtype = torch.long, device = DEVICE)
		position_embeddings = self.position_embedding(positions).repeat(input.shape[0], 1, 1)

		x = token_embeddings + position_embeddings
		x = self.init_dropout(x)

		x = self.blocks(x)

		x = self.final_norm(x)

		if only_last:
			return self.final_linear(x[:, -1])

		return self.final_linear(x)
