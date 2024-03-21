import torch, math
from torch import nn
from flash_attn import flash_attn_func

from dimgpt.training.layers import *
from dimgpt.settings import *


class AttentionBlock(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.query = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM)
		self.key = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM)
		self.value = Linear(EMBEDDING_DIM, NUM_HEADS * HEAD_DIM)
		self.projection = Linear(NUM_HEADS * HEAD_DIM, EMBEDDING_DIM)
		nn.init.normal_(self.projection.weight, mean = 0.0, std = INIT_STDDEV / math.sqrt(2 * NUM_BLOCKS))

		self.residual_dropout = nn.Dropout(DROPOUT)


	def forward(self, x: torch.Tensor):

		batch_size, context_size, _ = x.shape

		q = self.query(x)
		k = self.key(x)
		v = self.value(x)

		q = q.view(batch_size, context_size, NUM_HEADS, HEAD_DIM)
		k = k.view(batch_size, context_size, NUM_HEADS, HEAD_DIM)
		v = v.view(batch_size, context_size, NUM_HEADS, HEAD_DIM)

		x = flash_attn_func(q, k, v, dropout_p = DROPOUT if self.training else 0, causal = True)

		x = x.view(batch_size, context_size, NUM_HEADS * HEAD_DIM)

		return self.residual_dropout(self.projection(x))


# Model block
class TransformerBlock(Module):

	def __init__(self, **kwargs):

		super().__init__(**kwargs)

		self.norm_1 = LayerNorm(EMBEDDING_DIM)
		self.attention = AttentionBlock()
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
		self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(NUM_BLOCKS)])
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
