import torch
from torch import nn

from dimgpt.settings import *


class AdamW(torch.optim.AdamW):

	def __init__(self, params: list[nn.Parameter], learning_rate: float, **kwargs):

		decay_params = [p for p in params if p.requires_grad and p.dim() >= 2]
		other_params = [p for p in params if p.requires_grad and p.dim() < 2]

		groups = [
			{'params': decay_params, 'weight_decay': WEIGHT_DECAY},
			{'params': other_params, 'weight_decay': 0.0}
		]

		super().__init__(
			groups,
			lr = learning_rate,
			betas = (BETA_1, BETA_2),
			eps = EPSILON,
			fused = GPU_ENABLED,
			**kwargs
		)
