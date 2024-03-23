import torch

from dimgpt.settings import *


def create_rope_frequencies(dim: int, max_length: int, theta: float = ROPE_THETA) -> torch.Tensor:

	frequencies = 1.0 / (theta ** (torch.arange(0, dim, 2, device = DEVICE)[:(dim // 2)].float() / dim))
	t = torch.arange(max_length, device = DEVICE)
	frequencies = torch.outer(t, frequencies).float()

	return torch.polar(torch.ones_like(frequencies, device = DEVICE), frequencies)


def rotary_position_embedding(q: torch.Tensor, k: torch.Tensor, frequencies: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

	q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
	k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

	q_out = torch.view_as_real(q_complex * frequencies).flatten(3)
	k_out = torch.view_as_real(k_complex * frequencies).flatten(3)

	return q_out.type_as(q), k_out.type_as(k)