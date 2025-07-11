from .base import Sampler
from typing import Tuple, Optional
import torch

class IsotropicGaussian(Sampler):
    def __init__(self, sample_shape: Tuple[int], std: float = 1.0):
        super().__init__(sample_shape)
        self.register_buffer("std", torch.tensor(std))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        samples = self.std * torch.randn(num_samples, *self.sample_shape, device=self.std.device)
        return samples, None
