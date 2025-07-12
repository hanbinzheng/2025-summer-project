from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch
import torch.nn as nn


class Sampler(nn.Module, ABC):
    # interface to sample from p_init

    def __init__(self, sample_shape: Tuple[int]):
        # shape: the shape of a single tensor, (c, h, w)
        super().__init__()
        self.sample_shape = sample_shape
        self.register_buffer("dummy", torch.tensor(0,))

    def shape(self) -> Tuple[int]:
        return self.sample_shape

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # return num_samples from p_init, (num_samples, shape)
        pass
