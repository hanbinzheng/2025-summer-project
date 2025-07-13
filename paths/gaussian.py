from abc import ABC, abstractmethod
from .base import ConditionalPath
import torch
from torch.func import jacrev, vmap
from typing import Tuple
from samplers import IsotropicGaussian

class Alpha(ABC):
    def __init__(self):
        # check alpha_(t=0) = 0
        assert torch.allclose(
            self(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1)
        )
        # check alpha_(t=1) = 1
        assert torch.allclose(
            self(torch.ones(1, 1, 1, 1)), torch.ones(1, 1, 1, 1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 0.0, self(1.0) = 1.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - alpha_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt alpha_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt alpha_t (num_samples, 1, 1, 1)
        """
        device = t.device
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1).to(device)


class Beta(ABC):
    def __init__(self):
        # Check beta_0 = 1
        assert torch.allclose(
            self(torch.zeros(1,1,1,1)), torch.ones(1,1,1,1)
        )
        # Check beta_1 = 0
        assert torch.allclose(
            self(torch.ones(1,1,1,1)), torch.zeros(1,1,1,1)
        )

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates alpha_t. Should satisfy: self(0.0) = 1.0, self(1.0) = 0.0.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - beta_t (num_samples, 1, 1, 1)
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates d/dt beta_t.
        Args:
            - t: time (num_samples, 1, 1, 1)
        Returns:
            - d/dt beta_t (num_samples, 1, 1, 1)
        """
        device = t.device
        t = t.unsqueeze(1)
        dt = vmap(jacrev(self))(t)
        return dt.view(-1, 1, 1, 1).to(device)


class GaussianConditionalPath(ConditionalPath):
    def __init__(self, p_init_shape: Tuple[int], alpha: Alpha, beta: Beta):
        p_init = IsotropicGaussian(p_init_shape)
        super().__init__(p_init)
        self.alpha = alpha
        self.beta = beta

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_init = self.p_init.sample(z.shape[0])
        xt = self.alpha(t) * z + self.beta(t) * x_init
        return xt

    def conditional_velocity(self, xt: torch.Tensor, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        d_alpha_dt = self.alpha.dt(t)
        d_beta_dt = self.beta.dt(t)
        return (d_alpha_dt - d_beta_dt * alpha_t / beta_t) * z + d_beta_dt / beta_t * xt
