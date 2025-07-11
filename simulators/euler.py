from .base import Simulator
from models import VectorField
import torch

class EulerSimulator(Simulator):
    def __init__(self, velocity: VectorField):
        self.velocity = velocity

    @torch.no_grad()
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        return xt + self.velocity(xt, t, **kwargs) * dt
