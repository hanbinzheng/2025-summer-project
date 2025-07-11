from .base import Simulator
from models import VectorField
import torch


class HeunSimulator(Simulator):
    def __init__(self, velocity: VectorField):
        self.velocity = velocity


    @torch.no_grad()
    def step(self, xt: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, **kwargs) -> torch.Tensor:
        vt = self.velocity(xt, t, **kwargs)
        xt_next = xt + vt * dt
        t_next = t + dt
        vt_next = self.velocity(xt_next, t_next, **kwargs)
        return xt + 0.5 * dt * (vt + vt_next)
