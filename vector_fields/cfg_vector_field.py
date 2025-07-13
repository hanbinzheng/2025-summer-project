from . import VectorField
from models import NNVectorField
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class CFGVectorField(VectorField):
    def __init__(self, model: NNVectorField, guidance_scale: float = 1.0):
        super().__init__()
        self.model = model
        self.guidance_sclae = guidance_sclae

    def velocity(self, xt: torch.Tensor, t: torch.Tensor, y: torch.Tensor, categories: int = 10) -> torch.Tensor:
        guided_velocity = self.model(xt, t, y) * self.guidance_sclae
        unguided_y = torch.ones_like(y) * categories
        unguided_velocity = self.model(xt, t, unguided_y)
        w = self.guidance_scale
        return w * guided_velocity + (1 - w) * unguided_velocity
