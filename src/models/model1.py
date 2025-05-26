import torch
import torch.nn as nn
from .base import BaseModel

class LinearRegressionModel1(BaseModel):
    def __init__(self):
        super().__init_()
        self.Linear = nn.Linear(1,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Linear(x)