import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path))

    # def train(self, X: torch.Tensor, y: torch.Tensor, epochs: int, learning_rate: float):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
    #     loss_fn = nn.MSELoss()
