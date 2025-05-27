import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Optional, Tuple

@dataclass
class SyntheticDataConfig:
    true_weight: float
    true_bias: float
    n_samples: int
    noise: Optional[float] = None

class SyntheticDataGenerator:
    @staticmethod
    def generate_continuous_data(true_weight: float, true_bias: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        step = 1 / n_samples
        X: torch.Tensor = torch.arange(0, 1, step, dtype=torch.float32).unsqueeze(1)
        y: torch.Tensor = true_weight * X + true_bias
        return X, y
    
    @staticmethod
    def generate_continuous_data_with_noise(true_weight: float, true_bias: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = SyntheticDataGenerator.generate_continuous_data(true_weight, true_bias, n_samples)
        y = SyntheticDataGenerator.add_noise(y, n_samples)
        return X, y

    @staticmethod
    def generate_discontinuous_data(true_weight: float, true_bias: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x: List[float] = np.random.rand(n_samples).astype(np.float32)

        y: List[float] = true_weight * x + true_bias
        
        X: torch.Tensor = torch.from_numpy(x).reshape(-1, 1)
        y: torch.Tensor = torch.from_numpy(y).reshape(-1, 1)

        return X, y

    @staticmethod
    def generate_discontinuous_data_with_noise(true_weight: float, true_bias: float, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x: List[float] = np.random.rand(n_samples).astype(np.float32)
        y: List[float] = true_weight * x + true_bias
        y = SyntheticDataGenerator.add_noise(y, n_samples)

        X: torch.Tensor = torch.from_numpy(x).reshape(-1, 1)
        y: torch.Tensor = torch.from_numpy(y).reshape(-1, 1)

        return X, y
    
    @staticmethod
    def add_noise(y: List[float], n_samples: int) -> List[float]:
        # randn will create normally distributed noise
        noise: List[float] = np.random.rand(n_samples).astype(np.float32)
        print("noise min", noise.min(), "noise max", noise.max())
        y = y + noise
        return y

# def normalize_data(y: List[float]):
#     min = y.min()
#     max = y.max()
#     for i, val in enumerate(y):
#         y[i] = (val - min) / (max - min)
#     return y

# def plot_data(x, y):
#     # Plot the data
#     plt.scatter(x, y, label="Synthetic Data")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.title("Synthetic Data for Linear Regression")

#     # plot true line
#     # plt.plot(x, true_weight * x + true_bias, label="True Line", color="red")

#     plt.legend()
#     plt.show()