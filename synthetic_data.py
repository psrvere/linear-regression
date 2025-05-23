import numpy as np
import matplotlib.pyplot as plt


def generate_data(true_weight, true_bias, n_samples):
    x = np.random.rand(n_samples).astype(np.float32) * 100

    # Generate y values with some noise
    noise = np.random.randn(n_samples).astype(np.float32) * 30
    y = true_weight * x + true_bias + noise
    return x, y


def plot_data(x, y, true_weight, true_bias):
    # Plot the data
    plt.scatter(x, y, label="Synthetic Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Synthetic Data for Linear Regression")

    # plot true line
    plt.plot(x, true_weight * x + true_bias, label="True Line", color="red")

    plt.legend()
    plt.show()