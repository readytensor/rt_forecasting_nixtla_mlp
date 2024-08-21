import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    DotProduct,
)


class KernelSynthesizer:
    def __init__(self, n_kernels=5, kernel_bank=None):
        """
        Initialize the WindowedKernelSynthesizer.

        Parameters:
        - n_kernels (int): Number of kernels to sample.
        - kernel_bank (list): List of kernels to choose from.
        """
        self.n_kernels = n_kernels
        self.kernel_bank = (
            kernel_bank
            if kernel_bank is not None
            else [
                # Linear kernels with different sigma_0 values
                DotProduct(sigma_0=1.0),
                DotProduct(sigma_0=0.1),
                # RBF kernels with different length scales
                RBF(length_scale=1.0),
                RBF(length_scale=0.1),
                RBF(length_scale=10.0),
                # Periodic kernels with different length scales and periodicities
                ExpSineSquared(length_scale=1.0, periodicity=3.0),
                ExpSineSquared(length_scale=0.5, periodicity=1.0),
                ExpSineSquared(length_scale=2.0, periodicity=5.0),
                ExpSineSquared(length_scale=10.0, periodicity=2.0),
            ]
        )

    def generate(self, shape):
        """
        Generates synthetic time series data for each window in the input data.

        Parameters:
        - shape (tuple): The shape of the input data (n_series, series_length, n_dimensions).

        Returns:
        - np.ndarray: The synthetic time-series data for each dimension, maintaining the input shape.
        """
        n, l, d = shape
        synthetic_series = np.zeros((n, l, d))

        for i in tqdm(range(n), desc="Generating synthetic series"):
            for dim in range(d):
                # Sample kernels and combine them
                sampled_kernels = np.random.choice(
                    self.kernel_bank, size=self.n_kernels, replace=True
                )
                combined_kernel = sampled_kernels[0]
                for k in sampled_kernels[1:]:
                    if np.random.rand() > 0.5:
                        combined_kernel += k
                    else:
                        combined_kernel *= k

                # Create a GP model with the combined kernel
                gp = GaussianProcessRegressor(kernel=combined_kernel, random_state=42)

                # Assuming evenly spaced time points within each window
                time_points = np.linspace(0, l - 1, l).reshape(-1, 1)
                # Sample synthetic data from the GP model for the current window and dimension
                synthetic_series[i, :, dim] = gp.sample_y(time_points, 1).flatten()

        data = pd.DataFrame(synthetic_series.reshape(n * l, 1))

        ids = []
        time = []
        for i in range(n):
            ids.extend([f"id_{i}"] * l)
            time.extend(list(range(l)))

        data.insert(0, "unique_id", ids)
        data.insert(1, "ds", time)

        data.columns = ["unique_id", "ds", "y"]

        return data


k = KernelSynthesizer()

# Generate synthetic time series data
synthetic_data = k.generate((10, 100, 1))
