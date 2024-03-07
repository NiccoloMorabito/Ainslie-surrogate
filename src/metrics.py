import numpy as np
import sklearn.metrics as metrics


def peak_signal_noise_ratio(ground_truths, predictions, data_range: int = 1) -> float:
    total_mse = metrics.mean_squared_error(ground_truths, predictions)
    return float(10 * np.log10((data_range**2) / total_mse))
