import sklearn.metrics as metrics
import numpy as np

"""
def peak_signal_noise_ratio(ground_truths, predictions,
                            data_range: int = 1) -> float:
    psnr_values = list()
    for true_wakefield, predicted_wakefield in zip(ground_truths, predictions):
        true_wakefield = np.asarray(true_wakefield, dtype=np.float64)
        predicted_wakefield = np.asarray(predicted_wakefield, dtype=np.float64)
        mse = np.mean((true_wakefield - predicted_wakefield) ** 2, dtype=np.float64)
        if mse > 0:
            psnr = 10 * np.log10((data_range ** 2) / mse)
            psnr_values.append(psnr)
    return np.mean(psnr_values)
"""

def peak_signal_noise_ratio(ground_truths, predictions,
                            data_range: int = 1) -> float:
    total_mse = metrics.mean_squared_error(ground_truths, predictions)
    return float(10 * np.log10((data_range ** 2) / total_mse))
