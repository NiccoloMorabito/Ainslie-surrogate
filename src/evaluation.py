import datetime
import time
from typing import Optional, Union

import gpytorch
import sklearn.metrics as metrics
import torch
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
import os

from src.metrics import peak_signal_noise_ratio
from src.utils import save_metrics_to_csv

MODEL_DESC = "model_description"
PREDICTION_TIME = (
    "prediction_time"  # prediction time per simulation (i.e. whole wakefield)
)
TIMESTAMP = "timestamp"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINSET_CSV_FILEPATH = os.path.join(
    BASE_DIR, "metrics", "final_results", "trainset_results.csv"
)
TESTSET_CSV_FILEPATH = os.path.join(
    BASE_DIR, "metrics", "final_results", "testset_results.csv"
)


METRICS = [
    metrics.r2_score,
    metrics.explained_variance_score,
    metrics.mean_squared_error,
    metrics.mean_absolute_error,
    metrics.median_absolute_error,
    metrics.mean_absolute_percentage_error,
    peak_signal_noise_ratio,
]
COLUMNS_ORDER = (
    [MODEL_DESC, PREDICTION_TIME]
    + [metric.__name__ for metric in METRICS]
    + [TIMESTAMP]
)


def __get_predictions_groundtruths_time(
    model, dataloader: DataLoader
) -> tuple[torch.Tensor, torch.Tensor, float]:
    predictions = list()
    ground_truths = list()
    with torch.no_grad():
        start_time = time.time()
        for input, output in dataloader:
            predictions.append(model(input))
            ground_truths.append(output)
        end_time = time.time()

    predictions = torch.cat(predictions, dim=0)
    ground_truths = torch.cat(ground_truths, dim=0)
    num_simulations = ground_truths.shape[0]
    prediction_time = (end_time - start_time) / num_simulations
    return predictions, ground_truths, prediction_time


def __get_predictions_time(model, test_x) -> tuple[torch.Tensor, float]:
    start_time = time.time()
    predictions = model.predict(test_x)
    end_time = time.time()
    num_simulations = test_x.shape[0]
    prediction_time = (end_time - start_time) / num_simulations
    return predictions, prediction_time


def __get_predictions_time_gpy(model, likelihood, test_x) -> tuple[torch.Tensor, float]:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        start_time = time.time()
        predictions = likelihood(model(test_x))
        end_time = time.time()
    num_simulations = test_x.shape[0]
    prediction_time = (end_time - start_time) / num_simulations
    return predictions, prediction_time


def __compute_other_metrics(
    outputs, predictions, model_description: str, prediction_time: float
) -> dict[str, float]:
    metric_to_value = {}
    for metric in METRICS:
        metric_to_value[metric.__name__] = metric(outputs, predictions)
        print(f"{metric.__name__}={metric(outputs, predictions)}")
    print(f"Prediction time={prediction_time}s")
    metric_to_value[MODEL_DESC] = model_description
    metric_to_value[PREDICTION_TIME] = prediction_time
    metric_to_value[TIMESTAMP] = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    return metric_to_value


def evaluate_model(
    model,
    data: Union[DataLoader, tuple[torch.Tensor, torch.Tensor]],
    data_type: str,
    model_description: str,
    save_results: bool,
    experiment: Optional[str] = None,
) -> None:
    if data_type == "test":
        print(f"Test results for {model_description}")
        filepath = TESTSET_CSV_FILEPATH
    elif data_type == "train":
        print(f"Train results for {model_description}")
        filepath = TRAINSET_CSV_FILEPATH
    else:
        raise ValueError(f"dataloader_type must be 'train' or 'test', not {data_type}")

    if experiment is not None:
        filepath = filepath.replace(".csv", f"_{experiment}.csv")
        print(filepath)

    # PyTorch model
    if isinstance(model, torch.nn.Module) and isinstance(data, DataLoader):
        dataloader = data
        predictions, ground_truths, prediction_time = (
            __get_predictions_groundtruths_time(model, dataloader)
        )
    # sklearn model
    elif (
        isinstance(model, BaseEstimator)
        and isinstance(data, tuple)
        and all(isinstance(item, torch.Tensor) for item in data)
    ):
        x, ground_truths = data
        predictions, prediction_time = __get_predictions_time(model, x)
    else:
        raise ValueError(
            "Unsupported model or data type. Pass a PyTorch, GpyTorch or "
            "Sklearn model and a DataLoader or tuple of tensors."
        )
    """
    # GpyTorch model (not used for the moment, missing likelihood as parameter)
    elif isinstance(model, gpytorch.models.GP):
        x, ground_truths = data
        predictions, prediction_time = __get_predictions_time_gpy(model, likelihood, x)
    """

    metrics = __compute_other_metrics(
        ground_truths, predictions, model_description, prediction_time
    )

    if save_results:
        save_metrics_to_csv(filepath, metrics, COLUMNS_ORDER)
