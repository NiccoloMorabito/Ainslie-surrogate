import time
import datetime
import pandas as pd
import torch
from torch.utils.data import DataLoader
import gpytorch
from sklearn.base import BaseEstimator
import sklearn.metrics as metrics

METRICS = [metrics.r2_score, metrics.explained_variance_score, metrics.mean_squared_error, \
           metrics.mean_absolute_error, metrics.median_absolute_error, metrics.mean_absolute_percentage_error]
MODEL_DESC = "model_description"
PREDICTION_TIME = "prediction_time"
TIMESTAMP = "timestamp"
COLUMNS_ORDER = [MODEL_DESC, PREDICTION_TIME] + [metric.__name__ for metric in METRICS] + [TIMESTAMP]
TRAINSET_CSV_FILEPATH = "metrics/final results/trainset_results.csv"
TESTSET_CSV_FILEPATH = "metrics/final results/testset_results.csv"

#TODO move in another utils file
def __save_to_csv(filename: str, metrics: dict[str, float]) -> None:
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame()

    new_df = pd.DataFrame(metrics, index=[0])\
        .reindex(columns=COLUMNS_ORDER)

    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(filename, index=False)

def __get_predictions_groundtruths_time(
        model,
        dataloader: DataLoader)-> tuple[torch.Tensor, torch.Tensor, float]:
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
    prediction_time = (end_time-start_time) / ground_truths.shape[0]
    return predictions, ground_truths, prediction_time

def __get_predictions_time(model, test_x) -> tuple[torch.Tensor, float]:
    start_time = time.time()
    predictions = model.predict(test_x)
    end_time = time.time()
    prediction_time = (end_time-start_time) / test_x.shape[0]
    return predictions, prediction_time

def __get_predictions_time_gpy(model, likelihood, test_x) -> tuple[torch.Tensor, float]:
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        start_time = time.time()
        predictions = likelihood(model(test_x))
        end_time = time.time()
        
    prediction_time = (end_time-start_time) / test_x.shape[0]
    return predictions, prediction_time

def __compute_other_metrics(outputs, predictions,
                            model_description: str,
                            prediction_time: float) -> dict[str, float]: #TODO change name
    metric_to_value = dict()
    for metric in METRICS:
        metric_to_value[metric.__name__] = metric(outputs, predictions)
        print(f"{metric.__name__}={metric(outputs, predictions)}")
    print(f"Prediction time={prediction_time}s")
    metric_to_value[MODEL_DESC] = model_description
    metric_to_value[PREDICTION_TIME] = prediction_time
    metric_to_value[TIMESTAMP] = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    return metric_to_value
    

def evaluate_model(model, data: DataLoader | tuple[torch.Tensor, torch.Tensor],
                   data_type : str,
                   model_description: str,
                   save_results: bool) -> None:
    if data_type == 'train':
        print("Train results for " + model_description)
        filepath = TRAINSET_CSV_FILEPATH
    elif data_type == 'test':
        print("Test results for " + model_description)
        filepath = TESTSET_CSV_FILEPATH
    else:
        raise ValueError(f"dataloader_type must be 'train' or 'test', not {data_type}")
    
    # PyTorch model
    if isinstance(model, torch.nn.Module):
        dataloader = data
        predictions, ground_truths, prediction_time = __get_predictions_groundtruths_time(model, dataloader)
    # sklearn model
    elif isinstance(model, BaseEstimator):
        x, ground_truths = data
        predictions, prediction_time = __get_predictions_time(model, x)
    else:
        raise ValueError("Unsupported model type, pass a PyTorch, GpyTorch or Sklearn model.")
    """
    # GpyTorch model (not used for the moment, missing likelihood as parameter)
    elif isinstance(model, gpytorch.models.GP):
        x, ground_truths = data
        predictions, prediction_time = __get_predictions_time_gpy(model, likelihood, x)
    """

    metrics = __compute_other_metrics(ground_truths, predictions,
                                      model_description, prediction_time)    

    if save_results:
        __save_to_csv(filepath, metrics)
