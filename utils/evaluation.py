import time
import datetime
import pandas as pd
import torch
import sklearn.metrics as metrics

METRICS = [metrics.r2_score, metrics.explained_variance_score, metrics.mean_squared_error, \
           metrics.mean_absolute_error, metrics.median_absolute_error, metrics.mean_absolute_percentage_error]
#TODO are mean_percentage_error and relative_error missing?
MODEL_DESC = "model_description"
PREDICTION_TIME = "prediction_time"
TIMESTAMP = "timestamp"
COLUMNS_ORDER = [MODEL_DESC, PREDICTION_TIME] + [metric.__name__ for metric in METRICS] + [TIMESTAMP]
METRICS_CSV_FILEPATH = "results.csv"

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

def __get_predictions_outputs_time(model, test_dataloader) -> tuple[torch.Tensor, torch.Tensor, float]:
    predictions = list()
    outputs = list()
    with torch.no_grad():
        start_time = time.time()
        for input, output in test_dataloader:
            predictions.append(model(input))
            outputs.append(output)
        end_time = time.time()

    predictions = torch.cat(predictions, dim=0)
    outputs = torch.cat(outputs, dim=0)
    prediction_time = (end_time-start_time) / outputs.shape[0]
    return predictions, outputs, prediction_time

def __get_predictions_time(model, test_x) -> tuple[torch.Tensor, float]:
    start_time = time.time()
    preds = model.predict(test_x)
    end_time = time.time()
    prediction_time = (end_time-start_time) / test_x.shape[0]
    return preds, prediction_time

def __compute_other_metrics(outputs, predictions) -> dict[str, float]:
    metric_to_value = dict()
    for metric in METRICS:
        metric_to_value[metric.__name__] = metric(outputs, predictions)
        print(f"{metric.__name__}={metric(outputs, predictions)}")
    return metric_to_value

def test_pytorch_model(model, test_dataloader,
                                    model_description: str, save_results: bool) -> None:
    predictions, outputs, prediction_time = __get_predictions_outputs_time(model, test_dataloader)
    print("Test results for " + model_description)
    metrics = __compute_other_metrics(outputs, predictions)
    metrics[MODEL_DESC] = model_description
    metrics[PREDICTION_TIME] = prediction_time
    print(f"Prediction time={prediction_time}s")
    metrics[TIMESTAMP] = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if save_results:
        __save_to_csv(METRICS_CSV_FILEPATH, metrics)

def test_sklearn_model(model, test_x, test_y, model_description: str, save_results: bool) -> None:
    predictions, prediction_time = __get_predictions_time(model, test_x)
    metrics = __compute_other_metrics(test_y, predictions)
    metrics[MODEL_DESC] = model_description
    metrics[PREDICTION_TIME] = prediction_time
    metrics[TIMESTAMP] = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    if save_results:
        __save_to_csv(METRICS_CSV_FILEPATH, metrics)
