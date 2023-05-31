import numpy as np
import torch.nn as nn
import torch
import warnings
import matplotlib.pyplot as plt
import pandas as pd

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

def field_based_rmse(wf1, wf2):
    diff = wf1 - wf2
    diff = diff.detach().numpy()
    return np.sqrt(np.mean(np.square(diff)))

def cell_based_rmse(wf1, wf2):
    diff = wf1 - wf2
    return np.sqrt(np.mean(np.square(diff), axis=1))

def cell_based_mae(wf1, wf2):
    # Mean absolute error = L1
    absolute_diff = np.abs(wf1 - wf2)
    return np.mean(absolute_diff, axis=1)

class MetricsLogger:

    def __init__(self, name: str, df_metrics: pd.DataFrame | None = None) -> None:
        self.name = name
        if df_metrics is None:
            self.epoch_to_metrics = dict()
            self.logged_metrics = set()
            self.__logging = True
            print(f"Logging {name}", end="")
        else:
            self.epoch_to_metrics = dict()
            for index, row in df_metrics.iterrows():
                self.epoch_to_metrics[index] = row.to_dict()
            self.logged_metrics = set(self.epoch_to_metrics[0].keys())
            self.df = df_metrics
            self.__logging = False
    
    def log_metric(self, epoch_num: int, metric_name: str, metric_value: float) -> None:
        self.__logging = True
        if epoch_num not in self.epoch_to_metrics.keys():
            self.epoch_to_metrics[epoch_num] = dict()
            print(f"\nEpoch {epoch_num} ->", end="\t")
            if self.logged_metrics and metric_name not in self.logged_metrics:
                warnings.warn(
                    f"The metric '{metric_name}' has not been registered in the previous epochs.")
        self.epoch_to_metrics[epoch_num][metric_name] = metric_value
        self.logged_metrics.add(metric_name)
        print(f"{metric_name}={metric_value}", end="\t")
    
    def get_logged_metric_names(self) -> list[str]:
        return list(self.logged_metrics)
    
    def __stop(self) -> None:
        self.df = pd.DataFrame.from_dict(self.epoch_to_metrics, orient='index')
        self.df.index.name = 'epoch #'
        self.__logging = False

    def plot_metrics_by_epoch(self, metric_names: list[str] | None = None, all_in_one: bool = True):
        if self.__logging:
            self.__stop()

        metrics_to_plot = self.logged_metrics if metric_names is None else metric_names
        if all_in_one:
            self.__plot_many_metrics_by_epoch(metrics_to_plot)
        else:
            for metric_name in metrics_to_plot:
                self.__plot_single_metric_by_epoch(metric_name)
    
    def __plot_single_metric_by_epoch(self, metric_name: str):
        if metric_name not in self.logged_metrics:
            warnings.warn(f"{metric_name} has not been logged yet, no plot for this metric has been generated")
            return
        if self.__logging:
            self.__stop()
        
        plt.plot(self.df.index, self.df[metric_name])
        plt.xlabel('Epoch #')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} by Epoch')
        plt.grid(True)
        plt.show()
    
    def __plot_many_metrics_by_epoch(self, metric_names):
        missing_metrics = set(metric_names) - set(self.logged_metrics)
        assert len(missing_metrics) == 0, \
            f"The following metrics have not been logged yet: {', '.join(list(missing_metrics))}"
        if self.__logging:
            self.__stop()
        for metric_name in metric_names:
            plt.plot(self.df.index, self.df[metric_name], label=metric_name)
        plt.xlabel('Epoch #')
        plt.ylabel('Metric Value')
        plt.title('Metrics by Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_metrics(self, filepath: str | None = None) -> None:
        if self.__logging:
            self.__stop()

        if filepath is None:
            import datetime, os, re
            folder = "logged_metrics/"
            if not os.path.exists(folder):
                os.makedirs(folder)
            timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            name = re.sub(r'[^a-zA-Z0-9 -_]', '', self.name).replace(" ", "-")
            filename = f"{name}_{timestamp}.csv"
            filepath = os.path.join(folder, filename)
        self.df.to_csv(filepath)
        print(f"Metrics exported in the following csv file: {filepath}")
    
    @staticmethod
    def from_csv(filepath: str):
        logs_df = pd.read_csv(filepath, header=0, index_col='epoch #')
        metrics_logger = MetricsLogger(filepath.split("/")[1].split("_")[0], logs_df)
        return metrics_logger
