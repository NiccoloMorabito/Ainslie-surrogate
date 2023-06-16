import numpy as np
import torch.nn as nn
import torch
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import re
import time

"""
#TODO remove v
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-12):
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
#TODO remove ^
"""

class EpochTimer:
    def __init__(self, epoch_num: int):
        self.__epoch_num = epoch_num
        self.__start_time = time.time()
    
    def get_epoch_num(self) -> int:
        return self.__epoch_num

    def stop(self) -> float:
        end_time = time.time()
        return int(end_time - self.__start_time)

EPOCH_TIME_LABEL = "epoch_time (seconds)"

class MetricsLogger:

    def __init__(self, name: str, df_metrics: pd.DataFrame | None = None) -> None:
        self.name = name
        if df_metrics is None:
            self.__epoch_to_metrics = dict()
            self.__logged_metrics = set()
            self.__logging = True
            print(f"Logging {name}", end="")
        else:
            self.__epoch_to_metrics = dict()
            for index, row in df_metrics.iterrows():
                self.__epoch_to_metrics[index] = row.to_dict()
            self.__logged_metrics = set(self.__epoch_to_metrics[0].keys())
            self.df = df_metrics
            self.__logging = False
        self.__epoch_timer = None
        self.timestamp = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    def log_metric(self, epoch_num: int, metric_name: str, metric_value: float) -> None:
        # intermediate savings (TODO the logging boolean in this case doesn't make sense)
        if epoch_num not in self.__epoch_to_metrics.keys() and epoch_num%50==0 and epoch_num>0:
            self.__save_intermediate_metrics()
        self.__logging = True
        if epoch_num not in self.__epoch_to_metrics.keys():            
            self.__epoch_to_metrics[epoch_num] = dict()
            print(f"\nEpoch {epoch_num} ->", end="\t")
            if self.__logged_metrics and metric_name not in self.__logged_metrics:
                warnings.warn(
                    f"The metric '{metric_name}' has not been registered in the previous epochs.")

            # measure epoch time
            self.__start_epoch_timer(epoch_num)
        
        self.__epoch_to_metrics[epoch_num][metric_name] = metric_value
        self.__logged_metrics.add(metric_name)
        print(f"{metric_name}={metric_value}", end="\t")
    
    def __start_epoch_timer(self, epoch_num: int) -> None:
        self.__check_previous_epochs_time(epoch_num) #TODO
        if self.__epoch_timer is not None:
            self.__end_epoch_timer()
        self.__epoch_timer = EpochTimer(epoch_num)
    
    def __check_previous_epochs_time(self, epoch_num: int) -> None:
        #TODO this method should be deleted at some point (just to check for bugs currently)
        if epoch_num < 1:
            return
        for i in range(epoch_num-1):
            if EPOCH_TIME_LABEL not in self.__epoch_to_metrics[i]:
                warnings.warn(
                    f"The epoch {i} has not been correctly stopped while executing epoch {epoch_num}")
    
    def __end_epoch_timer(self) -> None:
        if self.__epoch_timer is None:
            warnings.warn(f"Attempting to stop a non-initialized timer")
            return
        epoch_num = self.__epoch_timer.get_epoch_num()
        epoch_time = self.__epoch_timer.stop()
        self.__epoch_timer = None
        self.__epoch_to_metrics[epoch_num][EPOCH_TIME_LABEL] = epoch_time
    
    def get_logged_metric_names(self) -> list[str]:
        return list(self.__logged_metrics)
    
    def __stop(self) -> None:
        self.__end_epoch_timer()
        self.df = pd.DataFrame.from_dict(self.__epoch_to_metrics, orient='index')
        self.df.index.name = 'epoch #'
        self.__logging = False

    def plot_metrics_by_epoch(self, metric_names: list[str] | None = None, all_in_one: bool = True):
        if self.__logging:
            self.__stop()

        metrics_to_plot = self.__logged_metrics if metric_names is None else metric_names
        if all_in_one:
            self.__plot_many_metrics_by_epoch(metrics_to_plot)
        else:
            for metric_name in metrics_to_plot:
                self.__plot_single_metric_by_epoch(metric_name)
    
    def __plot_single_metric_by_epoch(self, metric_name: str):
        if metric_name not in self.__logged_metrics:
            warnings.warn(f"{metric_name} has not been logged yet: " +\
                          "no plot for this metric has been generated")
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
        missing_metrics = set(metric_names) - set(self.__logged_metrics)
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
    
    def __get_filepath(self):
        folder = "logged_metrics/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        name = re.sub(r'[^a-zA-Z0-9 -_]', '', self.name).replace(" ", "-")
        filename = f"{name}_{self.timestamp}.csv"
        return os.path.join(folder, filename)
    
    def __save_intermediate_metrics(self) -> None:
        if self.__logging:
            self.__stop()
        filepath = self.__get_filepath()
        self.df.to_csv(filepath)

    def save_metrics(self, filepath: str | None = None) -> None:
        if self.__logging:
            self.__stop()

        if filepath is None:
            filepath = self.__get_filepath()
        
        self.df.to_csv(filepath)
        print(f"Metrics exported in the following csv file: {filepath}")
    
    @staticmethod
    def from_csv(filepath: str):
        logs_df = pd.read_csv(filepath, header=0, index_col='epoch #')
        metrics_logger = MetricsLogger(filepath.split("/")[1].split("_")[0], logs_df)
        return metrics_logger
