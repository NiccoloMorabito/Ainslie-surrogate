import datetime
import os
import re
import time
from typing import Optional
import warnings

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_LOGGER_FOLDER = os.path.join(BASE_DIR, "metrics", "logged_metrics/")

EPOCH_TIME_LABEL = "epoch_time (seconds)"

TIMESTAMP_FORMAT = "%d-%m-%Y_%H-%M-%S"


class EpochTimer:
    # NOTE: this timer considers also the idle time
    def __init__(self, epoch_num: int):
        self.__epoch_num = epoch_num
        self.__start_time = time.time()

    def get_epoch_num(self) -> int:
        return self.__epoch_num

    def stop(self) -> float:
        end_time = time.time()
        return int(end_time - self.__start_time)


class MetricsLogger:

    def __init__(
        self,
        name: str,
        automatic_save_after: int = 50,
        df_metrics: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        self.automatic_save_after = automatic_save_after
        self.__epoch_timer = None
        if df_metrics is None:
            self.name = name
            self.timestamp = datetime.datetime.now().strftime(TIMESTAMP_FORMAT)

            self.__epoch_to_metrics = {}
            self.__logged_metrics = set()
            self.__logging = True
            if self.verbose:
                print(f"Logging {self.name}", end="")
        else:
            self.filename = name
            self.name = " ".join(self.filename.split("_")[:-2])
            timestamp_string = "_".join(self.filename.split(".csv")[0].split("_")[-2:])
            try:
                self.timestamp = datetime.datetime.strptime(
                    timestamp_string, TIMESTAMP_FORMAT
                )
            except Exception:
                print(self.filename, timestamp_string)

            self.__epoch_to_metrics = {}
            for index, row in df_metrics.iterrows():
                self.__epoch_to_metrics[index] = row.to_dict()
            self.__logged_metrics = set(self.__epoch_to_metrics[0].keys())
            # self.__logged_metrics.remove(EPOCH_TIME_LABEL)
            self.df = df_metrics
            self.__logging = False
            if self.verbose:
                print(
                    f"Logged the following metrics for {self.name}: {self.get_logged_metric_names()}"
                )

    def log_metric(self, epoch_num: int, metric_name: str, metric_value: float) -> None:
        # intermediate savings (TODO the logging boolean in this case doesn't make sense)
        if (
            epoch_num not in self.__epoch_to_metrics.keys()
            and epoch_num % self.automatic_save_after == 0
            and epoch_num > 0
        ):
            self.__save_intermediate_metrics()
        self.__logging = True
        if epoch_num not in self.__epoch_to_metrics.keys():
            # measure epoch time
            self.__start_epoch_timer(epoch_num)

            self.__epoch_to_metrics[epoch_num] = {}
            if self.verbose and epoch_num % self.verbose == 0:
                print(f"\nEpoch {epoch_num} ->", end="\t")
            if self.__logged_metrics and metric_name not in self.__logged_metrics:
                warnings.warn(
                    f"The metric '{metric_name}' has not been registered in the previous epochs."
                )

        self.__epoch_to_metrics[epoch_num][metric_name] = metric_value
        self.__logged_metrics.add(metric_name)
        if self.verbose and epoch_num % self.verbose == 0:
            print(f"{metric_name}={metric_value}", end="\t")

    def __start_epoch_timer(self, epoch_num: int) -> None:
        self.__check_previous_epochs_time(epoch_num)
        if self.__epoch_timer is not None:
            self.__end_epoch_timer()
        self.__epoch_timer = EpochTimer(epoch_num)

    def __check_previous_epochs_time(self, epoch_num: int) -> None:
        if epoch_num < 1:
            return
        for i in range(epoch_num - 1):
            if EPOCH_TIME_LABEL not in self.__epoch_to_metrics[i]:
                warnings.warn(f"The epoch {i} has not been correctly stopped.")

    def __end_epoch_timer(self) -> None:
        if self.__epoch_timer is None:
            warnings.warn("Attempting to stop a non-initialized timer")
            return
        epoch_num = self.__epoch_timer.get_epoch_num()
        epoch_time = self.__epoch_timer.stop()
        if self.verbose and self.verbose == 1:
            print(f"{EPOCH_TIME_LABEL}={epoch_time}", end="\t")
        self.__epoch_timer = None
        self.__epoch_to_metrics[epoch_num][EPOCH_TIME_LABEL] = epoch_time

    def get_logged_metric_names(self) -> list[str]:
        return list(self.__logged_metrics)

    def __stop(self) -> None:
        self.__end_epoch_timer()
        self.df = pd.DataFrame.from_dict(self.__epoch_to_metrics, orient="index")
        self.df.index.name = "epoch #"
        self.__logging = False

    def plot_metrics_by_epoch(
        self,
        metric_names: Optional[list[str]] = None,
        all_in_one: bool = True,
        start_from_epoch: int = 1,
    ):
        if self.__logging:
            self.__stop()

        metrics_to_plot = (
            self.__logged_metrics if metric_names is None else metric_names
        )
        if all_in_one:
            self.__plot_many_metrics_by_epoch(metrics_to_plot, start_from_epoch)
        else:
            for metric_name in metrics_to_plot:
                self.__plot_single_metric_by_epoch(metric_name, start_from_epoch)

    def __plot_single_metric_by_epoch(
        self, metric_name: str, start_from_epoch: int = 1
    ):
        if metric_name not in self.__logged_metrics:
            warnings.warn(
                f"{metric_name} has not been logged yet: "
                + "no plot for this metric has been generated"
            )
            return
        if self.__logging:
            self.__stop()

        plt.plot(
            self.df.index[start_from_epoch:],
            self.df[metric_name].iloc[start_from_epoch:],
        )
        plt.xlabel("Epoch #")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} by Epoch")
        plt.grid(True)
        plt.show()

    def __plot_many_metrics_by_epoch(
        self, metric_names: list[str], start_from_epoch: int = 1
    ):
        missing_metrics = set(metric_names) - set(self.__logged_metrics)
        assert (
            len(missing_metrics) == 0
        ), f"The following metrics have not been logged yet: {', '.join(list(missing_metrics))}"
        if self.__logging:
            self.__stop()
        for metric_name in metric_names:
            if metric_name == EPOCH_TIME_LABEL:
                continue
            plt.plot(
                self.df.index[start_from_epoch:],
                self.df[metric_name].iloc[start_from_epoch:],
                label=metric_name,
            )
        plt.xlabel("Epoch #")
        plt.ylabel("Metric Value")
        plt.title("Metrics by Epoch")
        plt.legend()
        plt.grid(True)
        plt.show()

    def __get_filepath(self):
        if not os.path.exists(METRICS_LOGGER_FOLDER):
            os.makedirs(METRICS_LOGGER_FOLDER)
        name = re.sub(r"[^a-zA-Z0-9 -_]", "", self.name).replace(" ", "-")
        filename = f"{name}_{self.timestamp}.csv"
        return os.path.join(METRICS_LOGGER_FOLDER, filename)

    def __save_intermediate_metrics(self) -> None:
        if self.__logging:
            self.__stop()
        filepath = self.__get_filepath()
        self.df.to_csv(filepath)

    def save_metrics(self, filepath: Optional[str] = None) -> None:
        if self.__logging:
            self.__stop()

        if filepath is None:
            filepath = self.__get_filepath()

        self.df.to_csv(filepath)
        if self.verbose:
            print(f"Metrics exported in the following csv file: {filepath}")

    def get_training_time(
        self, according_to: str = "Validation loss", verbose: bool = False
    ) -> int:
        """Overall training time of the best model
        (i.e. overall time required for the best model to be obtained)"""

        according_to = self.__find_according_to_metric(according_to)

        if self.__logging:
            self.__stop()

        final_epoch = self.__find_best_model_epoch(according_to)
        training_time = self.__sum_training_times(final_epoch)
        if verbose:
            print(
                f"The best model has been generated during the {final_epoch}th epoch"
                + f" and it took {training_time} seconds for training."
            )
        return training_time

    def __find_according_to_metric(self, according_to: str):
        found = False
        for logged_metric in self.__logged_metrics:
            if according_to in logged_metric:
                according_to = logged_metric
                found = True
                break
        if not found:
            raise ValueError(f"The metric {according_to} has not been logged")
        return according_to

    def __find_best_model_epoch(self, according_to: str) -> int:
        # considering that 'according_to' is a loss -> the smallest the better
        # TODO add code to consider also other metrics (e.g. accuracy) with opposite behaviour
        final_epoch = 0
        best_value = 1_000_000.0
        for epoch, metrics in self.__epoch_to_metrics.items():
            value = metrics[according_to]
            if value < best_value:
                best_value = value
                final_epoch = epoch
        return final_epoch

    def __sum_training_times(self, final_epoch: int, threshold: float = 5) -> int:
        import numpy as np

        training_times = [
            metrics[EPOCH_TIME_LABEL]
            for epoch, metrics in self.__epoch_to_metrics.items()
            if epoch < final_epoch
        ]

        # remove outliers (sometimes epochs include stand-by time)
        avg_time_per_epoch = np.mean(training_times)
        z_scores = np.abs(
            (training_times - avg_time_per_epoch) / np.std(training_times)
        )

        training_times_cleaned = [
            int(time) if z_score <= threshold else int(avg_time_per_epoch)
            for time, z_score in zip(training_times, z_scores)
        ]

        return sum(training_times_cleaned)

    @staticmethod
    def from_csv(filepath: str, verbose: bool = False):
        logs_df = pd.read_csv(filepath, header=0, index_col="epoch #")
        return MetricsLogger(
            name=filepath.split("/")[-1], df_metrics=logs_df, verbose=verbose
        )

    @staticmethod
    def plot_many_logs_metrics_by_epoch(
        logs: list["MetricsLogger"],
        names: Optional[list[str]] = None,
        metric_names: Optional[list[str]] = None,
        start_from_epoch: int = 1,
        end_at_epoch: Optional[int] = None,
    ):
        COLORS = [
            ("darkblue", "deepskyblue"),
            ("firebrick", "lightsalmon"),
            ("darkgreen", "limegreen"),
        ]  # TODO make it dynamic
        # (a tuple of n colors for each log, where n is the number of metrics in metrics_to_plot)

        # TODO adapt the changes on this method also to the non-static version (plot_metrics_by_epoch())
        if len(logs) < 2:
            raise ValueError("The argument 'logs' should contain at least 2 logs")
        if names is None:
            names = [log.name for log in logs]
        if len(logs) != len(names):
            raise ValueError("The number of logs and names do not correspond")
        common_metrics = set(logs[0].get_logged_metric_names())
        for log in logs:
            common_metrics.intersection_update(log.get_logged_metric_names())
        if metric_names is None:
            metrics_to_plot = common_metrics
        else:
            missing_metrics = set()
            for _ in logs:
                print(
                    common_metrics,
                    set(metric_names) - set(common_metrics),
                    metric_names,
                )
                missing_metrics |= set(metric_names) - set(common_metrics)
            if len(missing_metrics) > 0:
                raise ValueError(
                    f"The following metrics have not been logged in all the passed logs:"
                    f" {', '.join(list(missing_metrics))}"
                )
            metrics_to_plot = metric_names
        metrics_to_plot.remove(EPOCH_TIME_LABEL)

        if end_at_epoch is None:
            max_epoch = min(log.df.index.max() for log in logs)
        else:
            max_epoch = end_at_epoch

        for i, log in enumerate(logs):
            for j, metric_name in enumerate(metrics_to_plot):
                plt.plot(
                    log.df.index[start_from_epoch:max_epoch],
                    log.df[metric_name].iloc[start_from_epoch:max_epoch],
                    label=metric_name + " - " + names[i],
                    color=COLORS[i][j],
                )
        plt.xlabel("Epoch #")
        plt.ylabel("MSE")
        # plt.title('Metrics by Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()
