import os
from typing import Optional, Union
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import src.data_utils as data_utils
import src.utils as utils
from src.scalers import RangeScaler

# DISCLAIMER: The code in this file is very dirty and it would need some refactoring and documentation.
# It is planned to clean up and refactor the code to improve readability and maintainability + add documentation.

# TODO the following things should be minded to clean and clarify the code:
# - explain the arguments of the different functions
# - explain univariate and multivariate as consequence of coords_as_input
# - fix the different methods for __(load_and)_split_data_* (versions for ws, versions without ws) (start from __load_and_split_data() method)
#   - distinguish among extrapolation and interpolation clearly
#   - try to standardize the extrapolation code
#   - make "consider ws" something standard and clear (not "WS in INPUT_VARIABLES")
# - fix the INPUT_VARIABLES list
# - create two subclasses for WakeDataset (one for univariate, one for multivariate) so that the different methods can be distinguished (preparation and get_params_for_plotting)
#   - perhaps implementing a different indexing for univariate, at least an additional one doing:
# - fix the possible confusion between x and y (input and output of th emodel) and coordinates x and y or the meshgrids X and Y

INPUT_VARIABLES = ["ti", "ct"]
WS = "ws"
OUTPUT_VARIABLE = "wind_deficit"
COORDS_VARIABLES = ["x/D", "y/D"]

INPUT_VARIABLE_TO_RANGE = {
    "ti": (0, 1),
    "ct": (0.1, 24 / 25),
    "ws": (4, 25),
    "x/D": (2, 30),
    "y/D": (-2, 2),
}

TI_CT_DEFAULT_REDUC_FACTOR = 4
DEFAULT_WS = 12


class WakeDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        coords_as_input: bool,
        scaler=RangeScaler(INPUT_VARIABLE_TO_RANGE),
    ) -> None:
        super().__init__()
        self.__df = df
        assert set(INPUT_VARIABLES) <= set(self.__df.columns) and {
            OUTPUT_VARIABLE
        } <= set(self.__df.columns)
        self.__scaler = scaler
        self.num_cells = df[COORDS_VARIABLES].drop_duplicates().shape[0]

        if coords_as_input:
            self.__prepare_univariate()
        else:
            self.__prepare_multivariate()

        assert len(self.inputs) == len(self.outputs)

    def __prepare_univariate(self) -> None:
        # ordering to have a wake field ordered in the self.num_cells value by coords
        self.__input_vars_names = INPUT_VARIABLES + COORDS_VARIABLES
        self.__df = self.__df.sort_values(by=self.__input_vars_names)
        self.unscaled_inputs = self.__df[self.__input_vars_names].values
        self.inputs = torch.FloatTensor(
            self.__scaler.fit_transform(self.__df[self.__input_vars_names])
        )
        self.outputs = self.__df[OUTPUT_VARIABLE].values
        self.outputs = torch.FloatTensor(self.outputs).unsqueeze(1)

        # meshgrids
        Xs, Ys = (
            self.unscaled_inputs[0 : self.num_cells, -2],
            self.unscaled_inputs[0 : self.num_cells, -1],
        )
        trasp_shape = len(np.unique(Xs)), len(np.unique(Ys))
        self.X_grid = Xs.reshape(trasp_shape)
        self.Y_grid = Ys.reshape(trasp_shape)

    def __prepare_multivariate(self) -> None:
        self.__input_vars_names = INPUT_VARIABLES

        # Group by input features and create input and output tensors
        inputs = []
        outputs = []
        for group, data in self.__df.groupby(self.__input_vars_names):
            input_tensor = torch.FloatTensor(group)
            inputs.append(input_tensor)

            output_tensor = data.pivot(
                index="y/D", columns="x/D", values=OUTPUT_VARIABLE
            ).values  # 2d output tensor with shape: (num_unique_y_values, num_unique_x_values)
            output_tensor = output_tensor.reshape(-1)  # 1d output tensor
            output_tensor = torch.FloatTensor(output_tensor)
            outputs.append(output_tensor)

            # meshgrids
            if not hasattr(self, "X_grid") or not hasattr(self, "Y_grid"):
                Xs = data[COORDS_VARIABLES[0]].values
                Ys = data[COORDS_VARIABLES[1]].values
                trasp_shape = len(np.unique(Xs)), len(np.unique(Ys))
                self.X_grid = Xs.reshape(trasp_shape).T
                self.Y_grid = Ys.reshape(trasp_shape).T

        self.inputs = torch.stack(inputs, dim=0)
        self.unscaled_inputs = self.inputs
        inputs_df = pd.DataFrame(self.inputs, columns=self.__input_vars_names)
        self.inputs = torch.FloatTensor(self.__scaler.fit_transform(inputs_df))
        self.outputs = torch.stack(outputs)

    def featurenum_to_featurename(self, featurenum: int) -> str:
        return self.__input_vars_names[featurenum]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def slice_for_field(self, field_idx: int):
        start = field_idx * self.num_cells
        end = start + self.num_cells
        return slice(start, end)

    def get_parameters_for_plotting_univariate(
        self, model, field_idx: int, transformed_inputs=None
    ) -> tuple:
        inputs, outputs = self[self.slice_for_field(field_idx)]
        unscaled_inputs = torch.Tensor(
            self.unscaled_inputs[self.slice_for_field(field_idx)]
        )
        ti, ct = unscaled_inputs[0, 0].item(), unscaled_inputs[0, 1].item()
        ws = unscaled_inputs[0, 2].item() if WS in INPUT_VARIABLES else None

        if transformed_inputs is not None:
            inputs = transformed_inputs

        if callable(model):  # PyTorch model
            predictions = model(inputs)
            # reshape outputs and predictions
            wake_field = outputs.view(-1).reshape(self.X_grid.shape)
            predicted_wake_field = predictions.view(-1).reshape(self.X_grid.shape)
        elif hasattr(model, "predict"):  # sklearn model
            predictions = model.predict(inputs)
            wake_field = outputs.reshape(self.X_grid.shape)
            predicted_wake_field = torch.Tensor(predictions).reshape(self.X_grid.shape)
        else:
            raise ValueError(
                "Invalid model type. Expected PyTorch model or sklearn model."
            )

        return ti, ct, ws, wake_field, predicted_wake_field

    def get_parameters_for_plotting_multivariate(self, model, idx: int) -> tuple:
        input, wake_field = self.inputs[idx], self.outputs[idx]
        if callable(model):  # PyTorch model
            predicted_wake_field = model(input)
        elif hasattr(model, "predict"):  # sklearn model
            predicted_wake_field = torch.tensor(
                model.predict(input.reshape(1, -1)).reshape(-1)
            )
        else:
            raise ValueError(
                "Invalid model type. Expected PyTorch model or sklearn model."
            )

        if WS in INPUT_VARIABLES:
            ti, ct, ws = self.unscaled_inputs[idx]
        else:
            ti, ct = input
            ws = None

        return ti, ct, ws, wake_field, predicted_wake_field


class DeficitDataset(WakeDataset):
    """Dataset for individual instances of the wake simulation (not the entire grid)"""

    def __init__(
        self,
        df: pd.DataFrame,
        coords_as_input: bool,
        scaler=RangeScaler(INPUT_VARIABLE_TO_RANGE),
    ) -> None:
        # super().__init__()
        self.__df = df
        assert set(INPUT_VARIABLES) <= set(self.__df.columns) and {
            OUTPUT_VARIABLE
        } <= set(self.__df.columns)
        self.__scaler = scaler

        self.__prepare_univariate()

        assert len(self.inputs) == len(self.outputs)

    def __prepare_univariate(self) -> None:
        # ordering to have a wake field ordered in the self.num_cells value by coords
        self.__input_vars_names = INPUT_VARIABLES + COORDS_VARIABLES
        self.__df = self.__df.sort_values(by=self.__input_vars_names)
        self.unscaled_inputs = self.__df[self.__input_vars_names].values
        self.inputs = torch.FloatTensor(
            self.__scaler.fit_transform(self.__df[self.__input_vars_names])
        )
        self.outputs = self.__df[OUTPUT_VARIABLE].values
        self.outputs = torch.FloatTensor(self.outputs).unsqueeze(1)

    def featurenum_to_featurename(self, featurenum: int) -> str:
        return self.__input_vars_names[featurenum]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def slice_for_field(self, field_idx: int):
        start = field_idx * self.num_cells
        end = start + self.num_cells
        return slice(start, end)

    def get_parameters_for_plotting_univariate(
        self, model, field_idx: int, transformed_inputs=None
    ) -> tuple:
        # TODO
        raise NotImplementedError(
            "Not possible to plot in this case as the wake field is not complete"
        )

    def get_parameters_for_plotting_multivariate(self, model, idx: int) -> tuple:
        input, wake_field = self.inputs[idx], self.outputs[idx]
        if callable(model):  # PyTorch model
            predicted_wake_field = model(input)
        elif hasattr(model, "predict"):  # sklearn model
            predicted_wake_field = torch.tensor(
                model.predict(input.reshape(1, -1)).reshape(-1)
            )
        else:
            raise ValueError(
                "Invalid model type. Expected PyTorch model or sklearn model."
            )

        if WS in INPUT_VARIABLES:
            ti, ct, ws = self.unscaled_inputs[idx]
        else:
            ti, ct = input
            ws = None

        return ti, ct, ws, wake_field, predicted_wake_field


def get_wake_dataloaders(
    data_filepath: str,
    consider_ws: bool,
    coords_as_input: bool,
    train_perc: float = 0.8,
    test_perc: float = 0.2,
    validation_perc: float = 0,
    input_var_to_train_reduction_factor: Optional[dict[str, int]] = None,
    input_var_to_train_ranges: Optional[dict[str, list[tuple[float, float]]]] = None,
    scaler=RangeScaler(INPUT_VARIABLE_TO_RANGE),  # MinMaxScaler()
    batch_size: Optional[int] = None,
    batch_multiplier: Optional[int] = None,
) -> Union[tuple[DataLoader, DataLoader], tuple[DataLoader, DataLoader, DataLoader]]:
    __check_train_params(
        consider_ws,
        train_perc,
        test_perc,
        validation_perc,
        input_var_to_train_reduction_factor,
        input_var_to_train_ranges,
    )
    __check_batch_params(coords_as_input, batch_size, batch_multiplier)

    dataframes = __load_and_split_data(
        data_filepath,
        consider_ws,
        train_perc,
        test_perc,
        validation_perc,
        input_var_to_train_reduction_factor,
        input_var_to_train_ranges,
    )
    datasets = __dataframes_to_datasets(dataframes, coords_as_input, scaler)

    if coords_as_input:  # batch size conversion in univariate
        try:
            batch_size = batch_multiplier * datasets[0].num_cells
        # this is for when the wake field is not complete (i.e. DeficitDataset) TODO fix it
        except Exception:
            batch_size = batch_multiplier * 500

    training_dataloader = DataLoader(datasets[0], batch_size, shuffle=True)
    if len(datasets) > 2:
        validation_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
        test_dataloader = DataLoader(datasets[2], batch_size, shuffle=False)
        return training_dataloader, validation_dataloader, test_dataloader
    test_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
    return training_dataloader, test_dataloader


def get_wake_datasets(
    data_filepath: str,
    consider_ws: bool,
    coords_as_input: bool,
    train_perc: float = 0.8,
    test_perc: float = 0.2,
    validation_perc: float = 0,
    input_var_to_train_reduction_factor: Optional[dict[str, int]] = None,
    input_var_to_train_ranges: Optional[dict[str, list[tuple[float, float]]]] = None,
    scaler=RangeScaler(INPUT_VARIABLE_TO_RANGE),
) -> list[WakeDataset]:
    __check_train_params(
        consider_ws,
        train_perc,
        test_perc,
        validation_perc,
        input_var_to_train_reduction_factor,
        input_var_to_train_ranges,
    )
    dataframes = __load_and_split_data(
        data_filepath,
        consider_ws,
        train_perc,
        test_perc,
        validation_perc,
        input_var_to_train_reduction_factor,
        input_var_to_train_ranges,
    )
    return __dataframes_to_datasets(dataframes, coords_as_input, scaler)


def __check_train_params(
    consider_ws: bool,
    train_perc: float,
    test_perc: float,
    validation_perc: float,
    input_var_to_train_reduction_factor,
    input_var_to_train_ranges,
) -> None:
    if (
        input_var_to_train_reduction_factor is not None
        and input_var_to_train_ranges is not None
    ):
        raise ValueError(
            "Cannot specify both 'input_var_to_train_reduction_factor'"
            " and 'input_var_to_train_range'"
        )
    if input_var_to_train_reduction_factor is not None:
        if consider_ws:
            raise ValueError(
                "parameter 'input_var_to_train_reduction_factor' "
                "should be specified only when not considering ws"
            )
        else:
            warnings.warn(
                f"\nIgnoring percentages of train-valid-test split "
                f"(train_perc={train_perc}, valid_perc={validation_perc}, test_perc={test_perc})\n"
                f"and using the reduction factors for the training set instead:\n"
                f"{input_var_to_train_reduction_factor}"
            )
    if input_var_to_train_ranges is not None:
        if consider_ws:
            raise ValueError(
                "parameter 'input_var_to_train_range' "
                "should be specified only when not considering ws"
            )
        else:
            warnings.warn(
                f"\nIgnoring percentages of train-valid-test split "
                f"(train_perc={train_perc}, valid_perc={validation_perc}, test_perc={test_perc})\n"
                f"and using the following ranges for the training set instead:\n"
                f"{input_var_to_train_ranges}"
            )


def __check_batch_params(
    coords_as_input: bool, batch_size: int, batch_multiplier: int
) -> None:
    if batch_size is not None and batch_multiplier is not None:
        raise ValueError("Cannot specify both batch_size and batch_multiplier.")
    elif coords_as_input and batch_multiplier is None:
        raise ValueError(
            "batch_multiplier must be specified in case of univariate setting."
        )
    elif not coords_as_input and batch_size is None:
        raise ValueError(
            "batch_size must be specified in case of multivariate setting."
        )


def __load_and_split_data(
    data_folder: str,
    consider_ws: bool,
    train_perc: float = 0.8,
    test_perc: float = 0.2,
    validation_perc: float = 0,
    input_var_to_train_reduction_factor: Optional[dict[str, int]] = None,
    input_var_to_train_ranges: Optional[dict[str, list[tuple[float, float]]]] = None,
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """In this moment, the method loads the data either including wind speed or not:
    - if yes, the data from different wind speeds is loaded and wind speed becomes an input feature;
        (different train-test splits are done on values of wind speed)
    - otherwise, a random wind speed is loaded and the input features are only the standard ones
        (different train-test splits are done on the other input features)

    This method returns 2 or 3 dataframes representing respectively: train, (validation) and test sets
    """
    if consider_ws:
        if WS not in INPUT_VARIABLES:
            INPUT_VARIABLES.append(WS)
        # TODO choose one of the following possibilities
        return __load_and_split_data_by_speed(data_folder, test_perc, validation_perc)
        # return __load_and_split_data_by_speed_alternative(data_folder, test_perc, validation_perc)
    else:
        if WS in INPUT_VARIABLES:
            INPUT_VARIABLES.remove(WS)

        df = data_utils.load_netcfd(
            data_folder, wind_speed=DEFAULT_WS, include_ws_column=True
        )
        if input_var_to_train_reduction_factor is not None:
            # TODO choose one of the following possibilities
            return __split_data_by_input_vars_uniformly(
                df, input_var_to_train_reduction_factor
            )
            # return __split_data_by_input_vars_uniformly_exclusive(df, input_var_to_train_reduction_factor)
        elif input_var_to_train_ranges is not None:
            return __split_data_by_input_vars_cutting(df, input_var_to_train_ranges)
        else:
            return __split_data_by_input_params_randomly(
                df, train_perc, test_perc, validation_perc
            )


def __load_and_split_data_by_speed(
    data_folder: str, test_perc: float = 0.2, valid_perc: float = 0
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """
    Function to split the data in training, test and possibly validation sets
    according to the wind speed.
    """
    assert os.path.isdir(
        data_folder
    ), "You need to pass a folder to load all the files in it"
    files = os.listdir(data_folder)
    assert len(files) > 0, "No files in this directory"
    files = {file for file in files if not file.startswith(".")}
    assert all(
        file.endswith(".nc") and file.startswith("ws_") for file in files
    ), "All the files in the specified folder should be nc for xarray dataset of a specific wind speed"
    wind_speeds = [int(file.split(".nc")[0].split("_")[1]) for file in files]

    ws_split_lists = utils.random_split_list(wind_speeds, test_perc, valid_perc)

    train_df = __build_set_for_different_ws(data_folder, ws_split_lists[0])
    if len(ws_split_lists) > 2:  # also validation
        validation_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
        test_df = __build_set_for_different_ws(data_folder, ws_split_lists[2])
        return train_df, validation_df, test_df
    test_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
    return train_df, test_df


def __load_and_split_data_by_speed_alternative(  # TODO change the name
    data_folder: str, test_perc: float = 0.2, valid_perc: float = 0
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """
    Function to split the data in training, test and possibly validation sets
    according to wind speed, thrust coefficient and turbulence intensity
    """
    # TODO fix and complete this
    # TODO currently, (the data is not split according to the percentages and) it doesn't utilize all the data
    # as for each set (training, test and valid) the corresponding percentage is taken for ws and then
    # for TI and CT, therefore the other values of TI and CT are not taken at all for that particular set
    assert os.path.isdir(
        data_folder
    ), "You need to pass a folder to load all the files in it"
    files = os.listdir(data_folder)
    assert len(files) > 0, "No files in this directory"
    files = {file for file in files if not file.startswith(".")}
    assert all(
        file.endswith(".nc") and file.startswith("ws_") for file in files
    ), "All the files in the specified folder should be nc for xarray dataset of a specific wind speed"
    wind_speeds = [int(file.split(".nc")[0].split("_")[1]) for file in files]
    ti_step, ct_step = data_utils.get_TIstep_from(
        data_folder
    ), data_utils.get_CTstep_from(data_folder)
    tis = list(utils.my_arange(0, 1, ti_step * TI_CT_DEFAULT_REDUC_FACTOR))
    cts = list(utils.my_arange(0.1, 24 / 25, ct_step * TI_CT_DEFAULT_REDUC_FACTOR))

    ws_split_lists = utils.random_split_list(wind_speeds, test_perc, valid_perc)
    # TODO the ordered method must be used for ti and ct for the method used later, but what about random?
    ti_split_lists = utils.ordered_split_list(tis, test_perc, valid_perc)
    ct_split_lists = utils.ordered_split_list(cts, test_perc, valid_perc)

    train_df = __build_set_for_different_ws(
        data_folder, ws_split_lists[0], ti_split_lists[0], ct_split_lists[0]
    )
    # TODO train_df is missing ON PURPOSE other dataframes:
    # 1) in case of validation
    # __build_set_for_different_ws(data_folder, ws_split_lists[1], ti_split_lists[2], ct_split_lists[2]) and
    # __build_set_for_different_ws(data_folder, ws_split_lists[2], ti_split_lists[1], ct_split_lists[1])
    # or, 2) in case of no validation
    # __build_set_for_different_ws(data_folder, ws_split_lists[1], ti_split_lists[0], ct_split_lists[0])?????
    if len(ws_split_lists) > 2:  # also validation
        validation_df1 = __build_set_for_different_ws(
            data_folder, ws_split_lists[1], ti_split_lists[1], ct_split_lists[1]
        )
        validation_df2 = __build_set_for_different_ws(
            data_folder, ws_split_lists[0], ti_split_lists[2], ct_split_lists[2]
        )
        # validation_df3 = __build_set_for_different_ws(data_folder, ws_split_lists[2],
        #                                             ti_split_lists[0], ct_split_lists[0]) excluded on purpose
        validation_df = pd.concat(
            [validation_df1, validation_df2]
        )  # , validation_df3])

        test_df1 = __build_set_for_different_ws(
            data_folder, ws_split_lists[2], ti_split_lists[2], ct_split_lists[2]
        )
        test_df2 = __build_set_for_different_ws(
            data_folder, ws_split_lists[0], ti_split_lists[1], ct_split_lists[1]
        )
        # test_df3 = __build_set_for_different_ws(data_folder, ws_split_lists[1],
        #                                       ti_split_lists[0], ct_split_lists[0]) excluded on purpose
        test_df = pd.concat([test_df1, test_df2])  # , test_df3])

        return train_df, validation_df, test_df
    test_df1 = __build_set_for_different_ws(
        data_folder, ws_split_lists[1], ti_split_lists[1], ct_split_lists[1]
    )
    test_df2 = __build_set_for_different_ws(
        data_folder, ws_split_lists[0], ti_split_lists[1], ct_split_lists[1]
    )
    test_df = pd.concat([test_df1, test_df2])
    return train_df, test_df


def __build_set_for_different_ws(
    data_folder: str,
    wind_speeds: list[int],
    tis: Optional[list[float]] = None,
    cts: Optional[list[float]] = None,
) -> pd.DataFrame:
    ti_range = (min(tis), max(tis)) if tis else None
    ct_range = (min(cts), max(cts)) if cts else None

    data_frames = []
    for wind_speed in wind_speeds:
        try:
            # by default for more speed values, only a subpart of TI-CT combinations are considered
            df = data_utils.load_netcfd(
                data_folder,
                wind_speed,
                include_ws_column=True,
                input_var_to_reduction_factor={
                    "ti": TI_CT_DEFAULT_REDUC_FACTOR,
                    "ct": TI_CT_DEFAULT_REDUC_FACTOR,
                },
                ti_range=ti_range,
                ct_range=ct_range,
            )
            data_frames.append(df)
        except Exception as e:
            print(f"Error loading data for wind speed {wind_speed}", e)

    return pd.concat(data_frames)


def __split_data_by_input_params_randomly(  # for interpolation
    df: pd.DataFrame,
    train_perc: float = 0.8,
    test_perc: float = 0.2,
    validation_perc: float = 0,
) -> Union[
    tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    """
    Function to split the data in training, test and possibly validation sets
    according to thrust coefficient and turbulence intensity
    """
    input_combs = df[INPUT_VARIABLES].drop_duplicates()

    # Sampling from input combinations
    train_size = int(len(input_combs) * train_perc)
    validation_size = int(len(input_combs) * validation_perc)
    test_size = int(len(input_combs) * test_perc)
    assert train_size + validation_size + test_size == len(
        input_combs
    ), "Split percentages should sum to 1"
    input_combs = input_combs.sample(frac=1, random_state=1)  # shuffle
    train_df = pd.merge(input_combs[:train_size], df, on=INPUT_VARIABLES, how="inner")
    validation_df = pd.merge(
        input_combs[train_size : train_size + validation_size],
        df,
        on=INPUT_VARIABLES,
        how="inner",
    )
    test_df = pd.merge(
        input_combs[train_size + validation_size :], df, on=INPUT_VARIABLES, how="inner"
    )

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    if len(validation_df) > 0:
        return train_df, validation_df, test_df
    return train_df, test_df


def __split_data_by_input_vars_uniformly(  # for interpolation
    df: pd.DataFrame, input_var_to_train_reduction_factor: dict[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_var_values = []

    variables = []
    for input_var, reduction_factor in input_var_to_train_reduction_factor.items():
        values = df[input_var].drop_duplicates().sort_values().reset_index(drop=True)
        train_values = list(values[::reduction_factor])
        variables.append(input_var)
        train_var_values.append(train_values)

    train_input_combs = pd.DataFrame(
        list(product(*train_var_values)), columns=variables
    )

    input_combs = (
        df[variables].drop_duplicates().sort_values(by=variables).reset_index(drop=True)
    )
    remaining_input_combs = pd.merge(
        input_combs, train_input_combs, on=variables, how="left", indicator=True
    )
    remaining_input_combs = remaining_input_combs[
        remaining_input_combs["_merge"] == "left_only"
    ][variables]

    valid_input_combs = remaining_input_combs.iloc[::2]
    test_input_combs = remaining_input_combs.iloc[1::2]

    assert len(train_input_combs) + len(valid_input_combs) + len(
        test_input_combs
    ) == len(input_combs)

    dfs = []
    for split_input_combs in [train_input_combs, valid_input_combs, test_input_combs]:
        split_df = pd.merge(split_input_combs, df, on=variables, how="inner")
        dfs.append(split_df)

    return tuple(dfs)


def __split_data_by_input_vars_uniformly_exclusive(  # for interpolation 2
    # TODO probably useless as the train values are the same, this just misses the chance to test on more
    df: pd.DataFrame,
    input_var_to_train_reduction_factor: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # this method is similar to __load_and_split_data_by_input_vars_uniformly
    # but it does not use all the combinations of input_vars,
    # it just combined the combinations of input_vars that each split (train, test, valid) has

    # the difference is basically whether a certain value a of CT (or TI) that is used in the
    # train with a value of TI (or CT) b appears also in the test, combined with a different
    # value c of TI (or CT) or not

    split_var_values = []

    for input_var, reduction_factor in input_var_to_train_reduction_factor.items():
        values = df[input_var].drop_duplicates().sort_values().reset_index(drop=True)
        train_values = list(values[::reduction_factor])
        remaining_values = values[~values.isin(train_values)]
        valid_values = list(remaining_values[::2])
        test_values = list(remaining_values[1::2])

        assert len(train_values) + len(valid_values) + len(test_values) == len(values)

        split_var_values.append((train_values, valid_values, test_values))

    train_list = [t[0] for t in split_var_values]
    valid_list = [t[1] for t in split_var_values]
    test_list = [t[2] for t in split_var_values]

    dfs = []
    variables = list(input_var_to_train_reduction_factor.keys())
    for split in [train_list, valid_list, test_list]:
        split_input_combs = (
            pd.DataFrame(list(product(*split)), columns=variables)
            .sort_values(by=variables)
            .reset_index(drop=True)
        )
        split_df = pd.merge(split_input_combs, df, on=variables, how="inner")
        dfs.append(split_df)

    return tuple(dfs)


def __split_data_by_input_vars_cutting(  # for extrapolation
    df: pd.DataFrame, input_var_to_train_ranges: dict[str, list[tuple[float, float]]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_var_values = []

    variables = []
    for input_var, train_ranges in input_var_to_train_ranges.items():
        values = df[input_var].drop_duplicates().sort_values().reset_index(drop=True)
        train_values = []

        for train_range in train_ranges:
            min_value, max_value = train_range
            train_values += [
                value for value in values if min_value <= value <= max_value
            ]

        variables.append(input_var)
        train_var_values.append(train_values)

    train_input_combs = pd.DataFrame(
        list(product(*train_var_values)), columns=variables
    )

    input_combs = (
        df[variables].drop_duplicates().sort_values(by=variables).reset_index(drop=True)
    )
    remaining_input_combs = pd.merge(
        input_combs, train_input_combs, on=variables, how="left", indicator=True
    )
    remaining_input_combs = remaining_input_combs[
        remaining_input_combs["_merge"] == "left_only"
    ][variables]

    valid_input_combs = remaining_input_combs.iloc[::2]
    test_input_combs = remaining_input_combs.iloc[1::2]

    assert len(train_input_combs) + len(valid_input_combs) + len(
        test_input_combs
    ) == len(input_combs)

    dfs = []
    for split_input_combs in [train_input_combs, valid_input_combs, test_input_combs]:
        split_df = pd.merge(split_input_combs, df, on=variables, how="inner")
        dfs.append(split_df)

    return tuple(dfs)


def __dataframes_to_datasets(dfs, coords_as_input: bool, scaler):
    # if the wake field is not complete
    if (
        not dfs[0][COORDS_VARIABLES]
        .drop_duplicates()
        .equals(dfs[1][COORDS_VARIABLES].drop_duplicates())
    ):
        print("Incomplete field, creating DeficitDatasets...")
        return [DeficitDataset(df, coords_as_input, scaler) for df in dfs]
    return [WakeDataset(df, coords_as_input, scaler) for df in dfs]
