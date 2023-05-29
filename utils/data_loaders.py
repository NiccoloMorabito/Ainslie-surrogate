import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import utils.utils as utils
import utils.data_utils as data_utils

INPUT_VARIABLES = ["ti", "ct"]
WS = "ws"
OUTPUT_VARIABLE = "wind_deficit"
COORDS_VARIABLES = ["x/D", "y/D"]

#TODO clean and clarify all this code, explaining:
    # - the arguments of the different functions
    # - univariate and multivariate as consequence of coords_as_input
#TODO fix the INPUT_VARIABLES list

def get_wake_dataloaders(data_filepath: str,
                         consider_ws: bool,
                         coords_as_input: bool,
                         train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0,
                         scaler=MinMaxScaler(), #TODO try also StandardScaler or choose a scaler taking ranges
                         batch_size: int | None = None,
                         batch_multiplier: int | None = None
                         ) -> tuple[DataLoader, DataLoader] | tuple[DataLoader, DataLoader, DataLoader]:
    if batch_size is not None and batch_multiplier is not None:
        raise ValueError("Cannot specify both batch_size and batch_multiplier.")
    elif coords_as_input and batch_multiplier is None:
        raise ValueError("batch_multiplier must be specified in case of univariate setting.")
    elif not coords_as_input and batch_size is None:
        raise ValueError("batch_size must be specified in case of multivariate setting.")

    dataframes = __load_and_split_data(data_filepath, consider_ws, train_perc, test_perc, validation_perc)
    datasets = __dataframes_to_datasets(dataframes, coords_as_input, scaler)
     # in univariate, batch_size is a multiplier
    if coords_as_input:
        batch_size = batch_multiplier * datasets[0].num_cells
    training_dataloader = DataLoader(datasets[0], batch_size, shuffle=True)
    if len(datasets) > 2:
        validation_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
        test_dataloader = DataLoader(datasets[2], batch_size, shuffle=False)
        return training_dataloader, validation_dataloader, test_dataloader
    test_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
    return training_dataloader, test_dataloader

def get_wake_datasets(data_filepath: str,
                           consider_ws: bool,
                           coords_as_input: bool,
                           train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0,
                           scaler=MinMaxScaler()):
    dataframes = __load_and_split_data(data_filepath, consider_ws, train_perc, test_perc, validation_perc)
    datasets = __dataframes_to_datasets(dataframes, coords_as_input, scaler)
    train_x, train_y = datasets[0].x, datasets[0].y
    if len(dataframes) > 2:
        valid_x, valid_y = datasets[1].x, datasets[1].y
        test_x, test_y = datasets[2].x, datasets[2].y
        return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)
    test_x, test_y = datasets[1].x, datasets[1].y
    return (train_x, train_y), (test_x, test_y)

def __load_and_split_data(data_folder: str, consider_ws: bool,
        train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0) \
            -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """In this moment, the method loads the data either including wind speed or not:
    - if yes, the data from different wind speeds is loaded and wind speed becomes an input feature;
        (in this case, the train-test split is done on values of wind speed randomly)
    - otherwise, a random wind speed is loaded and the input features are only the standard ones
        (in this case, the train-test split is done on the other input features)
    #TODO in the future, more combinations of train-test splits could be implemented

    This method returns 2 or 3 dataframes representing respectively: train, (validation) and test sets
    """
    if consider_ws:
        if WS not in INPUT_VARIABLES:
            INPUT_VARIABLES.append(WS)
        return __load_and_split_data_by_speed(data_folder, test_perc, validation_perc)
    else:
        if WS in INPUT_VARIABLES:
            INPUT_VARIABLES.remove(WS)
        # random ws in case ws is not important
        return __load_and_split_data_by_input_params(data_folder, train_perc, test_perc, validation_perc)

def __load_and_split_data_by_speed(
        data_folder: str,
        test_perc: float = 0.2, valid_perc: float = 0) \
            -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data in training, test and possibly validation sets
    according to the wind speed.
    """
    assert os.path.isdir(data_folder), "You need to pass a folder to load all the files in it"
    files = os.listdir(data_folder)
    assert len(files) > 0, "No files in this directory"
    assert all ([file.endswith(".nc") and file.startswith("ws_") for file in files]),\
         "All the files in the specified folder should be nc for xarray dataset of a specific wind speed"
    wind_speeds = [int(file.split(".nc")[0].split("_")[1]) for file in files]

    ws_split_lists = utils.random_split_list(wind_speeds, test_perc, valid_perc) #TODO

    train_df = __build_set_for_different_ws(data_folder, ws_split_lists[0])
    if len(ws_split_lists) > 2: #also validation
        validation_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
        test_df = __build_set_for_different_ws(data_folder, ws_split_lists[2])
        return train_df, validation_df, test_df
    test_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
    return train_df, test_df

def __build_set_for_different_ws(data_folder: str, wind_speeds: list[int]) -> pd.DataFrame:
    # by default, in case of more speeds, only a subpart of TI-CT combinations are considered
    return pd.concat([data_utils.load_netcfd(data_folder, wind_speed, include_ws_column=True,
                                             input_var_to_reduction_factor={'ti': 4, 'ct': 4})\
                      for wind_speed in wind_speeds])

def __load_and_split_data_by_input_params(
        data_folder: str,
        train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0)\
             -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data in training, test and possibly validation sets
    according to the input variables.
    """
    df = data_utils.load_netcfd(data_folder, wind_speed=12, include_ws_column=True) #TODO random value here
    input_combs = df[INPUT_VARIABLES].drop_duplicates()

    # Sampling from input combinations
    train_size = int(len(input_combs) * train_perc)
    validation_size = int(len(input_combs) * validation_perc)
    test_size = int(len(input_combs) * test_perc)
    assert train_size + validation_size + test_size == len(input_combs),\
        "Split percentages should sum to 1"
    input_combs = input_combs.sample(frac=1, random_state=1)
    train_df = pd.merge(input_combs[:train_size], df,\
                            on=INPUT_VARIABLES, how='inner')
    validation_df = pd.merge(input_combs[train_size:train_size+validation_size], df,\
                            on=INPUT_VARIABLES, how='inner')
    test_df = pd.merge(input_combs[train_size+validation_size:], df,\
                            on=INPUT_VARIABLES, how='inner')

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    if len(validation_df) > 0:
        return train_df, validation_df, test_df
    return train_df, test_df

def __dataframes_to_datasets(dfs,
                           coords_as_input: bool,
                           scaler):
    return [WakeDataset(df, coords_as_input, scaler) for df in dfs]

class WakeDataset(Dataset):

    def __init__(
            self, df: pd.DataFrame,
            coords_as_input: bool,
            scaler = MinMaxScaler()
        ) -> None:
        super().__init__()
        self.__df = df
        assert set(INPUT_VARIABLES) <= set(self.__df.columns) and \
            {OUTPUT_VARIABLE} <= set(self.__df.columns)
        self.__scaler = scaler

        if coords_as_input:
            self.__prepare_univariate()
        else:
            self.__prepare_multivariate()
        
        self.num_cells = df[COORDS_VARIABLES].drop_duplicates().shape[0]
        
        assert len(self.x) == len(self.y)
    
    def __prepare_univariate(self) -> None:
        #TODO change name and/or explain this approach

        self.unscaled_x = self.__df[INPUT_VARIABLES + COORDS_VARIABLES].values
        self.x = torch.FloatTensor(self.__scaler.fit_transform(self.unscaled_x))
        #TODO this scaler must be used to fit (and only to fit?) the test set
        self.y = self.__df[OUTPUT_VARIABLE].values
        self.y = torch.FloatTensor(self.y).unsqueeze(1)

    def __prepare_multivariate(self) -> None:
        #TODO change name and/or explain this approach

        # Group by input features and create input and output tensors
        inputs = list()
        outputs = list()

        for group, data in self.__df.groupby(INPUT_VARIABLES):
            input_tensor = torch.FloatTensor(group)
            inputs.append(input_tensor)

            output_tensor = data.pivot(index='y/D', columns='x/D',
                                       values=OUTPUT_VARIABLE).values # 2d output tensor with shape: (num_unique_y_values, num_unique_x_values)
            output_tensor = output_tensor.reshape(-1) #1d output tensor
            output_tensor = torch.FloatTensor(output_tensor)
            outputs.append(output_tensor)

        self.x = torch.stack(inputs, dim=0)
        if WS in INPUT_VARIABLES:
            # scaling only if wind speed is included, otherwise ct and ti have the same ranges
            self.unscaled_x = self.x
            self.x = torch.FloatTensor(self.__scaler.fit_transform(self.x))
        self.y = torch.stack(outputs)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
