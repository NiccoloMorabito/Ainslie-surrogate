import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import utils.utils as utils
import utils.data_utils as data_utils

INPUT_VARIABLES = ["ti", "ct"]
WS = "ws"
OUTPUT_VARIABLE = "wind_deficit"
COORDS_VARIABLES = ["x/D", "y/D"]

TI_CT_DEFAULT_REDUC_FACTOR = 4

#TODO clean and clarify all this code, explaining:
    # - the arguments of the different functions
    # - univariate and multivariate as consequence of coords_as_input
#TODO fix the INPUT_VARIABLES list

class WakeDataset(Dataset):
#TODO make "consider ws" something standard and clear (not "WS in INPUT_VARIABLES")
#TODO create two subclasses for univariate and multivariate:
# so that the different methods can be distinguished (preparation and get_params_for_plotting)
# do maybe a different indexing for univariate, at least an additional one doing:
#   self.unscaled_inputs[index:index+self.num_cells]
#TODO there is confusion between x and y (i.e. input and output of the model)
# and x and y as coordinates (in input) or the meshgrids X and Y
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
        self.num_cells = df[COORDS_VARIABLES].drop_duplicates().shape[0]

        if coords_as_input:
            self.__prepare_univariate()
        else:
            self.__prepare_multivariate()
                
        assert len(self.inputs) == len(self.outputs)
    
    def __prepare_univariate(self) -> None:
        #TODO change name and/or explain this approach
        
        # ordering to have a wake field ordered in the self.num_cells value by coords
        self.__input_vars_names = INPUT_VARIABLES + COORDS_VARIABLES
        self.__df = self.__df\
            .sort_values(by=self.__input_vars_names)
        self.unscaled_inputs = self.__df[INPUT_VARIABLES + COORDS_VARIABLES].values
        self.inputs = torch.FloatTensor(self.__scaler.fit_transform(self.unscaled_inputs))
        #TODO this scaler must be used to fit (and only to fit?) the test set
        # prob it's ok to fit_transform because I already know the ranges of the variables (?)
        self.outputs = self.__df[OUTPUT_VARIABLE].values
        self.outputs = torch.FloatTensor(self.outputs).unsqueeze(1)

        # meshgrids
        Xs, Ys = self.unscaled_inputs[0:self.num_cells, -2], self.unscaled_inputs[0:self.num_cells, -1]
        trasp_shape = len(np.unique(Xs)), len(np.unique(Ys))
        self.X_grid = Xs.reshape(trasp_shape)
        self.Y_grid = Ys.reshape(trasp_shape)

    def __prepare_multivariate(self) -> None:
        #TODO change name and/or explain this approach

        self.__input_vars_names = INPUT_VARIABLES

        # Group by input features and create input and output tensors
        inputs = list()
        outputs = list()
        for group, data in self.__df.groupby(self.__input_vars_names):
            input_tensor = torch.FloatTensor(group)
            inputs.append(input_tensor)

            output_tensor = data.pivot(index='y/D', columns='x/D',
                                       values=OUTPUT_VARIABLE).values # 2d output tensor with shape: (num_unique_y_values, num_unique_x_values)
            output_tensor = output_tensor.reshape(-1) #1d output tensor
            output_tensor = torch.FloatTensor(output_tensor)
            outputs.append(output_tensor)

            # meshgrids
            if not hasattr(self, 'X_grid') or not hasattr(self, 'Y_grid'):
                Xs = data[COORDS_VARIABLES[0]].values
                Ys = data[COORDS_VARIABLES[1]].values
                trasp_shape = len(np.unique(Xs)), len(np.unique(Ys))
                self.X_grid = Xs.reshape(trasp_shape).T
                self.Y_grid = Ys.reshape(trasp_shape).T

        self.inputs = torch.stack(inputs, dim=0)
        if WS in INPUT_VARIABLES:
            # scaling only if wind speed is included, otherwise ct and ti have the same ranges
            #TODO maybe I can scale by default
            self.unscaled_inputs = self.inputs
            self.inputs = torch.FloatTensor(self.__scaler.fit_transform(self.inputs))
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

    def get_parameters_for_plotting_univariate(self, model, field_idx: int,
                                               transformed_inputs = None) -> tuple:
        inputs, outputs = self[self.slice_for_field(field_idx)]
        unscaled_inputs = torch.Tensor(self.unscaled_inputs[self.slice_for_field(field_idx)])
        ti, ct = unscaled_inputs[0, 0].item(), unscaled_inputs[0, 1].item()
        ws = unscaled_inputs[0, 2].item() if WS in INPUT_VARIABLES else None

        if transformed_inputs is not None:
            inputs = transformed_inputs

        if callable(model): # PyTorch model
            predictions = model(inputs)
            # reshape outputs and predictions
            wake_field = outputs.view(-1).reshape(self.X_grid.shape)
            predicted_wake_field = predictions.view(-1).reshape(self.X_grid.shape)
        elif hasattr(model, 'predict'):  # sklearn model
            predictions = model.predict(inputs)
            wake_field = outputs.reshape(self.X_grid.shape)
            predicted_wake_field = torch.Tensor(predictions).reshape(self.X_grid.shape)
        else:
            raise ValueError("Invalid model type. Expected PyTorch model or sklearn model.")

        return ti, ct, ws, wake_field, predicted_wake_field

    def get_parameters_for_plotting_multivariate(self, model, idx: int) -> tuple:
        input, wake_field = self.inputs[idx], self.outputs[idx]
        if callable(model): # PyTorch model
            predicted_wake_field = model(input)
        elif hasattr(model, 'predict'):  # sklearn model
            predicted_wake_field = torch.tensor(model.predict(input.reshape(1, -1)).reshape(-1))
        else:
            raise ValueError("Invalid model type. Expected PyTorch model or sklearn model.")

        if WS in INPUT_VARIABLES:
            ti, ct, ws = self.unscaled_inputs[idx]
        else:
            ti, ct = input
            ws = None
        
        return ti, ct, ws, wake_field, predicted_wake_field

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

    if coords_as_input: #batch size conversion in univariate
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
                           scaler=MinMaxScaler()) -> list[WakeDataset]:
    dataframes = __load_and_split_data(data_filepath, consider_ws, train_perc, test_perc, validation_perc)
    return __dataframes_to_datasets(dataframes, coords_as_input, scaler)

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
        #TODO choose one of the following possibilities
        return __load_and_split_data_by_speed(data_folder, test_perc, validation_perc)
        #return __load_and_split_data_NEW(data_folder, test_perc, validation_perc)
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
    files = set([file for file in files if not file.startswith('.')])
    assert all ([file.endswith(".nc") and file.startswith("ws_") for file in files]),\
         "All the files in the specified folder should be nc for xarray dataset of a specific wind speed"
    wind_speeds = [int(file.split(".nc")[0].split("_")[1]) for file in files]

    ws_split_lists = utils.random_split_list(wind_speeds, test_perc, valid_perc)

    train_df = __build_set_for_different_ws(data_folder, ws_split_lists[0])
    if len(ws_split_lists) > 2: #also validation
        validation_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
        test_df = __build_set_for_different_ws(data_folder, ws_split_lists[2])
        return train_df, validation_df, test_df
    test_df = __build_set_for_different_ws(data_folder, ws_split_lists[1])
    return train_df, test_df

def __build_set_for_different_ws(data_folder: str, wind_speeds: list[int],
                                 tis: list[float] | None = None,
                                 cts: list[float] | None = None) -> pd.DataFrame: #TODO change name (still the previous one when no tis and cts were specified)
    ti_range = (min(tis), max(tis)) if tis else None
    ct_range = (min(cts), max(cts)) if cts else None

    data_frames = []
    for wind_speed in wind_speeds:
        try:
            # by default for more speed values, only a subpart of TI-CT combinations are considered
            df = data_utils.load_netcfd(data_folder, wind_speed, include_ws_column=True,
                                        input_var_to_reduction_factor={
                                            'ti': TI_CT_DEFAULT_REDUC_FACTOR,
                                            'ct': TI_CT_DEFAULT_REDUC_FACTOR},
                                            ti_range=ti_range, ct_range=ct_range)
            data_frames.append(df)
        except Exception as e: #TODO temporary code for non-working files
            print(f"Error loading data for wind speed {wind_speed}")
            
    result = pd.concat(data_frames)
    return result

def __load_and_split_data_by_input_params(
        data_folder: str,
        train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0)\
             -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data in training, test and possibly validation sets
    according to thrust coefficient and turbulence intensity
    """
    random_ws = 12 #TODO random value here
    df = data_utils.load_netcfd(data_folder, wind_speed=random_ws, include_ws_column=True)
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

#TODO fix and complete this
#TODO currently, (the data is not split according to the percentages and) it doesn't utilize all the data
# as for each set (training, test and valid) the corresponding percentage is taken for ws and then
# for TI and CT, therefore the other values of TI and CT are not taken at all for that particular set
def __load_and_split_data_NEW(
        data_folder: str,
        test_perc: float = 0.2, valid_perc: float = 0) \
            -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to split the data in training, test and possibly validation sets
    according to wind speed, thrust coefficient and turbulence intensity
    """
    assert os.path.isdir(data_folder), "You need to pass a folder to load all the files in it"
    files = os.listdir(data_folder)
    assert len(files) > 0, "No files in this directory"
    files = set([file for file in files if not file.startswith('.')])
    assert all ([file.endswith(".nc") and file.startswith("ws_") for file in files]),\
         "All the files in the specified folder should be nc for xarray dataset of a specific wind speed"
    wind_speeds = [int(file.split(".nc")[0].split("_")[1]) for file in files]
    ti_step, ct_step = data_utils.get_TIstep_from(data_folder), data_utils.get_CTstep_from(data_folder)
    tis = list(utils.my_arange(0, 1, ti_step * TI_CT_DEFAULT_REDUC_FACTOR))
    cts = list(utils.my_arange(0.1, 24/25, ct_step * TI_CT_DEFAULT_REDUC_FACTOR))

    ws_split_lists = utils.random_split_list(wind_speeds, test_perc, valid_perc)
    ti_split_lists = utils.ordered_split_list(tis, test_perc, valid_perc) #TODO random or ordered?
    ct_split_lists = utils.ordered_split_list(cts, test_perc, valid_perc) #TODO random or ordered?
    print(ws_split_lists, ti_split_lists, ct_split_lists)

    train_df = __build_set_for_different_ws(data_folder, ws_split_lists[0],
                                            ti_split_lists[0], ct_split_lists[0])
    #TODO train_df is missing ON PURPOSE other dataframes:
    # 1) in case of validation
    # __build_set_for_different_ws(data_folder, ws_split_lists[1], ti_split_lists[2], ct_split_lists[2]) and
    # __build_set_for_different_ws(data_folder, ws_split_lists[2], ti_split_lists[1], ct_split_lists[1])
    # or, 2) in case of no validation
    # __build_set_for_different_ws(data_folder, ws_split_lists[1], ti_split_lists[0], ct_split_lists[0])?????
    if len(ws_split_lists) > 2: #also validation
        validation_df1 = __build_set_for_different_ws(data_folder, ws_split_lists[1],
                                                     ti_split_lists[1], ct_split_lists[1])
        validation_df2 = __build_set_for_different_ws(data_folder, ws_split_lists[0],
                                                     ti_split_lists[2], ct_split_lists[2])
        # validation_df3 = __build_set_for_different_ws(data_folder, ws_split_lists[2],
        #                                             ti_split_lists[0], ct_split_lists[0]) excluded on purpose
        validation_df = pd.concat([validation_df1, validation_df2])#, validation_df3])

        test_df1 = __build_set_for_different_ws(data_folder, ws_split_lists[2],
                                               ti_split_lists[2], ct_split_lists[2])
        test_df2 = __build_set_for_different_ws(data_folder, ws_split_lists[0],
                                               ti_split_lists[1], ct_split_lists[1])
        #test_df3 = __build_set_for_different_ws(data_folder, ws_split_lists[1],
        #                                       ti_split_lists[0], ct_split_lists[0]) excluded on purpose
        test_df = pd.concat([test_df1, test_df2])#, test_df3])
        
        return train_df, validation_df, test_df
    test_df1 = __build_set_for_different_ws(data_folder, ws_split_lists[1],
                                           ti_split_lists[1], ct_split_lists[1])
    test_df2 = __build_set_for_different_ws(data_folder, ws_split_lists[0],
                                           ti_split_lists[1], ct_split_lists[1])
    test_df = pd.concat([test_df1, test_df2])
    return train_df, test_df

def __dataframes_to_datasets(dfs,
                           coords_as_input: bool,
                           scaler):
    return [WakeDataset(df, coords_as_input, scaler) for df in dfs]
