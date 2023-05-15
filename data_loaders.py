import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# TODO these variables control also the architecture of the ML models
INPUT_VARIABLES = ["ti", "ct"]
OUTPUT_VARIABLES = ["wind_deficit"]
COORDS_VARIABLES = ["x/D", "y/D"]

assert set(INPUT_VARIABLES).isdisjoint(OUTPUT_VARIABLES),\
    "Wrong lists of input and output variables"

#TODO clean and clarify all this code
# especially this function and the arguments of the different functions

def load_and_split_data(
        data_filepath: str,
        train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0
    ):
    """
    Function to split the data in training, test and possibly validation
    """
    df = pd.read_csv(data_filepath)
    input_combs = df[INPUT_VARIABLES].drop_duplicates()

    # Sampling from input combinations
    train_size = int(len(input_combs) * train_perc)
    validation_size = int(len(input_combs) * validation_perc)
    test_size = int(len(input_combs) * test_perc)
    assert train_size + validation_size + test_size == len(input_combs), "Split percentages should sum to 1"
    input_combs = input_combs.sample(frac=1, random_state=1)
    train_df = pd.merge(    input_combs[:train_size], df,\
                            on=INPUT_VARIABLES, how='inner')
    validation_df = pd.merge(input_combs[train_size:train_size+validation_size], df,\
                            on=INPUT_VARIABLES, how='inner')
    test_df = pd.merge(     input_combs[train_size+validation_size:], df,\
                             on=INPUT_VARIABLES, how='inner')

    assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    if len(validation_df) > 0:
        return train_df, validation_df, test_df
    return train_df, test_df

class WakeDataset(Dataset):

    def __init__(
            self, df: pd.DataFrame,
            coords_as_input: bool,
            scaler = MinMaxScaler() #TODO MinMaxScaler or StandardScaler?
            #TODO must accept the diameter step at least (to standardize the grid)
        ) -> None:
        super().__init__()
        self.__df = df
        assert set(INPUT_VARIABLES) <= set(self.__df.columns) and \
            set(OUTPUT_VARIABLES) <= set(self.__df.columns)
        self.__scaler = scaler

        if coords_as_input:
            self.__prepare_univariate()
        else:
            self.__prepare_multivariate()
        
        assert len(self.x) == len(self.y)
    
    def __prepare_univariate(self):
        #TODO change name and/or explain this approach

        self.x = self.__df[INPUT_VARIABLES + ["x/D", "y/D"]]
        self.x = torch.FloatTensor(self.__scaler.fit_transform(self.x))
        #TODO this scaler ^^^^^^^ myst be used to fit (and only to fit?) the test set
        self.y = self.__df[OUTPUT_VARIABLES]
        self.y = torch.FloatTensor(self.y.values)

    def __prepare_multivariate(self):
        #TODO change name and/or explain this approach

        # Group by input features and create input and output tensors
        inputs = list()
        outputs = list()

        for group, data in self.__df.groupby(INPUT_VARIABLES):
            input_tensor = torch.FloatTensor(group)
            inputs.append(input_tensor)

            #TODO so far only one output variable (WS_eff)
            output_tensor = data.pivot(index='x/D', columns='y/D', values='wind_deficit').values # 2d output tensor
            #output_tensor = data['WS_eff'].values) #1d output tensor
            output_tensor = torch.FloatTensor(output_tensor)
            outputs.append(output_tensor)

        self.x = torch.stack(inputs, dim=0)
        self.x = torch.FloatTensor(self.__scaler.fit_transform(self.x))
        self.y = torch.stack(outputs, dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def dataframes_to_datasets(dfs,
                           coords_as_input: bool = False,
                           scaler=MinMaxScaler() #TODO MinMaxScaler or StandardScaler?
                           ):
    return [WakeDataset(df, coords_as_input, scaler) for df in dfs]

def get_wake_dataloaders(data_filepath: str,
                         coords_as_input: bool, #TODO create two methods for univariate and multivariate
                         train_perc: float = 0.8, test_perc: float = 0.2, validation_perc: float = 0,
                         scaler=MinMaxScaler(), #TODO MinMaxScaler or StandardScaler?
                         batch_size: int = 64
                         ):
    dataframes = load_and_split_data(data_filepath, train_perc, test_perc, validation_perc)
    datasets = dataframes_to_datasets(dataframes, coords_as_input, scaler)
    training_dataloader = DataLoader(datasets[0], batch_size, shuffle=True)
    if len(datasets) > 2:
        validation_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
        test_dataloader = DataLoader(datasets[2], batch_size, shuffle=False)
        return training_dataloader, validation_dataloader, test_dataloader
    test_dataloader = DataLoader(datasets[1], batch_size, shuffle=False)
    return training_dataloader, test_dataloader