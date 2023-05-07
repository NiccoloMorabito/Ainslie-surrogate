import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#TODO try to merge or standardize these two ways of loading data:
# - associate better names to them
# - scale in both
# - both of them have to accept input features and output features to be parametrizable




class MultivariateWakeDataset(Dataset):
    def __init__(self, data_filepath: str, input_variables: list[str]): #TODO scaler????
        #TODO MinMaxScaler or StandardScaler?
        self._dataframe = pd.read_csv(data_filepath)

        # Group by input features and create input and output tensors
        inputs = []
        outputs = []
        for group, data in self._dataframe.groupby(input_variables):
            inputs.append(torch.tensor(group, dtype=torch.float32))
            outputs.append(torch.tensor(data['WS_eff'].values, dtype=torch.float32))

        self.x = inputs
        self.y = outputs
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_multivariate_dataloader(data_filepath: str, input_variables: list[str], batch_size: int):
    ds = MultivariateWakeDataset(data_filepath, input_variables)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


class UnivariateWakeDataset(Dataset):
    def __init__(self, data_filepath: str, output_space: int, scaler = MinMaxScaler()):
        #TODO MinMaxScaler or StandardScaler?
        self._dataframe = pd.read_csv(data_filepath)

        self.x = self._dataframe.iloc[:, :-output_space]
        self.x = torch.FloatTensor(scaler.fit_transform(self.x))
        self.y = self._dataframe.iloc[:, -output_space:]
        self.y = torch.FloatTensor(self.y.values)
        assert len(self.x) == len(self.y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_univariate_dataloader(data_filepath: str, output_space: int, batch_size: int):
    ds = UnivariateWakeDataset(data_filepath, output_space)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)