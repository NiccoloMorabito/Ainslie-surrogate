import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

""" OLD METHOD TO LOAD DATA (FOR MULTIVARIATE REGRESSION)"""

class MultivariateWakeDataset(Dataset):
    def __init__(self, input_tensors_filepath, output_tensors_filepath):
        self.input_tensors = torch.load(input_tensors_filepath)
        self.output_tensors = torch.load(output_tensors_filepath)
        
        #TODO scale variables (here or somewhere else)
        print(len(self.input_tensors), len(self.output_tensors))
        assert len(self.input_tensors) == len(self.output_tensors)

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]
    
    def get_input_shape(self):
        return self.input_tensors[0].shape[0]

    def get_output_shape(self):
        return self.output_tensors[0].shape[0]
    
def get_multivariate_dataloader(input_filepath: str, output_filepath: str, batch_size: int):
    ds = MultivariateWakeDataset(input_filepath, output_filepath)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)




class UnivariateWakeDataset(Dataset):
    def __init__(self, dataframe_csv: str, output_space: int, scaler = MinMaxScaler()):
        #TODO MinMaxScaler or StandardScaler?
        self._dataframe = pd.read_csv(dataframe_csv)
        self.x = self._dataframe.iloc[:, :-output_space]
        self.x = torch.FloatTensor(scaler.fit_transform(self.x))
        self.y = self._dataframe.iloc[:, -output_space:]
        self.y = torch.FloatTensor(self.y.values)
        assert len(self.x) == len(self.y)
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def get_univariate_dataloader(dataframe_csv: str, output_space: int, batch_size: int):
    ds = UnivariateWakeDataset(dataframe_csv, output_space)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)