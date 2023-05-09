import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

class WakeDataset(Dataset):

    def __init__(
            self, data_filepath: str,
            input_variables: list[str],
            output_variables: list[str],
            scaler = MinMaxScaler() #TODO MinMaxScaler or StandardScaler?
        ) -> None:
        super().__init__()

        self.__df = pd.read_csv(data_filepath)

        assert set(input_variables) <= set(self.__df.columns) and \
            set(output_variables) <= set(self.__df.columns) and \
            set(input_variables).isdisjoint(output_variables)
        self.__input_vars = input_variables
        self.__output_vars = output_variables

        self.__scaler = scaler

        if set(["x", "y"]) <= set(input_variables):
            self.__prepare_univariate()
        else:
            self.__prepare_multivariate()
        
        assert len(self.x) == len(self.y)
    
    def __prepare_univariate(self):
        #TODO change name and/or explain this approach

        self.x = self.__df[self.__input_vars]
        self.x = torch.FloatTensor(self.__scaler.fit_transform(self.x))
        self.y = self.__df[self.__output_vars]
        self.y = torch.FloatTensor(self.y.values)

    def __prepare_multivariate(self):
        #TODO change name and/or explain this approach

        warnings.warn("Currently, only one output feature (WS_eff) is considered in multivariate setting, the others are ignored") #TODO
        # Group by input features and create input and output tensors
        inputs = list()
        outputs = list()

        for group, data in self.__df.groupby(self.__input_vars):
            input_tensor = torch.FloatTensor(group)
            inputs.append(input_tensor)

            #TODO so far only one output variable (WS_eff)
            output_tensor = data.pivot(index='x', columns='y', values='WS_eff').values # 2d output tensor
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

def get_wake_dataloader(data_filepath: str, input_variables: list[str], output_variables: list[str],
                        scaler: MinMaxScaler(), #TODO
                        batch_size: int):
    ds = WakeDataset(data_filepath, input_variables, output_variables, scaler)
    return DataLoader(ds, batch_size=batch_size) #, shuffle=True)
