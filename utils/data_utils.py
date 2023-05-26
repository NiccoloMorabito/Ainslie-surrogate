import os
import pandas as pd
import xarray as xr
import numpy as np
from .utils import my_arange

def get_dir_name(x_start_factor: int, x_end_factor: int,
                    y_start_factor: int, y_end_factor: int,
                    grid_step_factor: float, ti_step: float, ct_step: float) -> str:
    dir = f"data/discr_factors_x{x_start_factor}_{x_end_factor}_y{y_start_factor}_{y_end_factor}_"+\
        f"step{grid_step_factor}_TIstep{ti_step}_CTstep{ct_step}"
    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir.replace("data/", "saved_models/")) #TODO find a better way
    return dir

def get_filepath(x_start_factor: int, x_end_factor: int,
                    y_start_factor: int, y_end_factor: int,
                    grid_step_factor: float, wind_speed: int,
                    ti_step: float, ct_step: float) -> str: #TODO more explaining method name
    dir = get_dir_name(x_start_factor, x_end_factor, y_start_factor, y_end_factor,
                       grid_step_factor, ti_step, ct_step)
    filename = f"ws_{wind_speed}.nc"
    return os.path.join(dir, filename)

def get_parameters_from(dir_name: str) -> tuple[int, int, int, int, float, float, float]:
    import re
    def strip_letters(s: str) -> str:
        return re.sub(r"[^\d.-]+", '', s)

    factors = [strip_letters(f) for f in dir_name.strip("/").split("_")[2:]]
    x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_factor, ti_step, ct_step = \
        int(factors[0]), int(factors[1]), int(factors[2]), int(factors[3]), float(factors[4]), \
        float(factors[5]), float(factors[6])
    return x_start_factor, x_end_factor, y_start_factor, y_end_factor, grid_factor, ti_step, ct_step

def load_csv(data_folder: str, wind_speed: int, include_ws_column: bool = True) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.csv")
    df = pd.read_csv(filepath)
    if include_ws_column:
        df["ws"] = wind_speed
    return df

def load_netcfd(data_folder: str, wind_speed: int, include_ws_column: bool = True,
                   filter_condition: str | None = None,
                   input_var_to_reduced_step: dict | None = None,
                   resolution_reduction_factor: int | None = None) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.nc")
    df = xr.open_dataset(filepath)\
            .to_dataframe()\
            .reset_index()\
            .rename(columns={"x:D": "xD", "y:D": "yD"})
    
    if filter_condition:
        df = df.query(filter_condition.replace("/", ""))
    df.rename(columns={"xD": "x/D", "yD": "y/D"}, inplace=True)

    if resolution_reduction_factor:
        df = reduce_grid_resolution(df, resolution_reduction_factor, resolution_reduction_factor)
    
    if input_var_to_reduced_step:
        df = reduce_range_input_params(df, input_var_to_reduced_step)

    if include_ws_column:
        return df.rename(columns={"WS": "ws"}) #TODO
    return df.drop("WS", axis=1)

def reduce_grid_resolution(df: pd.DataFrame, x_factor: int, y_factor: int) -> pd.DataFrame:
    dataframes = list()
    # only reduce resolution for each combination of input variables
    for _, data in df.groupby(['ti', 'ct', 'WS']):
        vars_to_avg = ['wind_deficit', 'WS_eff']
        agg_dict = {**{col: 'first' for col in df.columns if col not in vars_to_avg},\
            **{col: 'mean' for col in vars_to_avg}}
        
        # combining rows for y/D
        data = data.sort_values(by=['y/D', 'x/D']).reset_index(drop=True)
        data['group'] = (data.index // y_factor) + 1
        data = data.groupby('group').agg(agg_dict)
        
        # combining rows for x/D
        data = data.sort_values(by=['x/D', 'y/D']).reset_index(drop=True)
        data['group'] = (data.index // x_factor) + 1
        data = data.groupby('group').agg(agg_dict)

        dataframes.append(data.reset_index(drop=True))
    return pd.concat(dataframes)

def reduce_range_input_params(df: pd.DataFrame, input_var_to_reduced_step: dict[str, float]) -> pd.DataFrame:
    conditions = [condition_for_reduced_step(df, column=var, desired_step=input_var_to_reduced_step[var])
                  for var in input_var_to_reduced_step.keys()]
    return df[np.logical_and.reduce(conditions)].reset_index(drop=True)

def condition_for_reduced_step(df: pd.DataFrame, column: str, desired_step: float) -> pd.Series:
    return df[column].isin(my_arange(df[column].min(), df[column].max(),
                                           desired_step, include_end=True))


if __name__=='__main__':
    folder = "data/discr_factors_x2_50_y-1_1_step0.125_TIstep0.01_CTstep0.01"
    reduced_df = load_netcfd(folder, 5,
                    filter_condition = "x/D > 10 and y/D > 0",
                   input_var_to_reduced_step = {'ti': 0.02, 'ct': 0.02},
                   resolution_reduction_factor = 2)