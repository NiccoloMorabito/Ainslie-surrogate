import os
import pandas as pd
import xarray as xr

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
                   x_scaled_range: tuple[int, int] | None = None,
                   y_scaled_range: tuple[int, int] | None = None,
                   input_var_to_reduction_factor: dict | None = None,
                   resolution_reduction_factor: int | None = None) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.nc")
    ds = xr.open_dataset(filepath)

    if x_scaled_range:
        if len(x_scaled_range) != 2 or x_scaled_range[0] >= x_scaled_range[1]:
            raise ValueError("The range of x/D is not valid")
        ds = ds.where((ds['x:D'] >= x_scaled_range[0]) & (ds['x:D'] <= x_scaled_range[1]), drop=True)
    if y_scaled_range:
        if len(y_scaled_range) != 2 or y_scaled_range[0] >= y_scaled_range[1]:
            raise ValueError("The range of y/D is not valid")
        ds = ds.where((ds['y:D'] >= y_scaled_range[0]) & (ds['y:D'] <= y_scaled_range[1]), drop=True)
    
    if input_var_to_reduction_factor:
        for input_var, skip_factor in input_var_to_reduction_factor.items():
            ds = ds.isel({input_var:slice(0, ds[input_var].size, skip_factor)})
    
    if resolution_reduction_factor:
        ds = ds.isel({'x:D':slice(0, ds['x:D'].size*2, resolution_reduction_factor)})\
                .isel({'y:D':slice(0, ds['y:D'].size*2, resolution_reduction_factor)})

    df = ds.to_dataframe()\
            .reset_index()\
            .rename(columns={"x:D": "x/D", "y:D": "y/D"})
    
    if include_ws_column:
        return df.rename(columns={"WS": "ws"}) #TODO
    return df.drop("WS", axis=1)


if __name__=='__main__':
    folder = "data/discr_factors_x2_50_y-1_1_step0.125_TIstep0.01_CTstep0.01"
    reduced_df = load_netcfd(folder, 12,
                    x_scaled_range=(2, 30), y_scaled_range=(-1, 1),
                    input_var_to_reduction_factor={'ti': 10, 'ct': 10}, #10 times less values of ti and ct
                    resolution_reduction_factor=2) #grid resolution halved
