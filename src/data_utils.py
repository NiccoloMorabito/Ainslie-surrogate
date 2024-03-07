import os

import pandas as pd
import xarray as xr

TI = "ti"
CT = "ct"
X_NC = "x:D"
Y_NC = "y:D"
X = "x/D"
Y = "y/D"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_dir_name(
    x_start_factor: int,
    x_end_factor: int,
    y_start_factor: int,
    y_end_factor: int,
    grid_step_factor: float,
    ti_step: float,
    ct_step: float,
) -> str:
    return os.path.join(
        BASE_DIR,
        "data",
        f"discr_factors_x{x_start_factor}_{x_end_factor}_y{y_start_factor}_{y_end_factor}_step{grid_step_factor}_TIstep{ti_step}_CTstep{ct_step}",
    )


def get_filepath(
    x_start_factor: int,
    x_end_factor: int,
    y_start_factor: int,
    y_end_factor: int,
    grid_step_factor: float,
    wind_speed: int,
    ti_step: float,
    ct_step: float,
) -> str:  # TODO more explaining method name
    folder = get_dir_name(
        x_start_factor,
        x_end_factor,
        y_start_factor,
        y_end_factor,
        grid_step_factor,
        ti_step,
        ct_step,
    )
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder.replace("data/", "saved_models/"))  # TODO find a better way
    filename = f"ws_{wind_speed}.nc"
    return os.path.join(folder, filename)


def get_parameters_from(
    dir_name: str,
) -> tuple[int, int, int, int, float, float, float]:
    import re

    def strip_letters(s: str) -> str:
        return re.sub(r"[^\d.-]+", "", s)

    factors = [strip_letters(f) for f in dir_name.strip("/").split("_")[2:]]
    (
        x_start_factor,
        x_end_factor,
        y_start_factor,
        y_end_factor,
        grid_factor,
        ti_step,
        ct_step,
    ) = (
        int(factors[0]),
        int(factors[1]),
        int(factors[2]),
        int(factors[3]),
        float(factors[4]),
        float(factors[5]),
        float(factors[6]),
    )
    return (
        x_start_factor,
        x_end_factor,
        y_start_factor,
        y_end_factor,
        grid_factor,
        ti_step,
        ct_step,
    )


def get_TIstep_from(dir_name: str) -> float:
    return get_parameters_from(dir_name)[-2]


def get_CTstep_from(dir_name: str) -> float:
    return get_parameters_from(dir_name)[-1]


def load_csv(
    data_folder: str, wind_speed: int, include_ws_column: bool = True
) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.csv")
    df = pd.read_csv(filepath)
    if include_ws_column:
        df["ws"] = wind_speed
    return df


def load_netcfd(
    data_folder: str,
    wind_speed: int,
    include_ws_column: bool = True,
    x_scaled_range: tuple[int, int] | None = None,
    y_scaled_range: tuple[int, int] | None = None,
    ti_range: tuple[float, float] | None = None,
    ct_range: tuple[float, float] | None = None,
    input_var_to_reduction_factor: dict | None = None,
    resolution_reduction_factor: int | None = None,
) -> pd.DataFrame:
    filepath = os.path.join(data_folder, f"ws_{wind_speed}.nc")
    ds = xr.open_dataset(filepath)

    # ranges
    if x_scaled_range:
        if len(x_scaled_range) != 2 or x_scaled_range[0] >= x_scaled_range[1]:
            raise ValueError(f"The range {x_scaled_range} for x/D is not valid")
        ds = ds.where(
            (ds[X_NC] >= x_scaled_range[0]) & (ds[X_NC] <= x_scaled_range[1]), drop=True
        )
    if y_scaled_range:
        if len(y_scaled_range) != 2 or y_scaled_range[0] >= y_scaled_range[1]:
            raise ValueError(f"The range {y_scaled_range} for y/D is not valid")
        ds = ds.where(
            (ds[Y_NC] >= y_scaled_range[0]) & (ds[Y_NC] <= y_scaled_range[1]), drop=True
        )

    if ti_range:
        if len(ti_range) != 2 or ti_range[0] >= ti_range[1]:
            raise ValueError(f"The range {ti_range} for TI is not valid")
        ds = ds.where((ds[TI] >= ti_range[0]) & (ds[TI] <= ti_range[1]), drop=True)
    if ct_range:
        if len(ct_range) != 2 or ct_range[0] >= ct_range[1]:
            raise ValueError(f"The range {ct_range} for CT is not valid")
        ds = ds.where((ds[CT] >= ct_range[0]) & (ds[CT] <= ct_range[1]), drop=True)

    # reduction
    if input_var_to_reduction_factor:
        for input_var, skip_factor in input_var_to_reduction_factor.items():
            ds = ds.isel({input_var: slice(0, ds[input_var].size, skip_factor)})

    if resolution_reduction_factor:
        ds = ds.isel(
            {X_NC: slice(0, ds[X_NC].size * 2, resolution_reduction_factor)}
        ).isel({Y_NC: slice(0, ds[Y_NC].size * 2, resolution_reduction_factor)})

    df = ds.to_dataframe().reset_index().rename(columns={X_NC: X, Y_NC: Y})

    if include_ws_column:
        return df.rename(columns={"WS": "ws"})  # TODO
    return df.drop("WS", axis=1)


if __name__ == "__main__":
    folder = "data/discr_factors_x2_30_y-2_2_step0.125_TIstep0.01_CTstep0.01"
    reduced_df = load_netcfd(
        folder,
        11,
        x_scaled_range=(4, 30),
        y_scaled_range=(-1, 1),
        input_var_to_reduction_factor={
            TI: 4,
            CT: 4,
        },  # 4 times less values of ti and ct
        resolution_reduction_factor=2,
    )  # grid resolution halved
