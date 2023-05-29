import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from py_wake.wind_turbines import WindTurbine, WindTurbines

DEFAULT_XLABEL = "Downwind distance [x/D]"
DEFAULT_YLABEL = "Crosswind distance [y/D]"

DEFICIT_LEVELS = np.linspace(0, 1, 100) # default levels for deficit, as the range is [0,1)

def plot_ct_curve(turbines: list[WindTurbine]) -> None:
    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('CT [-]')
    wts = WindTurbines.from_WindTurbine_lst(turbines)
    ws = np.arange(0,30)
    for t in wts.types():
        plt.plot(ws, wts.ct(ws, type=t),'.-', label=wts.name(t))
    plt.legend(loc=1)
    plt.show()

# CONTOUR PLOTS
# TODO write more meaninfgul default values for levels (i.e. the number of values of z that are shown)
# for instance doing levels = np.linspace(-np.max(tensor), np.max(tensor), 100)
# MORE LIKELY, choose a range of deficit (for instance) that is always valid and use that,
# so that the plots can be better comparable
# TODO think of using the log scale

# TODO missing things to make the plot more readable (in case I need these plots for the thesis):
    # - start with the x also including the near-wake region, just without plotting anything there
    # - plot the turbine (see plot_windturbines() method in py_wake.flow_map.py)

def plot_deficit_map_from_df(df: pd.DataFrame, levels = DEFICIT_LEVELS, cmap: str ="Blues") -> None:
    plot_map_from_df(df, zname="wind_deficit", zlabel="Wind deficit", levels=levels, cmap=cmap)

def plot_wake_map_from_df(df: pd.DataFrame, levels: int = 500, cmap: str ="Greens") -> None:
    plot_map_from_df(df, zname="WS_eff", zlabel="Effective wind speed [m/s]", levels=levels, cmap=cmap)

def plot_map_from_df(df: pd.DataFrame, zname: str, zlabel: str, levels, cmap: str = "Blues") -> None:
    assert df.ti.nunique() == 1 and df.ct.nunique() == 1 and df.WS.nunique() == 1, \
        "The input dataframe should only contain one value of CT, TI and wind speed"
    ti, ct, ws= df[["ti", "ct", "WS"]].values[0]

    X, Y = np.meshgrid(df["x/D"].unique(), df["y/D"].unique())
    Z = df.pivot(index='y/D', columns='x/D', values=zname).values
    title = f"Contour map of {zlabel} for TI={ti:.2f}, CT={ct:.2f}"
    if ws is not None:
        title += f", WS={ws}"
    __plot_contour(X, Y, Z,
                   xlabel=DEFAULT_XLABEL,
                   ylabel=DEFAULT_YLABEL,
                   zlabel=zlabel,
                   title=title,
                   levels=levels, cmap=cmap)

def plot_map(X, Y, wake_field,
             ti: float, ct: float, ws: int | None = None, zlabel: str = "Wind deficit",
             levels = DEFICIT_LEVELS, cmap: str ='Blues') -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    if wake_field.dim() == 1:
        wake_field = wake_field.reshape(X.shape)
    title = f"Contour map of {zlabel} for TI={ti:.2f}, CT={ct:.2f}"
    if ws is not None:
        title += f", WS={ws}"
    __plot_contour(X, Y, wake_field,
                   DEFAULT_XLABEL, DEFAULT_YLABEL, zlabel,
                   title=title,
                   levels=levels, cmap=cmap)

def plot_maps(X, Y, original, predicted,
              ti: float, ct: float, ws: int | None = None,
              zlabel: str = "Wind deficit", error_to_plot: str | None = None) -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    assert original.shape == predicted.shape, "Original and predicted do not have the same shape"
    max_deficit = max(original.max(), predicted.max())
    levels = np.linspace(0, max_deficit, 100)
    plot_map(X, Y, original, ti, ct, ws, zlabel=f"Actual {zlabel}", levels=levels)
    plot_map(X, Y, predicted, ti, ct, ws, zlabel=f"Predicted {zlabel}", levels=levels)
    if error_to_plot is not None:
        plot_error_map(X, Y, original, predicted, error_to_plot, ti=ti, ct=ct, ws=ws)

def plot_error_map(X, Y, original_wake_field, predicted_wake_field,
                   error_to_plot: str, ti: float, ct: float, ws: int | None = None) -> None:
    #TODO standardize the levels for a better comparison?
    #TODO standardize torch and numpy use
    if error_to_plot=='signed':
        diff_wake_field = original_wake_field - predicted_wake_field
        cmap = 'coolwarm'
        levels = np.linspace(-torch.max(diff_wake_field), torch.max(diff_wake_field), 100) #TODO
    elif error_to_plot=='absolute':
        diff_wake_field = np.abs(original_wake_field - predicted_wake_field)
        cmap = 'Reds'
        levels = np.linspace(0, torch.max(diff_wake_field), 100)
    elif error_to_plot=='relative':
        epsilon = 1e-10
        diff_wake_field = np.abs(original_wake_field - predicted_wake_field) /\
             np.abs(predicted_wake_field + epsilon)
        cmap = 'Reds'
        levels = np.linspace(0, torch.max(diff_wake_field), 100)
    else:
        raise ValueError(f"Invalid type_of_error: {error_to_plot}. " +\
                         "Expected 'signed', 'absolute', or 'relative'.")
    zlabel = f"{error_to_plot} deficit error"
    plot_map(X, Y, diff_wake_field, ti, ct, ws, zlabel, levels, cmap)

def __plot_contour(X, Y, Z,
                   xlabel: str, ylabel: str, zlabel: str, title: str, levels, cmap: str) -> None:
    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    c = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(c, label=zlabel, ax=ax)
    plt.title(title)
    plt.show()
