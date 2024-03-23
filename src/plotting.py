from typing import Optional
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from py_wake.wind_turbines import WindTurbine, WindTurbines

DEFAULT_XLABEL = "Downwind distance [x/D]"
DEFAULT_YLABEL = "Crosswind distance [y/D]"

DEFICIT_LEVELS = [
    0,
    0.0025,
    0.005,
    0.0075,
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
    0.035,
    0.040,
    0.045,
    0.050,
    0.055,
]
PCT_LEVELS_ABS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
PCT_LEVELS_REL = np.append(-np.flip(PCT_LEVELS_ABS), np.insert(PCT_LEVELS_ABS, 0, 0))

MIN_X = -1
MAX_X = 29.875
MIN_Y = -1.875
MAX_Y = 1.875

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TURBINE_SYMBOL_PATH = os.path.join(BASE_DIR, "src/turbine.png")

plt.rcParams["font.size"] = 11


def plot_ct_curve(turbines: list[WindTurbine]) -> None:
    plt.xlabel("Wind speed [m/s]")
    plt.ylabel("CT [-]")
    wts = WindTurbines.from_WindTurbine_lst(turbines)
    ws = np.arange(0, 30)
    for t in wts.types():
        plt.plot(ws, wts.ct(ws, type=t), ".-", label=wts.name(t))
    plt.legend(loc=1)
    plt.show()


def plot_deficit_map_from_df(
    df: pd.DataFrame,
    levels=DEFICIT_LEVELS,
    cmap: str = plt.cm.Blues,
    add_near_wake: bool = False,
    plot_wind_turbine: bool = False,
) -> None:
    plot_map_from_df(
        df,
        zname="wind_deficit",
        zlabel="Wind Deficit",
        levels=levels,
        cmap=cmap,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def plot_wake_map_from_df(
    df: pd.DataFrame,
    levels: int = 500,
    cmap: str = plt.cm.Greens,
    add_near_wake: bool = False,
    plot_wind_turbine: bool = False,
) -> None:
    plot_map_from_df(
        df,
        zname="WS_eff",
        zlabel="Effective wind speed [m/s]",
        levels=levels,
        cmap=cmap,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def plot_map_from_df(
    df: pd.DataFrame,
    zname: str,
    zlabel: str,
    levels,
    cmap: str = plt.cm.Blues,
    add_near_wake: bool = False,
    plot_wind_turbine: bool = False,
) -> None:
    assert (
        df.ti.nunique() == 1 and df.ct.nunique() == 1 and df.WS.nunique() == 1
    ), "The input dataframe should only contain one value of CT, TI and wind speed"
    ti, ct, ws = df[["ti", "ct", "WS"]].values[0]

    X, Y = np.meshgrid(df["x/D"].unique(), df["y/D"].unique())
    Z = df.pivot(index="y/D", columns="x/D", values=zname).values
    title = f"Contour map of {zlabel} for TI={ti:.2f}, CT={ct:.2f}"
    if ws is not None:
        title += f", WS={ws}"
    __plot_contour(
        X,
        Y,
        Z,
        xlabel=DEFAULT_XLABEL,
        ylabel=DEFAULT_YLABEL,
        zlabel=zlabel,
        title=title,
        levels=levels,
        cmap=cmap,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def plot_map(
    X,
    Y,
    wake_field,
    ti: float,
    ct: float,
    ws: Optional[int] = None,
    zlabel: str = "Wind Deficit",
    levels=DEFICIT_LEVELS,
    cmap: str = plt.cm.Blues,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
) -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    if wake_field.dim() == 1:
        wake_field = wake_field.reshape(X.shape)
    title = f"Contour map of {zlabel} for TI={ti:.2f}, CT={ct:.2f}"
    if ws is not None:
        title += f", WS={ws}"
    __plot_contour(
        X,
        Y,
        wake_field,
        DEFAULT_XLABEL,
        DEFAULT_YLABEL,
        zlabel,
        title=title,
        levels=levels,
        cmap=cmap,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def plot_maps(
    X,
    Y,
    original,
    predicted,
    ti: float,
    ct: float,
    ws: Optional[int] = None,
    zlabel: str = "Wind Deficit",
    error_to_plot: Optional[str] = None,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
    save_path: Optional[str] = None,
) -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    assert (
        original.shape == predicted.shape
    ), "Original and predicted do not have the same shape"

    if error_to_plot is None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    else:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    plot_submap(
        X,
        Y,
        original,
        zlabel=f"Actual {zlabel}",
        levels=DEFICIT_LEVELS,
        ax=axs[0],
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )
    plot_submap(
        X,
        Y,
        predicted,
        zlabel=f"Predicted {zlabel}",
        levels=DEFICIT_LEVELS,
        ax=axs[1],
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )
    if error_to_plot is not None:
        plot_error_submap(
            X,
            Y,
            original,
            predicted,
            error_to_plot,
            ax=axs[2],
            add_near_wake=add_near_wake,
            plot_wind_turbine=plot_wind_turbine,
        )

    suptitle = f"Wind Deficit Contour Maps for TI={ti:.2f}, CT={ct:.2f}"
    if ws is not None:
        suptitle += f", WS={ws}"
    fig.suptitle(suptitle, fontsize=16)  # Main title for all the images
    fig.tight_layout()  # Adjust the spacing between subplots
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def save_cut_maps(
    X,
    Y,
    original,
    predicted,
    ti: float,
    ct: float,
    ws: Optional[int] = None,
    zlabel: str = "Wind Deficit",
    error_to_plot: Optional[str] = None,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
    filepath: Optional[str] = None,
) -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    assert (
        original.shape == predicted.shape
    ), "Original and predicted do not have the same shape"

    filepath = filepath or f"TI{ti:.2f}_CT{ct:.2f}{f'_WS{str(ws)}' if ws else ''}.pdf"
    max_deficit = max(original.max(), predicted.max())
    levels = np.linspace(0, max_deficit, 5000)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    plot_submap(
        X,
        Y,
        predicted,
        zlabel=f"Predicted {zlabel}",
        levels=levels,
        ax=axs[0],
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )
    if error_to_plot is not None:
        plot_error_submap(
            X,
            Y,
            original,
            predicted,
            error_to_plot,
            ax=axs[1],
            add_near_wake=add_near_wake,
            plot_wind_turbine=plot_wind_turbine,
        )

    fig.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close()


def plot_submap(
    X,
    Y,
    wake_field,
    zlabel: str = "Wind Deficit",
    levels=DEFICIT_LEVELS,
    cmap: str = plt.cm.Blues,
    ax=None,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
) -> None:
    assert X.shape == Y.shape, "X and Y grids have not the same shape"
    if wake_field.dim() == 1:
        wake_field = wake_field.reshape(X.shape)
    __plot_contour(
        X,
        Y,
        wake_field,
        DEFAULT_XLABEL,
        DEFAULT_YLABEL,
        zlabel,
        title=zlabel,
        levels=levels,
        cmap=cmap,
        ax=ax,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def plot_error_submap(
    X,
    Y,
    original_wake_field,
    predicted_wake_field,
    error_to_plot: str,
    ax=None,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
) -> None:

    if error_to_plot.lower() == "signed":
        diff_wake_field = original_wake_field - predicted_wake_field
        cmap = plt.cm.coolwarm
        levels = np.linspace(
            torch.min(diff_wake_field), torch.max(diff_wake_field), 5000
        )
    elif error_to_plot.lower() == "absolute":
        diff_wake_field = np.abs(original_wake_field - predicted_wake_field)
        cmap = plt.cm.Reds
        levels = np.linspace(0, torch.max(diff_wake_field), 5000)
    elif error_to_plot.lower() == "relative":
        epsilon = 1e-10
        diff_wake_field = np.abs(original_wake_field - predicted_wake_field) / np.abs(
            predicted_wake_field + epsilon
        )
        cmap = plt.cm.Reds
        levels = np.linspace(0, torch.max(diff_wake_field), 5000)
    elif error_to_plot.lower() == "absolute percentage":
        diff_wake_field = 100 * torch.abs(
            predicted_wake_field / original_wake_field - 1
        )
        cmap = plt.cm.Reds
        levels = PCT_LEVELS_ABS
    elif error_to_plot.lower() == "signed percentage":
        diff_wake_field = 100 * predicted_wake_field / original_wake_field - 1
        cmap = plt.cm.PiYG
        levels = PCT_LEVELS_REL
    else:
        raise ValueError(
            f"Invalid type_of_error: {error_to_plot}. "
            + "Expected one of the following: ['signed', 'absolute', 'relative', 'absolute percentage', 'signed percentage']"
        )
    zlabel = f"{error_to_plot.capitalize()} Deficit Error"
    if torch.max(diff_wake_field) == 0.0:
        levels = np.linspace(0, 1e-2, 100)
        zlabel += " (zero)"
    plot_submap(
        X,
        Y,
        diff_wake_field,
        zlabel,
        levels,
        cmap,
        ax,
        add_near_wake=add_near_wake,
        plot_wind_turbine=plot_wind_turbine,
    )


def __plot_contour(
    X,
    Y,
    Z,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    levels,
    cmap: str,
    ax=None,
    add_near_wake: bool = True,
    plot_wind_turbine: bool = True,
) -> None:
    if not add_near_wake and plot_wind_turbine:
        raise ValueError("Cannot plot wind turbine without adding near-wake region.")

    show = False
    if ax is None:
        ax = plt.gca()
        show = True

    if add_near_wake:
        # changing the space to plot also the near-wake region
        additional_X_row = np.full((1, X.shape[1]), MIN_X)
        X = np.vstack((additional_X_row, X))
        Y = np.vstack((Y, Y[0]))
        additional_Z_row = np.full((1, Z.shape[1]), np.nan)
        Z = np.vstack((additional_Z_row, Z))

    if plot_wind_turbine:
        __plot_standard_wind_turbine(ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    c = ax.contourf(
        X,
        Y,
        Z,
        levels=levels,
        cmap=cmap,
        cbar_kwargs={"label": None, "spacing": "proportional"},
    )
    ax.set_xlim([MIN_X, MAX_X])
    ax.set_ylim([MIN_Y, MAX_Y])

    plt.colorbar(c, label=zlabel, ax=ax)

    if show:
        plt.show()


def __plot_standard_wind_turbine(ax):
    image = mpimg.imread(TURBINE_SYMBOL_PATH)
    image = np.rot90(image, k=1)
    ax.imshow(image, extent=[-0.5, 0.5, -0.5, 0.5], aspect="auto", zorder=10)
