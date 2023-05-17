import matplotlib.pyplot as plt
import numpy as np

'''
#TODO DELETE
def plot_deficit_map(fm, D, cmap='Blues', levels=np.linspace(0, 10, 55)):
    fm.plot(fm.WS_eff, clabel='Deficit [m/s]', levels=levels, cmap=cmap, normalize_with=D)
    setup_plot(grid=False, ylabel="Crosswind distance [y/D]", xlabel= "Downwind distance [x/D]",
               xlim=[fm.x.min()/D, fm.x.max()/D], ylim=[fm.y.min()/D, fm.y.max()/D], axis='auto')
    
def plot_wake_map(flow_map, levels=100, cmap=None, plot_colorbar=True, plot_windturbines=True,
                      normalize_with=1, ax=None):
    print(flow_map.P.dims)
    sum_dims = [d for d in ['wd', 'time', 'ws'] if d in flow_map.P.dims]
    print(sum_dims)
    WS_eff = (flow_map.WS_eff * flow_map.P / flow_map.P.sum(sum_dims))
    print(WS_eff)
    WS_eff = WS_eff.sum(sum_dims)
    print(WS_eff)

    return flow_map.plot(WS_eff, clabel='wind speed [m/s]',
                        levels=levels, cmap=cmap, plot_colorbar=plot_colorbar,
                         plot_windturbines=plot_windturbines, normalize_with=normalize_with, ax=ax)
'''

def plot_map(df, z_name: str, clabel: str, levels=500, cmap='Blues_r'):
    # TODO missing things to plot (in case I need these plots for the thesis):
    # - the color should get darker where the deficit is higher
    # - different color between wake_map and deficit_map
    # - start with the x also including the near-wake region, just without plotting anything there
    # - plot the turbine (see plot_windturbines() method in py_wake.flow_map.py)
    X, Y = np.meshgrid(df["x/D"].unique(), df["y/D"].unique())
    Z = df.pivot(index='y/D', columns='x/D', values=z_name).values

    # Plot WS_eff as a contour plot
    ax = plt.gca()
    ax.set_xlabel("Downwind distance [x/D]")
    ax.set_ylabel("Crosswind distance [y/D]")
    c = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    plt.colorbar(c, label=clabel, ax=ax)
    plt.title("TODO title")
    plt.show()

def plot_deficit_map(df, levels=500, cmap="Blues_r"):
    plot_map(df, z_name="wind_deficit", clabel="Wind deficit", levels=levels, cmap=cmap)

def plot_wake_map(df, levels=500, cmap="Blues_r"):
    plot_map(df, z_name="WS_eff", clabel="Effective wind speed [m/s]", levels=levels, cmap=cmap)