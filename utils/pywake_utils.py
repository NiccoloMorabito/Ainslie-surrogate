import numpy as np
import xarray as xr

from py_wake import HorizontalGrid

# turbines and sites
from py_wake.site import XRSite, UniformSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.utils.generic_power_ct_curves import standard_power_ct_curve

"""PyWake utils"""
def get_site(ti: float, ws: int) -> XRSite:
    #return XRSite(ds=xr.Dataset(data_vars={'WS': WS_RANGE, 'P': 1, 'TI': turbulence_intensity}, coords={'wd': WD_RANGE}))
    return UniformSite(ti=ti, ws=ws)

def get_wind_turbine(diameter: int, hub_height: int, power_norm: int,
                     constant_ct: float, ti: float) -> WindTurbine:
    # for power ct function (similar to GenericWindTurbine but putting a constant ct)
    wsp_lst = np.arange(.1, 30, .1) #TODO this parameter decides the number of elements in u, p and ct_lst
    u, p, ct_lst = standard_power_ct_curve(power_norm, diameter, turbulence_intensity=ti, 
                                       constant_ct=constant_ct, wsp_lst=wsp_lst)
    ct_lst = [constant_ct] * len(ct_lst) # make the ct constant
    ct_function = PowerCtTabular(u, p * 1000, 'w', ct_lst, ct_idle=constant_ct)
    return WindTurbine(
        name="AinslieTurbine",
        diameter=diameter,
        hub_height=hub_height,
        power_norm=power_norm,
        powerCtFunction=ct_function
    )

def get_discretized_grid(diameter: int,
                         x_start_factor: int, x_end_factor: int,
                         y_start_factor: int, y_end_factor: int,
                         grid_step_factor: float) -> HorizontalGrid:
    # TODO check the discretization
    #   - see Javier thesis, appendix B in particular)
    #   - see also datadriven wind turbine wake modelling via probabilistic ML (e.g. Fig. 3) to set these parameters
    x_range = np.arange(diameter*x_start_factor, diameter*x_end_factor, diameter*grid_step_factor)
    y_range = np.arange(diameter*y_start_factor, diameter*y_end_factor, diameter*grid_step_factor)
    return HorizontalGrid(x = x_range, y = y_range)

# using xarray netcdf for storing in a more efficient way (including compression)
def generate_wake_dataset(model, wind_speed: float, wind_direction: float,
                            wind_diameter: int, turbine_xs: list[int], turbine_ys: list[int],
                            horizontal_grid: HorizontalGrid, wind_turbine: WindTurbine
                            ) -> xr.Dataset:
    # the flow_map creates a warning for near-wake calculations
    sim_res = model(
        x=turbine_xs, y=turbine_ys, # wind turbine positions (setting also wt domain, i.e. the number of turbines)
        wd=wind_direction,          # Wind direction (None for default -> 0-360° in bins of 1°)
        ws=wind_speed,              # Wind speed (None for default -> 3-25 m/s in bins of 1 m/s)
        #yaw=0                      # TODO try to change this parameter?
        #h=None,                    # wind turbine heights (defaults to the heights defined in windTurbines)
        #type=0,                    # Wind turbine types
    )
    # create the input arrays
    ti = sim_res.TI.item()
    ti = xr.DataArray([ti], dims="ti").astype('float32')
    ct = wind_turbine.ct(ws=wind_speed)
    ct = xr.DataArray([ct], dims="ct").astype('float32')

    flow_map = xr.Dataset(
        sim_res.flow_map(horizontal_grid),
        coords={"ti": ti, "ct": ct} # adding the input variables as coords (and dims)
    )
    
    # removing h, wd and ws + all useless variables
    flow_map = flow_map.sel(h=flow_map["h"].item())\
        .sel(wd=wind_direction)\
        .sel(ws=wind_speed)\
        .drop_vars(["h", "wd", "ws", "WD", "P", "TI", "TI_eff"])

    # computing the wind_deficit as new data variable
    flow_map["wind_deficit"] = 1 - flow_map["WS_eff"] / flow_map["WS"]
    # scaling x and y according to the diameter
    flow_map["x"] = (flow_map["x"] / wind_diameter).astype('float32')
    flow_map["y"] = (flow_map["y"] / wind_diameter).astype('float32')
    flow_map = flow_map.rename({"x": "x:D", "y": "y:D"})

    flow_map = flow_map.astype({'WS_eff': 'float32', 'wind_deficit': 'float32', 'WS': 'int32'})
    return flow_map