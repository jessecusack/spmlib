#!/usr/bin/env python
# coding: utf-8
"""
Tools for reading binary model data from Dimitris' global high resolution
model into an xarray Dataset.
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
from dask import delayed

class LLCRegion:
    """ A class that describes a region MITgcm Lat-Lon-Cap setup.
     Initially forked from https://github.com/crocha700/llctools
     but then changed so much that it was easier to simply integrate this here
     than loading it as its own module.
    """

    def __init__(self,
                 grid_dir = None,
                 data_dir = None,
                 Nlon = None,
                 Nlat = None,
                 Nz   = None,
                 tini = 228096,
                 tend = 1485648,
                 dtype = np.dtype('>f4')
                ):

        self.grid_dir = grid_dir     # Grid directory
        self.data_dir = data_dir     # Parent data directory
        self.Nlon = Nlon             # Number of longitude points in the regional subset
        self.Nlat = Nlat             # Number of latitude points in the regional subset
        self.Nz = Nz                 # Number of vertical levels in the regional subset

        self.tini = _parse_model_time(tini) # Initial time
        self.tend = _parse_model_time(tend) # End time
        self.dtype = dtype           # Data type (default >f4, mean float single precision)

        self.grid_size = str(Nlon)+ 'x' + str(Nlat)  # grid size string
        self.grid_size3d = str(Nlon)+ 'x' + str(Nlat)+ 'x' + str(Nz)  # grid size string

    def load_grid(self):
        """Load various grid parameters"""
        self.xc = np.memmap(self.grid_dir+'/XC_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.yc = np.memmap(self.grid_dir+'/YC_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.z = np.memmap(self.grid_dir+'/thk90',dtype=self.dtype,
                            shape=(self.Nz), mode='r')

        self.hb = np.memmap(self.grid_dir+'/Depth_'+self.grid_size,
                            dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                            mode='r')

        self.hfacc = np.memmap(self.grid_dir+'/hFacC_'+self.grid_size3d,
                            dtype=self.dtype,shape=(self.Nz,self.Nlat,self.Nlon),
                            mode='r')

        self.drf = np.memmap(self.grid_dir+'/drf',dtype=self.dtype,
                            shape=(self.Nz), mode='r')

        self.rac = np.memmap(self.grid_dir+'/RAC_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.raz = np.memmap(self.grid_dir+'/RAZ_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.xg = np.memmap(self.grid_dir+'/XG_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.yg = np.memmap(self.grid_dir+'/YG_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
 
        self.dxc = np.memmap(self.grid_dir+'/DXC_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
 
        self.dyc = np.memmap(self.grid_dir+'/DYC_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

        self.dxg = np.memmap(self.grid_dir+'/DXG_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')       

        self.dyg = np.memmap(self.grid_dir+'/DYG_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')       
 
        self.dxv= np.memmap(self.grid_dir+'/DXV_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
 
        self.dyu = np.memmap(self.grid_dir+'/DYU_'+self.grid_size,
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
 

    def load_2d_data(self, fni):
        return np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon), mode='r')

    def load_3d_data(self, fni):
        with open(fni, 'rb') as f:
            data = np.memmap(f, dtype=self.dtype,
                         shape=(self.Nz, self.Nlat,self.Nlon), mode='r')
        return data


def generate_model_class(grid_dir, data_dir, Nlon=936, Nlat=1062, Nz=90):
    """
    Wrapper function for generating the LLCRegion object describing the
    model region. The wrapper automatically reads the grid information.
    Default values for grid size are for the Samoan Passage box (Box 12
    in Dimitris' notation).

    Parameters
    ----------
    grid_dir : str
        Path to grid files
    data_dir : str
        Path to data files
    Nlon : int
        Number of grid points in the zonal direction
    Nlat : int
        Number of grid points in the meridional
    Nz : int
        Number of grid points in the vertical

    Returns
    -------
    m : LLCRegion model class
    """
    m = LLCRegion(grid_dir=grid_dir, data_dir=data_dir,
                      Nlon=Nlon, Nlat=Nlat, Nz=Nz)
    print('loading grid...')
    m.load_grid()
    print(m.grid_size3d)
    return m


def create_dataset(m, timestep, var='all', chunks=(10, 300, 300)):
    """
    Create xarray Dataset from binary model data
    for one time step. This also incorporates all model
    grid information and dimensions, regardless of the variable selected.

    Parameters
    ----------
    m : LLCRegion
        Model class generated with LLCRegion()
    var : str, optional
        Variable to be read. Defaults to 'all', but only one variable,
        e.g. 'v', or a list of variabbles, e.g. ['t', 'v']
        can be selected here instead.
    chunks : tuple, optional
        Chunk size for dask. Defaults to (10, 300, 300)

    Returns
    -------
    ds : xarray Dataset
        Dataset
    """

    if var is 'all':
        vars = _model_variables
    else:
        vars = {k: _model_variables[k] for k in var}
        # vars = {var: _model_variables[var]}

    # reduce xc/yc, xg/yg to 1d vector
    lon, lat = _reduce_2d_coords(m.xc, m.yc)
    xc, yc = _reduce_2d_coords(m.xc, m.yc)
    xg, yg = _reduce_2d_coords(m.xg, m.yg)

    # calculate Zu, Zl, Zp1 (combination of Zu, Zl)
    tmp = m.drf
    tmp = np.insert(tmp, 0, 0)
    Zp1 = np.cumsum(tmp)
    Zl = Zp1[0:-1]
    Zu = Zp1[1::]

    # calculate drc
    drc = np.diff(m.z)
    drc = np.insert(drc, 0, m.z[0])
    drc = np.append(drc, Zp1[-1]-m.z[-1])

    # generate xarray dataset with only grid information first
    ds = xr.Dataset(coords={'xc': (['xc'], xc, {'axis': 'X'}),
                        'yc': (['yc'], yc, {'axis': 'Y'}),
                        'lon': (['xc'], xc, {'axis': 'X'}),
                        'lat': (['yc'], yc, {'axis': 'Y'}),
                        'dxc': (['yc', 'xg'], m.dxc),
                        'dyc': (['yg', 'xc'], m.dxc),
                        'xg': (['xg'], xg, {'axis': 'X', 'c_grid_axis_shift': -0.5}),
                        'yg': (['yg'], yg, {'axis': 'Y', 'c_grid_axis_shift': -0.5}),
                        'dxg': (['yg', 'xc'], m.dxg),
                        'dyg': (['yc', 'xg'], m.dyg),
                        'dxv': (['yg', 'xg'], m.dxv),
                        'dyu': (['yg', 'xg'], m.dyu),
                        'z': (['z'], m.z, {'axis': 'Z'}, {'axis': 'Z'}),
                        'zl': (['zl'], Zl, {'axis': 'Z', 'c_grid_axis_shift': -0.5}),
                        'zu': (['zu'], Zu, {'axis': 'Z', 'c_grid_axis_shift': +0.5}),
                        'zp1': (['zp1'], Zp1, {'axis': 'Z', 'c_grid_axis_shift': (-0.5,0.5)}),
                        'drc': (['zp1'], drc, {'axis': 'Z'}),
                        'drf': (['z'], m.drf, {'axis': 'Z'}),
                        'ra': (['yc', 'xc'], m.rac),
                        'raz': (['yg', 'xg'], m.raz),
                        'depth': (['yc', 'xc'], m.hb),
                        'hfacc': (['z', 'yc', 'xc'], m.hfacc)})

    # define dictionary that will hold dask arrays
    d = {}
    # read all variables into a dict with dask arrays
    for k, v in vars.items():
        filename = m.data_dir+'{}/{:010d}_{}'.format(v, timestep, v)+\
                   '_10609.6859.1_936.1062.90'
        exist = _check_file_exists(filename)
        # account for funky V file names
        if ~exist & (v=='V'):
            filename = m.data_dir+'{}/{:010d}_{}'.format(v, timestep, v)+\
                   '_10609.6858.1_936.1062.90_Neg'
            exist = _check_file_exists(filename)
        d[k] = da.from_delayed(delayed(m.load_3d_data)(filename), (m.Nz, m.Nlat, m.Nlon), m.dtype)
        d[k] = d[k].rechunk(chunks)


    for k, v in d.items():
        ds[k] = (_grid_association[k], v)
    del d

    # add 2d variables
    if var is 'all':
        vars2d = _model_2dvariables
        d = {}
        for k, v in vars2d.items():
            filename = m.data_dir+'{}/{:010d}_{}'.format(v, timestep, v)+\
                       '_10609.6859.1_936.1062.1'
            exist = _check_file_exists(filename)
            d[k] = da.from_delayed(delayed(m.load_2d_data)(filename), (m.Nlat, m.Nlon), m.dtype)
            d[k] = d[k].rechunk(chunks[1:])
        for k, v in d.items():
            ds[k] = (_grid_association[k], v)
        del d

    return ds


def create_dataset_timeseries(m, time_range, var='all', chunks=(10, 300, 300),
                              verbose=True):
    """
    Create xarray Dataset from binary model data files
    for more than one time step.

    Parameters
    ----------
    m : LLCRegion
        Model class generated with LLCRegion
    time_range : list
        List with time steps to access model files
    var : str, optional
        Variable to be read. Defaults to all, but only one variable,
        e.g. 'v', can be selected here.
    chunks : tuple, optional
        Chunk size for dask. Defaults to (10, 300, 300)

    Returns
    -------
    d : xarray Dataset
        Dataset
    """

    ds = []
    for ti in time_range:
        tmp = create_dataset(m, ti, var=var, chunks=chunks)
        ds.append(tmp)
        tmp.close()
        del tmp

    if var is 'all':
        d = xr.concat(ds, dim='time',
                      data_vars=list(_model_variables.keys()) + list(_model_2dvariables),
                      coords='all')
    else:
        d = xr.concat(ds, dim='time',
                      data_vars=list(var),
                      coords='all')
    del ds

    time = _generate_time_vector(time_range)
    d.coords['time'] = (('time'), time, {'axis': 'T'})

    # Somehow xarray expands all coordinates that are not dimensions along the
    # time vector. Not sure how to suppress this behaviour, so we simply remove
    # this here
    vv = ['lon', 'lat', 'dxc', 'dyc', 'dxv', 'drc', 'drf',
          'ra', 'hfacc']
    for vi in vv:
        d[vi] = d[vi].isel(time=0, drop=True)

    if verbose:
        print('chunks')
        print('======')
        for k, v in d.chunks.items():
            print('{}: {}'.format(k, v))

    return d


def _reduce_2d_coords(xin, yin):
    xc = xin
    xc1 = xin[:, 0]
    if xc1[0] == xc1[-1]:
        xc = xc[0, :]
        yc = yin[:, 0]
    else:
        xc = xc[:, 0]
        yg = yin[0, :]
    return xc, yc


def _guess_delta_t(time_range):
    """Guess time period between model snapshots based on
    time range.
    """
    median_dt = np.median(np.diff(time_range))
    if median_dt == 432:
        delta_t = '3H'
    elif median_dt == 144:
        delta_t = '1H'
    return delta_t


def _parse_model_time(ts):
    basetime = np.datetime64('{:04d}-{:02d}-{:02d}'.format(_model_startyr,
                                                           _model_startmo,
                                                           _model_startdy))
    # out = basetime + np.timedelta64(ts * _model_deltat, 's')
    out = pd.Timestamp(basetime + np.timedelta64(np.int(ts) * _model_deltat, 's'))
    return out


def _timestamp_to_model_time(tsin):
    basetime = np.datetime64('{:04d}-{:02d}-{:02d}'.format(_model_startyr,
                                                           _model_startmo,
                                                           _model_startdy))
    delta = tsin-basetime
    deltas = np.int((delta.seconds + delta.days*24*3600)/25)
    return deltas


def _generate_time_vector(time_range):
    return [_parse_model_time(np.int(ti)) for ti in time_range]


def _find_model_time_steps(tini, tend):
    """Generate a list of model times between a given start and end time.
    """
    return list_of_model_times


def find_model_timestamps(tini, tend):
    """Generate a list of model times between a given start and end time.

    Note that we have model data from
    2011-11-15 00:00:00
    through
    2012-11-14 00:00:00

    Parameters
    ----------
    tini : str
        Start date/time
    tend : str
        End date/time

    Returns
    -------
    time_range : list
        List with model timestamps to use in create_dataset_timeseries().
    """

    # calculate first and last timestamp
    mfirst = 228096
    mlast = 1489536
    tsfirst, tslast = [_parse_model_time(ti) for ti in [mfirst, mlast]]
    
    # generate range of timestamps at 3H interval
    tsall = pd.date_range(start=tsfirst, end=tslast, freq='3H')
    
    # calculate timestamps for tini, tend
    tsini, tsend = [pd.Timestamp(ti) for ti in [tini, tend]]
    
    # find timestamps within tini, tend
    mask = ((tsall>=tsini) & (tsall<=tsend))
    # print time range to screen
    print(('----------------------------------------\n'
           'generating list of model timestamps from\n{}\nto\n{}\n'
            '----------------------------------------').format(
            tsall[mask].min(), tsall[mask].max()))
    
    # generate list of model timesteps for reading datafiles
    time_range = [_timestamp_to_model_time(ti) for ti in tsall[mask]]
    return time_range


def _check_file_exists(filename):
    """Test if a file exists

    Parameters
    ----------
    filename : str
        Path and filename to check

    Returns
    -------
    exit : bool
        True if file exists, False otherwise
    """

    try:
        with open(filename) as file:
            pass
        exist = True
    except IOError:
        print('{} does not exist'.format(filename))
        exist = False
    return exist

# Dict with all variables that we will load into the dataset
_model_variables = {'u': 'U', 'v': 'V', 'w': 'W', 's': 'Salt', 't': 'Theta'}
_model_2dvariables = {'eta': 'Eta', 'phibot': 'PhiBot'}

# The grids associated with each model variable
_grid_association = {'u': ['z', 'yc', 'xg'],
                     'v': ['z', 'yg', 'xc'],
                     'w': ['zl', 'yc', 'xc'],
                     's': ['z', 'yc', 'xc'],
                     't': ['z', 'yc', 'xc'],
                     'eta': ['yc', 'xc'],
                     'phibot': ['yc', 'xc']}

# Model reference time
_model_startyr = 2011
_model_startmo = 9
_model_startdy = 10
# One model timestep corresponds to 25s
_model_deltat = 25
# The first model snapshot we have is at t0:
_model_t0 = 228096
