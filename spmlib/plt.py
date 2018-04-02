#!/usr/bin/env python
# coding: utf-8
"""
Tools for plotting data from Dimitris' global high resolution
model once read into xarray Dataset.
"""

import numpy as np
import xarray as xr
import gsw
import scipy as sp
import gvpy as gv
from scipy import interpolate


def distance(lon, lat, p=np.array([0]), axis=-1):
    """
    From gsw: Great-circle distance in m between lon, lat points.
    Parameters
    ----------
    lon, lat : array-like, 1-D or 2-D (shapes must match)
        Longitude, latitude, in degrees.
    p : array-like, scalar, 1-D or 2-D, optional, default is 0
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    axis : int, -1, 0, 1, optional
        The axis or dimension along which *lat and lon* vary.
        This differs from most functions, for which axis is the
        dimension along which p increases.
    Returns
    -------
    distance : 1-D or 2-D array
        distance in meters between adjacent points.
    """
    earth_radius = 6371e3

    if not lon.shape == lat.shape:
        raise ValueError('lon, lat shapes must match; found %s, %s'
                          % (lon.shape, lat.shape))
    if not (lon.ndim in (1, 2) and lon.shape[axis] > 1):
        raise ValueError('lon, lat must be 1-D or 2-D with more than one point'
                         ' along axis; found shape %s and axis %s'
                          % (lon.shape, axis))
    if lon.ndim == 1:
        one_d = True
        lon = lon[np.newaxis, :]
        lat = lat[np.newaxis, :]
        axis = -1
    else:
        one_d = False

    one_d = one_d and p.ndim == 1

    if axis == 0:
        indm = (slice(0, -1), slice(None))
        indp = (slice(1, None), slice(None))
    else:
        indm = (slice(None), slice(0, -1))
        indp = (slice(None), slice(1, None))

    if np.all(p == 0):
        z = 0
    else:
        lon, lat, p = np.broadcast_arrays(lon, lat, p)

        p_mid = 0.5 * (p[indm] + p[indp])
        lat_mid = 0.5 * (lat[indm] + lat[indp])

        z = z_from_p(p_mid, lat_mid)

    lon = np.radians(lon)
    lat = np.radians(lat)

    dlon = np.diff(lon, axis=axis)
    dlat = np.diff(lat, axis=axis)

    a = ((np.sin(dlat / 2)) ** 2 + np.cos(lat[indm]) *
         np.cos(lat[indp]) * (np.sin(dlon / 2)) ** 2)

    angles = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = (earth_radius + z) * angles

    if one_d:
        distance = distance[0]

    return distance


def model_bathy_section(lon,lat,d,res=1,ext=0):
    """Extract Samoan Passage bathymetry along sections defined by lon/lat
    coordinates.

    Parameters
    ----------
    lon : arraylike
        Longitude position along section
    lat : arraylike
        Latitude position along section
    d : Dataset
        Model Dataset to access lon/lat/BottomDepth
    res : float
        Bathymetry resolution
    ext : float
        Extension on both sides in km. Set to 0 for no extension

    Returns
    -------
    out : dict
        Dictionary with output variables
    """

    # Make sure lon and lat have the same dimensions
    assert lon.shape==lat.shape, 'lat and lon must have the same size'
    # Make sure lon and lat have at least 3 elements
    assert len(lon)>1 and len(lat)>1, 'lon/lat must have at least 2 elements'

    # Load bathymetry
#     plon = b['lon']
    plon = d.lon.values
#     plat = b['lat']
    plat = d.lat.values
#     ptopo = -b['merged']
    ptopo = d.BottomDepth.values

    # 2D interpolation function used below
    f = interpolate.f = interpolate.RectBivariateSpline(plat,plon,ptopo)

    # calculate distance between original points
    dist = np.cumsum(distance(lon, lat, np.array([0]))/1000)
    # distance 0 as first element
    dist = np.insert(dist,0,0)

    # Extend lon and lat if ext>0
    if ext:
        '''
        Linear fit to start and end points. Do two separate fits if more than
        4 data points are give. Otherwise fit all points together.

        Need to calculate distance first and then scale the lon/lat extension with distance.
        '''
        if len(lon)<5:
            # only one fit for 4 or less data points
            dlo = np.abs(lon[0]-lon[-1])
            dla = np.abs(lat[0]-lat[-1])
            dd  = np.abs(dist[0]-dist[-1])
            # fit either against lon or lat, depending on orientation of section
            if dlo>dla:
                bfit = np.polyfit(lon,lat,1)
                # extension expressed in longitude (scale dist to lon)
                lonext = 1.1*ext/dd*dlo
                if lon[0]<lon[-1]:
                    elon = np.array([lon[0]-lonext,lon[-1]+lonext])
                else:
                    elon = np.array([lon[0]+lonext,lon[-1]-lonext])
                blat = np.polyval(bfit,elon)
                nlon = np.hstack((elon[0],lon,elon[-1]))
                nlat = np.hstack((blat[0],lat,blat[-1]))
            else:
                bfit = np.polyfit(lat,lon,1)
                # extension expressed in latitude (scale dist to lat)
                latext = 1.1*ext/dd*dla
                if lat[0]<lat[-1]:
                    elat = np.array([lat[0]-lonext,lat[-1]+lonext])
                else:
                    elat = np.array([lat[0]+lonext,lat[-1]-lonext])
                blon = np.polyval(bfit,elat)
                nlon = np.hstack((blon[0],lon,blon[-1]))
                nlat = np.hstack((elat[0],lat,elat[-1]))

        else:
            # one fit on each side of the section as it may change direction
            dlo1 = np.abs(lon[0]-lon[2])
            dla1 = np.abs(lat[0]-lat[2])
            dd1  = np.abs(dist[0]-dist[2])
            dlo2 = np.abs(lon[-3]-lon[-1])
            dla2 = np.abs(lat[-3]-lat[-1])
            dd2  = np.abs(dist[-3]-dist[-1])

            # deal with one side first
            if dlo1>dla1:
                bfit1 = np.polyfit(lon[0:3],lat[0:3],1)
                lonext1 = 1.1*ext/dd1*dlo1
                if lon[0]<lon[2]:
                    elon1 = np.array([lon[0]-lonext1,lon[0]])
                else:
                    elon1 = np.array([lon[0]+lonext1,lon[0]])
                elat1 = np.polyval(bfit1,elon1)
            else:
                bfit1 = np.polyfit(lat[0:3],lon[0:3],1)
                latext1 = 1.1*ext/dd1*dla1
                if lat[0]<lat[2]:
                    elat1 = np.array([lat[0]-latext1,lat[0]])
                else:
                    elat1 = np.array([lat[0]+latext1,lat[0]])
                elon1 = np.polyval(bfit1,elat1)

            # now the other side
            if dlo2>dla2:
                bfit2 = np.polyfit(lon[-3:],lat[-3:],1)
                lonext2 = 1.1*ext/dd2*dlo2
                if lon[-3]<lon[-1]:
                    elon2 = np.array([lon[-1],lon[-1]+lonext2])
                else:
                    elon2 = np.array([lon[-1],lon[-1]-lonext2])
                elat2 = np.polyval(bfit2,elon2)
            else:
                bfit2 = np.polyfit(lat[-3:],lon[-3:],1)
                latext2 = 1.1*ext/dd2*dla2
                if lat[-3]<lat[-1]:
                    elat2 = np.array([lat[-1],lat[-1]+latext2])
                else:
                    elat2 = np.array([lat[-1],lat[-1]-latext2])
                elon2 = np.polyval(bfit2,elat2)

            # combine everything
            nlon = np.hstack((elon1[0],lon,elon2[1]))
            nlat = np.hstack((elat1[0],lat,elat2[1]))

        lon = nlon
        lat = nlat

    # Original points (but including extension if there are any)
    olat = lat
    olon = lon

    # Interpolated points
    ilat = []
    ilon = []

    # Output dict
    out = {}

    # calculate distance between points
    dist2 = distance(lon,lat,np.array([0]))/1000
    dist2 = dist2
    cdist2 = np.cumsum(dist2)
    cdist2 = np.insert(cdist2,0,0)

    # Create evenly spaced points between lon and lat
    # for i = 1:length(lon)-1
    for i in np.arange(0,len(lon)-1,1):

        n = dist2[i]/res

        dlon = lon[i+1]-lon[i]
        if not dlon==0:
            deltalon = dlon/n
            lons = np.arange(lon[i],lon[i+1],deltalon)
        else:
            lons = np.tile(lon[i],np.ceil(n))
        ilon = np.hstack([ilon,lons])

        dlat = lat[i+1]-lat[i]
        if not dlat==0:
            deltalat = dlat/n
            lats = np.arange(lat[i],lat[i+1],deltalat)
        else:
            lats = np.tile(lat[i],np.ceil(n))
        ilat = np.hstack([ilat,lats])

        if i==len(lon)-1:
            ilon = np.append(ilon,olon[-1])
            ilat = np.append(ilat,olat[-1])

        if i==0:
            odist = np.array([0,dist[i]])
        else:
            odist = np.append(odist,odist[-1]+dist2[i])

    # Evaluate the 2D interpolation function
    itopo = f.ev(ilat,ilon)
    idist = np.cumsum(distance(ilon,ilat,np.array([0]))/1000)
    # distance 0 as first element
    idist = np.insert(idist,0,0)

    out['ilon'] = ilon
    out['ilat'] = ilat
    out['idist'] = idist
    out['itopo'] = itopo

    out['otopo'] = f.ev(olat,olon)
    out['olat'] = olat
    out['olon'] = olon
    out['odist'] = cdist2

    out['res'] = res
    out['ext'] = ext

    # Extension specific
    if ext:
        # Remove offset in distance between the two bathymetries
        out['olon']  = out['olon'][1:-1]
        out['olat']  = out['olat'][1:-1]
        out['otopo']    = out['otopo'][1:-1]
        # set odist to 0 at initial lon[0]
        offset = out['odist'][1]-out['odist'][0]
        out['odist'] = out['odist'][1:-1]-offset
        out['idist'] = out['idist']-offset

    return out


@xr.register_dataarray_accessor('spplt')
class ModelVar(object):
    """
    This class adds methods to xarray dataarrays
    (individual variables).
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def plot(self, ti, **kwargs):
        # todo: make sure its only one time step
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 5))
        h = gv.figure.pcm(self._obj.dist, self._obj.z, self._obj,
                          ax=ax, **kwargs)
        ax.set(ylim=(5300, 4000))
        plt.colorbar(h)
        return h

    def extract_section(self, lon, lat, spm, res=1, ext=0):
        z = spm.z.values
        out = model_bathy_section(lon, lat, spm, res=res, ext=ext)
        loni = out['ilon']
        lati = out['ilat']
        # here we must repeat first value*len(z), second value*len(z)
        # and so forth
        lonp = np.tile(loni, (len(z),1))
        lonp = np.reshape(lonp.T,(-1,))
        latp = np.tile(lati, (len(z),1))
        latp = np.reshape(latp.T,(-1))
        zp = np.tile(z,len(loni))
        interp_points = np.vstack((zp, latp, lonp)).T
        from scipy.interpolate import interpn
        test = interpn((self._obj.z.values, self._obj.lat.values, self._obj.lon.values),
                       np.ma.masked_equal(self._obj.values, 0), interp_points,
                       method='linear', bounds_error=True, fill_value=np.nan)
        section = np.reshape(test,(len(loni), len(z))).T
        out['z'] = z
        return out, section

