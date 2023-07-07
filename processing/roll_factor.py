# ------------------------------------------------
# Name: roll_factor.py
# Author: Brian R. Greene and Robert M. Frost
# University of Oklahoma
# Created: 10 March 2023
# Purpose: Calculating 2d autocorrelation, 
# interpolating to a polar grid, and calculating
# roll factor.
# ------------------------------------------------
import sys
sys.path.append("/home/rfrost/LES-utils/")

from spec import autocorr_2d
import xarray as xr
import numpy as np
import yaml
from dask.diagnostics import ProgressBar
# ---------------------------------
# Read in settings
# ---------------------------------
with open('/home/rfrost/processing/roll_factor.yaml', 'r') as f:
    settings = yaml.safe_load(f)
# ---------------------------------
# Calculate autocorrelation
# ---------------------------------
# read in volumetric simulation output
dnc = settings['dnc']
t0 = settings['t0']
t1 = settings['t1']
dt = settings['dt']
delta_t = settings['delta_t']
# load in volumetric output
df = load_full(dnc, t0, t1, dt, delta_t)
Lx = max(df.x)
# compute autocorrelation if desired
if settings['autocorr']:
    # list of variables to have autocorrelation calculated over
    var = ["w"]
    # compute 2D autocorrelation
    autocorr_2d(dnc, df, var, timeavg=False)

# ---------------------------------
# Convert to polar coords
# ---------------------------------
if settings['polar']:
    # calculate t'w'
    df["tw_cov_res"] = xr.cov(df.theta, df.w, dim=("x", "y")).compute()
    # calculate zi
    idx = df.tw_cov_res.argmin(axis=1)
    jz = np.zeros(df.time.size)
    for jt in range(0, df.time.size):
        # find jz for defined z/zi
        jz[jt] = abs(df.z/df.z[idx].isel(time=jt) - settings['height']).argmin()
    
    # read in autocorrelation dataset
    r = xr.open_dataset(f"{dnc}R_2d.nc")
    for t in range(df.time.size):
        w = r.w.isel(z=jz[t].astype(int))
    # time average autocorrelation
    if settings['coarsen']:
        w = w.coarsen(time=settings['avg'], boundary="trim").mean()
    
    # calculate 2d arrays of theta=theta(x,y), r=r(x,y)
    theta = np.arctan2(w.y, w.x)
    r = (w.x**2. + w.y**2.) ** 0.5
    # grab sizes of x and y dimensions for looping
    nx, ny = w.x.size, w.y.size
    # set up bin centers for averaging
    ntbin = settings['ntbin']
    nrbin = settings['nrbin']
    rbin = np.linspace(0, Lx//2, nrbin)
    tbin = np.linspace(-np.pi, np.pi, ntbin)
    # intiialize empty arrays for storing values and counter for normalizing
    wall, count = [np.zeros((w.time.size, ntbin, nrbin), dtype=np.float64) for _ in range(2)]
    # set up dimensional arrays to store roll factor stats
    Rmax_r = np.zeros((w.time.size, nrbin))
    rbin_zi = np.zeros((w.time.size, nrbin))
    RR = np.zeros((w.time.size))

    print("Rotating to polar coordinates")
    # loop over x, y pairs
    for jx in range(nx):
        for jy in range(ny):
            # find nearest bin center for each r(jx,jy) and theta(jx,jy)
            jr = abs(rbin - r.isel(x=jx,y=jy).values).argmin()
            jt = abs(tbin - theta.isel(x=jx,y=jy).values).argmin()
            for t in range(0, w.time.size):
                # store w[jt,jr] in wall, increment count
                wall[t,jt,jr] += w[t,jx,jy]
                count[t,jt,jr] += 1
    # set up dimensial array for wmean
    wmean = np.zeros((w.time.size, ntbin, nrbin))
    for t in range(0, w.time.size):
        # normalize wall by count
        wmean[t,:,:] = wall[t,:,:] / count[t,:,:]
    
    # convert polar Rww to xarray data array
    w_pol = xr.DataArray(data=wmean,
                        coords=dict(time=w.time, theta=tbin, r=rbin),
                        dims=["time", "theta", "r"])
    # output polar Rww data
    w_pol.to_netcdf(f"{settings['dnc']}R_pol_zzi{int(settings['height']*100)}_TRUE.nc")

# ---------------------------------
# Calculate roll factor
# ---------------------------------
if settings['roll']:
    print("Calculating roll factor")
    # loop over time
    for t in range(0, w.time.size):
        # calculate roll factor
        Rmax_r[t,:] = np.nanmax(wmean[t,:,:], axis=0) - np.nanmin(wmean[t,:,:], axis=0)
        rbin_zi[t,:] = rbin / df.z[idx].isel(time=t).values
        RR[t] = np.nanmax(Rmax_r[t, rbin_zi[t,:] >= 0.5])
    print("Roll factor calculation complete!")
    # create xarray data array
    roll = xr.DataArray(data=RR,
                        coords=dict(time=w.time),
                        dims=["time"])
    
    # save data
    if settings['coarsen']:
        fsave = f"{settings['dnc']}rollfactor_zzi{int(settings['height']*100)}_{int(settings['avg']*dt/60)}min.nc"
    else:
        fsave = f"{settings['dnc']}rollfactor_zzi{int(settings['height']*100)}_raw.nc"
    # output to netCDF
    print(f"Saving file: {fsave}")
    with ProgressBar():
        roll.to_netcdf(fsave, mode="w")
    print("Finished!")