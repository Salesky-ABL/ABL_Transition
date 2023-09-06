# ------------------------------------------------
# Name: runtran_rot.py
# Author: Robert M. Frost
# University of Oklahoma
# Created: 23 August 2023
# Purpose: Script for running functions in 
# tranutils.py and LESutils.py. Testing new 
# method of calculating roll factor and integral 
# length scales along the mean wind angle
# ------------------------------------------------
import sys
sys.path.append("/home/rfrost/LES-utils/")

import xarray as xr
import numpy as np
from spec import autocorr_2d
from LESutils import sim2netcdf, load_full, nc_rotate
from tranutils import calc_stats_tran, polar_grid, roll_factor, length_scales
# ---------------------------------
# Parameters for script
# --------------------------------- 

# directory for raw simulation output
dout = "/home/rfrost/simulations/abl_transition/full_step_9/output/"
# directory for netCDF files to be read and saved
dnc = "/home/rfrost/simulations/nc/full_step_9/"
# simulation resolution
nx = 160
ny = 160
nz = 160
# domain size in meters
Lx = 12000
Ly = 12000
Lz = 2000
# scales from LES code (uscale, tscale)
scales = [0.4, 300]
# start and end timesteps to be read in
t0 = 576000
t1 = 1152000
# output frequency in timesteps
dt = 1000
# temporal resolution in seconds
delta_t = 0.05

# flag to run nc_rotate
rotate = False
# flag to run autocorr_2d
autocorr = False
# flag to run ls_rot
ls = True
# ---------------------------------
# Rotated functions
# --------------------------------- 

def ls_rot(dnc, t0, t1, dt):
    """Calculate integral length scales normal to and along alpha

    :param str dnc: absolute path to directory for saving output netCDF files
    :param int t0: first timestep for stats to be computed
    :param int t1: final timestep for stats to be computed
    :param int dt: number of timesteps between files to load
    :param float height: dimensionless height
    :param int avg_method: 0 = no average, 1 = rolling average, 2 = coarsen
    :param int avg_time: number of timesteps to average over (6 step default)
    """

    # read in rotated autocorrelation
    Rww = xr.open_dataarray(f"{dnc}R_2d_rot.nc")
    # remove negative lags
    Rww = Rww.where(Rww.x >= 0, drop=True)
    Rww = Rww.where(Rww.y >= 0, drop=True)

    # number of time points
    ntime = int((t1 - t0) / dt + 1)
    # number of z levels
    nz = int(Rww.z.size)

    # arrays to hold indice of first negative value
    tazero, ta2zero, Lwa, Lwa2 = [np.empty((ntime, nz)) for _ in range(4)]

    # load data into memory
    Rww.load()

    # loop over time and z
    for jt in range(ntime):
        for jz in range(nz):
            # find the indices of the first zero
            ta_array = np.where(Rww[jt,:,0,jz] <= 0)[0]
            ta2_array = np.where(Rww[jt,0,:,jz] <= 0)[0]
            # exclude nans
            if ta_array.size > 0:
                tazero[jt,jz] = ta_array[0]
            else:
                tazero[jt,jz] = np.argmin(Rww[jt,:,0,jz].values)
            if ta2_array.size > 0:
                ta2zero[jt,jz] = ta2_array[0]
            else:
                ta2zero[jt,jz] = np.argmin(Rww[jt,0,:,jz].values)

    # loop over time and z
    for jt in range(ntime):
        for jz in range(nz):
            # calculate length scale along alpha
            Lwa[jt,jz] = Rww.isel(time=jt, y=0, z=jz, x=range(0,int(tazero[jt,jz]))).integrate("x")
            # calculate length scale normal to alpha
            Lwa2[jt,jz] = Rww.isel(time=jt, x=0, z=jz, y=range(0,int(ta2zero[jt,jz]))).integrate("y")

    # back to xarray
    Lw = xr.Dataset(
        data_vars=dict(
            rolls = (["time","z"], Lwa),
            normal = (["time","z"], Lwa2)),
        coords=dict(
            time=Rww.time,
            z=Rww.z)
        )

    # save out data
    fsave = f"{dnc}{t0}_{t1}_length_scale_rot.nc"
    Lw.to_netcdf(fsave, mode="w")
    return
# ------------------- 

def roll_factor(dnc, stats):
    """Calculate roll factor using polar grid autocorrelation of w
    
    :param str dnc: absolute path to directory for saving output netCDF files
    :param Dataset stats: output from calc_stats_tran
    """
    # read in rotated autocorr
    Rww = xr.open_dataarray(f"{dnc}R_2d_rot.nc")
    # extract important stuff
    ntime = Rww.time.size

    # calculate roll factor
    roll = np.nanmax(Rww, axis=1) - np.nanmin(Rww, axis=2)
    
    fsave = f"{dnc}rollfactor_rot.nc"

    # output to netCDF
    print(f"Saving file: {fsave}")
    with ProgressBar():
        roll.to_netcdf(fsave, mode="w")
    print("Finished!")
    return roll
# ---------------------------------
# Calling functions to run
# --------------------------------- 

# rotate coords along wind angle
if rotate:
    print("Begin nc_rotate...")
    nc_rotate(dnc, t0, t1, dt)
    print("Finished nc_rotate!")

# autocorr_2d
if autocorr:
    print("Begin autocorr_2d...")
    # load in volumetric output
    df = load_full(dnc, t0, t1, dt, delta_t, rotate=True)
    # list of variables to have autocorrelation calculated over
    var = ["w"]
    # compute 2D autocorrelation
    autocorr_2d(dnc, df, var, timeavg=False)
    print("Finished autocorr_2d!")

# ls_rot
if ls:
    print("Begin ls_rot...")
    ls_rot(dnc, t0, t1, dt)
    print("Finished ls_rot!")